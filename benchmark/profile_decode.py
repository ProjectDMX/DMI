"""Nsight Systems profiling script: decode with CUDA graphs, baseline vs ring_null.

Run with:
  nsys profile --trace=cuda,nvtx,osrt \
       --output /tmp/profile_decode \
       conda run -n ring_offload python -m benchmark.profile_decode

Then analyse:
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_memtransfer_sum /tmp/profile_decode.nsys-rep
  nsys export --type sqlite /tmp/profile_decode.nsys-rep  &&  python -m benchmark.profile_decode --analyse /tmp/profile_decode.sqlite
"""

from __future__ import annotations
import argparse
import sys
import time
import uuid


# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
MODEL_ID    = "Qwen/Qwen3-4B"
BATCH       = 4
PROMPT_LEN  = 1
NEW_TOKENS  = 64   # short enough for a clean profile, enough steps to see pattern
WARMUP      = 3
MEASURE     = 2    # NVTX-annotated measured iterations per mode
RING_TASK_ENTRIES = 65536
RING_PAYLOAD_MB   = 4096
RING_PINNED_MB    = 4096
RING_CHUNK_KB     = 4096


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_inputs(tokenizer, device):
    import torch
    ids  = tokenizer.encode("The quick brown fox") * 10
    ids  = ids[:PROMPT_LEN]
    rows = [torch.tensor(ids, dtype=torch.long) for _ in range(BATCH)]
    input_ids = torch.stack(rows).to(device)
    return input_ids, torch.ones_like(input_ids)


def _make_ring_engine_cfg():
    from monitoring._native_engine import RingConfig  # type: ignore
    rc = RingConfig()
    rc.task_ring_entries  = RING_TASK_ENTRIES
    rc.payload_ring_bytes = RING_PAYLOAD_MB * 1024 * 1024
    rc.chunk_bytes        = RING_CHUNK_KB   * 1024
    rc.pinned_staging_bytes  = RING_PINNED_MB  * 1024 * 1024
    return rc


def _make_monitoring_cfg():
    from monitoring import MonitoringConfig  # type: ignore
    from monitoring.config import CaptureSchedule  # type: ignore
    return MonitoringConfig(
        schedule=CaptureSchedule(capture_prefill=True, capture_decode=True),
    )


class _NullHostEngine:
    def start(self):                 pass
    def stop(self, *a, **kw):        pass
    def join(self, *a, **kw):        return True
    def close_input(self):           pass
    def request_abort(self):         pass
    def failures(self):              return []
    def raise_if_failed(self):       pass
    def submit_direct(self, *a, **kw): pass


def _make_null_engine(model_id: str):
    from monitoring import MonitoringEngine  # type: ignore
    engine = MonitoringEngine(
        config=_make_monitoring_cfg(),
        model_id=model_id, host_engine=_NullHostEngine(),
    )
    engine.enable_ring_transport(_make_ring_engine_cfg())
    return engine


def _run(model, input_ids, attention_mask, eos_id, pad_id, use_monitoring: bool):
    import torch
    from monitoring.generate import generate_with_monitoring  # type: ignore
    extra = {"cache_implementation": "static"}
    with torch.no_grad():
        if use_monitoring:
            generate_with_monitoring(
                model, input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=NEW_TOKENS, do_sample=False,
                pad_token_id=pad_id, eos_token_id=eos_id, **extra,
            )
        else:
            model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=NEW_TOKENS, do_sample=False,
                pad_token_id=pad_id, eos_token_id=eos_id, **extra,
            )


# ---------------------------------------------------------------------------
# Profile run
# ---------------------------------------------------------------------------
def run_profile():
    import torch
    device = torch.device("cuda")

    from transformers import AutoTokenizer  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    input_ids, attention_mask = _make_inputs(tokenizer, device)

    from transformers.models.qwen3_p.modeling_qwen3 import HookedQwen3ForCausalLM  # type: ignore
    print("Loading model...")
    model = HookedQwen3ForCausalLM.from_pretrained(
        MODEL_ID, attn_implementation="eager", torch_dtype=torch.float16)
    model.to(device).eval()

    print("Compiling...")
    model = torch.compile(model, mode="reduce-overhead")
    print("  done")

    # -----------------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------------
    model.monitoring_engine = None

    print(f"\nBaseline warmup ({WARMUP} iter)...")
    for _ in range(WARMUP):
        _run(model, input_ids, attention_mask, eos_id, pad_id, False)
    torch.cuda.synchronize()

    print(f"Baseline measure ({MEASURE} iter) — NVTX 'baseline'")
    for i in range(MEASURE):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"baseline_iter{i}")
        t0 = time.perf_counter()
        _run(model, input_ids, attention_mask, eos_id, pad_id, False)
        torch.cuda.synchronize()
        print(f"  baseline iter {i}: {(time.perf_counter()-t0)*1000:.1f} ms")
        torch.cuda.nvtx.range_pop()

    # -----------------------------------------------------------------------
    # ring_null monitoring
    # -----------------------------------------------------------------------
    model_id = f"profile::ring_null::{uuid.uuid4().hex[:8]}"
    engine   = _make_null_engine(model_id)
    model.monitoring_engine = engine

    ring_engine    = getattr(engine, "_ring_engine",    None)
    ring_transport = getattr(engine, "_ring_transport", None)

    print(f"\nring_null warmup ({WARMUP} iter, null mode)...")
    if ring_engine    is not None: ring_engine.set_null_mode(True)
    if ring_transport is not None: ring_transport.null_offload = True
    for _ in range(WARMUP):
        _run(model, input_ids, attention_mask, eos_id, pad_id, True)
    torch.cuda.synchronize()
    if ring_transport is not None:
        n_specs = len(ring_transport._active_specs)
        print(f"  [debug] active_specs={n_specs}"
              f"  using_forward_hooks={ring_transport._using_forward_hooks}")
    if ring_engine    is not None: ring_engine.set_null_mode(False)
    if ring_transport is not None: ring_transport.null_offload = False

    print(f"ring_null measure ({MEASURE} iter) — NVTX 'ring_null'")
    for i in range(MEASURE):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"ring_null_iter{i}")
        t0 = time.perf_counter()
        _run(model, input_ids, attention_mask, eos_id, pad_id, True)
        torch.cuda.synchronize()
        print(f"  ring_null iter {i}: {(time.perf_counter()-t0)*1000:.1f} ms")
        torch.cuda.nvtx.range_pop()

    engine.close()
    print("\nDone. Profile captured.")


# ---------------------------------------------------------------------------
# SQLite analysis — parse nsys export and report per-kernel/memcpy breakdown
# ---------------------------------------------------------------------------
def analyse_sqlite(path: str):
    import sqlite3, collections
    con = sqlite3.connect(path)
    cur = con.cursor()

    # shortName in CUPTI tables is an integer ID into StringIds
    cur.execute("SELECT id, value FROM StringIds")
    sid: dict = dict(cur.fetchall())

    # Map NVTX range names to time windows
    cur.execute("""
        SELECT text, start, end FROM NVTX_EVENTS
        WHERE text LIKE 'baseline_%' OR text LIKE 'ring_null_%'
        ORDER BY start
    """)
    nvtx_ranges = cur.fetchall()
    if not nvtx_ranges:
        print("No NVTX ranges found — check the sqlite file.")
        return

    # Build windows per mode
    windows: dict = {"baseline": [], "ring_null": []}
    for name, start, end in nvtx_ranges:
        if not (name and start and end):
            continue
        mode = "baseline" if name.startswith("baseline") else "ring_null"
        windows[mode].append((start, end))
        print(f"  {name}: {(end-start)/1e6:.1f} ms")

    def in_windows(t, wins):
        for s, e in wins:
            if s <= t <= e:
                return True
        return False

    # Per-mode kernel breakdown using start times; duration = end - start
    print("\n=== Kernel time breakdown by NVTX region ===")
    for mode in ("baseline", "ring_null"):
        wins = windows[mode]
        if not wins:
            print(f"  {mode}: no windows found")
            continue
        cur.execute("SELECT shortName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end IS NOT NULL")
        rows = cur.fetchall()
        totals: dict = {}
        for name_id, start, end in rows:
            name = sid.get(name_id, str(name_id))
            dur = end - start
            if in_windows(start, wins):
                if name not in totals:
                    totals[name] = [0, 0]
                totals[name][0] += dur
                totals[name][1] += 1
        top = sorted(totals.items(), key=lambda x: -x[1][0])[:20]
        total_ns = sum(v[0] for v in totals.values())
        n_launches = sum(v[1] for v in totals.values())
        print(f"\n  [{mode}]  total GPU kernel time: {total_ns/1e6:.1f} ms  ({n_launches} kernel launches)")
        print(f"  {'kernel':<60} {'ms':>8}  {'calls':>6}  {'%':>5}")
        print(f"  {'-'*83}")
        for name, vals in top:
            ns, cnt = vals[0], vals[1]
            short = (name[:57] + "...") if len(name) > 60 else name
            print(f"  {short:<60} {ns/1e6:>7.2f}ms  {cnt:>6}  {ns/total_ns*100:>4.1f}%")

    # Per-mode memcpy breakdown; copyKind: 1=HtoD, 2=DtoH, 8=DtoD
    print("\n=== Memory transfer breakdown by NVTX region ===")
    kind_name = {1: "HtoD", 2: "DtoH", 3: "HtoH", 8: "DtoD", 10: "PeerToPeer"}
    for mode in ("baseline", "ring_null"):
        wins = windows[mode]
        cur.execute("SELECT copyKind, start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE end IS NOT NULL")
        rows = cur.fetchall()
        totals2: dict = {}
        for kind, start, end, nbytes in rows:
            dur = end - start
            if in_windows(start, wins):
                if kind not in totals2:
                    totals2[kind] = [0, 0, 0]
                totals2[kind][0] += dur
                totals2[kind][1] += 1
                totals2[kind][2] += nbytes
        print(f"\n  [{mode}]")
        if not totals2:
            print("    (no memory transfers)")
        for kind in sorted(totals2.keys()):
            ns, cnt, nb = totals2[kind]
            kname = kind_name.get(kind, f"kind{kind}")
            print(f"    {kname:8s}: {ns/1e6:7.2f} ms  {cnt:5d} ops  "
                  f"{nb/1024**2:8.1f} MB  "
                  f"({nb/max(ns,1)*1e9/1024**3:.1f} GB/s)")

    con.close()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--analyse", metavar="SQLITE", help="Analyse exported nsys SQLite file")
    args = p.parse_args()

    if args.analyse:
        analyse_sqlite(args.analyse)
    else:
        run_profile()
