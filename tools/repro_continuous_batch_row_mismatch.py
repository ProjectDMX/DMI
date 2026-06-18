"""Standalone reproducer: under vLLM V1 continuous batching, DMI attaches each
captured activation to the WRONG request (row<->request mismatch) once requests
start finishing mid-run.

Self-contained -- depends only on DMI (monitoring + integration) + vLLM + a small
model. No probe, no dataset, no application code.

Idea: capture one mid-layer resid_pre per generated token, keyed by the slice's
own (req_id, end_token) metadata, in TWO modes:
  (A) one request per generate()  -> the known-good baseline (batch size 1, no
      row ambiguity).
  (B) all prompts in ONE generate() -> continuous batching (many concurrent).
Then compare, for the SAME request at the SAME token position, cosine(A, B) of
the captured hidden. It must be ~1.0; under the bug it collapses to ~0.4-0.5.

Two things are essential to make this a VALID and DISCRIMINATING test:

  * Verify the controlled variable first. Greedy (temperature=0) can still flip
    the argmax at ties under different batch shapes, so the two modes may emit
    DIFFERENT tokens from some position on. Past that divergence the same
    end_token is a different token and a low cosine is EXPECTED, not a bug. We
    detect the first divergence per prompt and compare ONLY token-matched
    positions, bucketed by the TRUE decode position (end_token - prompt_len).

  * Trigger the bug. It only bites once vLLM frees a finished request's slot and
    CONDENSES input_batch, making the scheduler-dict order diverge from the slot
    order. Uniform output lengths -> all finish together -> no condense -> the
    bug never fires (verified: buggy and fixed give identical output). So we give
    each request a DIFFERENT max_tokens; the collapse turns on exactly at the
    decode position where the shortest requests complete.

  BUGGY  : token-matched cosine collapses (mean ~0.55, >80% < 0.9) from the first
           post-completion decode position onward                       -> BUG
  FIXED  : token-matched cosine ~1.0 at every decode position           -> PASS
  (token divergence / thin delivery only)                               -> INCONCLUSIVE

Run:
    cd ~/DMI-hallu && source <env that puts monitoring/integration + the built
    vLLM fork on PYTHONPATH>            # e.g. ~/hallu-monitor/env.sh
    CUDA_VISIBLE_DEVICES=0 python tools/repro_continuous_batch_row_mismatch.py
"""
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process worker
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")

import numpy as np

MODEL = os.environ.get("REPRO_MODEL", "Qwen/Qwen3-0.6B")
LAYER = int(os.environ.get("REPRO_LAYER", "14"))        # any resid_pre layer
ACT = "blocks.hook_resid_pre"

# 24 short factual prompts (the per-request max_tokens below, not the prompt
# texts, is what staggers completion and triggers the condense that exposes the
# bug). temperature=0 -> mostly identical tokens single-vs-continuous.
PROMPTS = [
    "The capital of France is", "Two plus two equals", "The sky is",
    "Water is made of hydrogen and", "The opposite of hot is",
    "A group of lions is called a", "The first president of the United States was",
    "Photosynthesis happens in the", "The square root of nine is",
    "The largest planet is", "Ice is frozen", "The author of Hamlet is",
    "A triangle has", "The speed of light is about", "Bees make",
    "The chemical symbol for gold is", "The Earth orbits the",
    "A baby dog is called a", "The freezing point of water is",
    "Mount Everest is the tallest", "The currency of Japan is the",
    "Red mixed with blue makes", "The human body has 206",
    "The sun rises in the",
]


def main():
    from vllm import LLM, SamplingParams
    from integration.vllm_adapter import register_sink_factory, normalize_vllm_request_id

    # in-memory callable sink: record final hidden per (req_id, end_token).
    # SubmitFn signature mirrors monitoring's p2p delivery.
    store = {}

    def sink(model_id, shard_rank, req_id, act_name, layer_no,
             start_token, end_token, tensor):
        if act_name == ACT and layer_no == LAYER and tensor.shape[0] == 1:
            store[(req_id, int(end_token))] = tensor[-1].float().cpu().numpy()

    register_sink_factory(lambda: sink)

    llm = LLM(model=MODEL, worker_cls="integration.vllm_adapter.DMXGPUWorker",
              additional_config={"dmx_hook_selection": "hidden-states",
                                 "dmx_ring_payload_mb": 1024, "dmx_ring_pinned_mb": 1024},
              max_model_len=256, max_num_batched_tokens=2048, enforce_eager=True,
              enable_prefix_caching=False,   # keep absolute token positions clean
              gpu_memory_utilization=float(os.environ.get("REPRO_GPU_MEM", "0.5")),
              tensor_parallel_size=1)
    tok = llm.get_tokenizer()
    # VARYING max_tokens per request: requests finish at STAGGERED steps, so vLLM
    # frees + condenses input_batch slots mid-run and the scheduler-dict order
    # diverges from the input_batch slot order -- the condition under which the
    # row<->request mapping bug bites. (Uniform lengths -> all finish together ->
    # no condense -> no divergence -> the bug is never triggered and the repro
    # cannot discriminate; verified.)
    params = [SamplingParams(temperature=0.0, max_tokens=4 + 2 * i)
              for i in range(len(PROMPTS))]

    def snapshot(o):
        # everything we need about ONE finished request: delivered hiddens keyed by
        # end_token, the generated token ids, and the prompt length.
        rid = normalize_vllm_request_id(o.request_id)
        hid = {et: store[(rid, et)] for (r, et) in store if r == rid}
        return {"rid": rid, "hid": hid,
                "gen": list(o.outputs[0].token_ids),
                "plen": len(o.prompt_token_ids)}

    def drain():
        for _ in range(6):
            llm.collective_rpc("dmx_flush"); time.sleep(0.15)

    # (A) single request per generate (batch size 1 -> no row ambiguity = truth)
    single = {}
    for i, p in enumerate(PROMPTS):
        store.clear()
        o = llm.generate([p], params[i])[0]
        drain()
        single[p] = snapshot(o)

    # (B) all prompts in one generate -> continuous batching (same per-request params)
    store.clear()
    outs = llm.generate(PROMPTS, params)
    drain()
    cont = {p: snapshot(o) for p, o in zip(PROMPTS, outs)}

    # --- (1) verify the CONTROLLED variable: are the generated tokens identical? ---
    #     If a token diverges (greedy argmax can flip under different batch shapes),
    #     every later same-end_token hidden is a DIFFERENT token -> low cosine is
    #     EXPECTED and is not a capture bug. We only compare positions strictly
    #     before the first divergence.
    tok_equal = 0
    first_div = {}          # prompt -> index of first differing generated token (or len)
    for p in PROMPTS:
        sg, cg = single[p]["gen"], cont[p]["gen"]
        n = min(len(sg), len(cg))
        div = next((i for i in range(n) if sg[i] != cg[i]), n if len(sg) == len(cg) else n)
        first_div[p] = div
        tok_equal += (sg == cg)
    print(f"\n(1) token-sequence equality: {tok_equal}/{len(PROMPTS)} prompts identical "
          f"single-vs-continuous")
    for i, p in enumerate(PROMPTS):
        if single[p]["gen"] != cont[p]["gen"]:
            print(f"    [{i:2d}] diverges at generated-token index {first_div[p]} "
                  f"(<= this index is comparable)")

    # --- (2)/(3) compare only TOKEN-MATCHED positions, bucket by TRUE decode_pos ---
    #     decode_pos = end_token - prompt_len. A hidden at decode_pos d depends on
    #     generated tokens 0..d-1, so it is comparable iff d <= first_div.
    cos = []
    by_pos = {}             # decode_pos -> [cosine]
    low = []                # (cos, prompt_idx, rid, end_token, plen, decode_pos)
    n_excluded_tokdiv = 0
    for i, p in enumerate(PROMPTS):
        s, c = single[p], cont[p]
        plen = s["plen"]
        div = first_div[p]
        for et in sorted(set(s["hid"]) & set(c["hid"])):
            dpos = et - plen
            if dpos > div:           # this hidden sits past the token divergence
                n_excluded_tokdiv += 1
                continue
            a, b = s["hid"][et], c["hid"][et]
            v = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            cos.append(v)
            by_pos.setdefault(dpos, []).append(v)
            if v < 0.9:
                low.append((v, i, c["rid"], et, plen, dpos))

    print(f"\n(2) cosine by TRUE decode_pos (= end_token - prompt_len):")
    for dpos in sorted(by_pos):
        v = np.array(by_pos[dpos])
        print(f"    decode_pos {dpos:2d}: mean={v.mean():.3f} "
              f"frac<0.9={np.mean(v < 0.9):.2f} (n={len(v)})")

    if low:
        print(f"\n(3) token-MATCHED positions with cosine < 0.9 (real suspects):")
        for v, i, rid, et, plen, dpos in sorted(low)[:15]:
            print(f"    prompt[{i:2d}] req={rid} end_token={et} plen={plen} "
                  f"decode_pos={dpos} cosine={v:.3f}")
    else:
        print(f"\n(3) no token-matched position has cosine < 0.9")

    cos = np.array(cos) if cos else np.array([1.0])
    print(f"\nmodel={MODEL} layer={LAYER}")
    print(f"compared {len(cos)} token-MATCHED (request, token) positions "
          f"({n_excluded_tokdiv} excluded for token divergence)")
    print(f"  mean={cos.mean():.4f}  median={np.median(cos):.4f}  "
          f"min={cos.min():.4f}  frac<0.9={np.mean(cos < 0.9):.2f}")

    # --- (4) three-state verdict ---
    enough = len(cos) >= 4 * len(PROMPTS)        # need a healthy number of matched positions
    cos_ok = np.median(cos) > 0.99 and np.mean(cos < 0.9) < 0.05
    if not enough:
        print("\nINCONCLUSIVE: too few token-matched / delivered positions to judge "
              f"(have {len(cos)}, want >= {4 * len(PROMPTS)}). "
              "Token divergence or incomplete delivery dominates.")
        return 2
    if cos_ok:
        print("\nPASS: token-matched continuous-batch capture == single-request (correct).")
        return 0
    print("\nBUG: tokens equal + enough positions, but captured hidden differs "
          "(continuous-batch attaches the wrong row to requests).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
