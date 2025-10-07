"""
CLI/脚本版：读取 HF_Prometheus/results 下 1..5/6 轮目录，
比较以下基线：modified_hf, modified_hf_hook, modified_hf_hook_async, hf_hook，
并输出主要开销来源以及按类别聚合的柱状图（可选保存为 PNG）。

用法：
  python HF_Prometheus/notebooks/overhead_analysis.py \
      --runs HF_modified_decode_async HF_modified_decode_async_2 ... \
      --save-dir HF_Prometheus/results/overhead_charts

如果不传 --runs，默认尝试 1..5 目录。
"""
from __future__ import annotations

import argparse
import os
import json
from collections import defaultdict, OrderedDict
from pathlib import Path

try:
    import matplotlib.pyplot as plt  # type: ignore
    import pandas as pd  # type: ignore
    import seaborn as sns  # type: ignore
except Exception:
    plt = None
    pd = None
    sns = None

# 结果根目录的自动探测（支持 --root / PROM_RESULTS_ROOT / 常见相对路径）
def detect_results_root(explicit: str | None) -> Path:
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    env = os.getenv('PROM_RESULTS_ROOT')
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    cands.append(here.parent.parent / 'results')          # HF_Prometheus/results（相对脚本位置）
    cands.append(Path('HF_Prometheus') / 'results')       # 从仓库根运行
    cands.append(Path('results'))                         # 从 HF_Prometheus 目录运行
    cands.append(Path.cwd() / 'HF_Prometheus' / 'results')
    cands.append(Path.cwd() / 'results')
    for p in cands:
        if p.exists() and p.is_dir():
            return p
    # 尽量返回一个合理的默认值（即使不存在），便于报错提示
    return Path('HF_Prometheus') / 'results'
DEFAULT_RUNS = [
    'HF_modified_decode_async',
    'HF_modified_decode_async_2',
    'HF_modified_decode_async_3',
    'HF_modified_decode_async_4',
    'HF_modified_decode_async_5',
    'HF_modified_decode_async_6',
]
BASELINES = OrderedDict([
    ('hf_modified', 'modified_hf'),
    ('hf_modified_hook', 'modified_hf_hook'),
    ('hf_modified_hook_async', 'modified_hf_hook_async'),
    ('huggingface_hook', 'hf_hook'),
])

GROUPS = OrderedDict({
    'slice': ['aten::slice'],
    'events': ['cudaEventRecordWithFlags', 'cudaStreamWaitEvent', 'cudaStreamIsCapturing'],
    'tensor_rearrange': ['aten::reshape', 'aten::view', 'aten::empty', 'aten::clone', 'aten::copy_', 'aten::as_strided'],
    'compute_heads': ['aten::addmm', 'aten::matmul', 'aten::native_layer_norm', 'aten::layer_norm', 'aten::bmm'],
})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=None, help='结果目录名列表（默认尝试 1..5）')
    ap.add_argument('--save-dir', default=None, help='如提供则保存 PNG 到该目录')
    ap.add_argument('--root', default=None, help='结果根目录（默认自动探测）')
    return ap.parse_args()


def find_trace_file(run_dir: Path, subdir: str) -> Path | None:
    d = run_dir / subdir
    if not d.exists():
        return None
    files = sorted(d.glob('*.json'))
    return files[-1] if files else None


def aggregate_ops(trace: dict) -> tuple[float, dict]:
    agg = defaultdict(float)
    total = 0.0
    for evt in trace.get('traceEvents', []):
        if evt.get('ph') != 'X':
            continue
        cat = evt.get('cat', '')
        name = evt.get('name', '')
        # 过滤无关包裹事件
        if cat in ('user_annotation', 'gpu_user_annotation'):
            continue
        if cat == 'Trace' and 'PyTorch Profiler' in name:
            continue
        dur = evt.get('dur')
        if dur is None:
            continue
        key = (cat, name)
        agg[key] += dur
        total += dur
    return total, agg


def load_runs(results_root: Path, run_names: list[str]):
    runs = []
    for name in run_names:
        run_dir = results_root / name
        if not run_dir.exists():
            continue
        tables = {}
        totals = {}
        for sub, label in BASELINES.items():
            p = find_trace_file(run_dir, sub)
            if p is None:
                continue
            try:
                trace = json.loads(p.read_text())
            except Exception as e:
                print('Failed to read', p, e)
                continue
            total, agg = aggregate_ops(trace)
            tables[label] = agg
            totals[label] = total / 1e6  # ms
        if tables:
            runs.append((name, tables, totals))
    return runs


def top_deltas(base: dict, to: dict, k=15):
    keys = set(base) | set(to)
    items = []
    for key in keys:
        delta = (to.get(key, 0.0) - base.get(key, 0.0)) / 1e3
        if abs(delta) >= 1.0:
            items.append((delta, key))
    items.sort(reverse=True)
    return items[:k]


def group_delta(base: dict, to: dict):
    out = {g: 0.0 for g in GROUPS}
    for (cat, name), dur in to.items():
        delta = (dur - base.get((cat, name), 0.0)) / 1e3
        for g, keys in GROUPS.items():
            if any(name.startswith(k) for k in keys):
                out[g] += delta
                break
    return out


def main():
    args = parse_args()
    run_names = args.runs or DEFAULT_RUNS
    results_root = detect_results_root(args.root)
    print('Using results root:', results_root)
    runs = load_runs(results_root, run_names)
    if not runs:
        print('No runs found under', results_root)
        return

    # 1) 打印总时长
    print('== 总时长概览 (ms) ==')
    for run_name, _, totals in runs:
        print('  -', run_name)
        for label, ms in totals.items():
            print(f'    {label:>22}: {ms:8.1f} ms')

    # 2) Top-K 相对 modified_hf
    print('\n== Top-K 相对 modified_hf 的开销 (ms) ==')
    for run_name, tables, _ in runs:
        base = tables.get('modified_hf')
        if not base:
            continue
        for tgt in ['modified_hf_hook', 'modified_hf_hook_async', 'hf_hook']:
            tab = tables.get(tgt)
            if not tab:
                continue
            diff = top_deltas(base, tab, 15)
            print(f'  [{run_name}] {tgt}')
            for d, k in diff:
                print(f'     {d:8.1f} ms  {k}')

    # 3) 类别聚合柱状图（若无依赖则跳过）
    rows = []
    for run_name, tables, _ in runs:
        base = tables.get('modified_hf')
        if not base:
            continue
        for tgt in ['modified_hf_hook', 'modified_hf_hook_async', 'hf_hook']:
            tab = tables.get(tgt)
            if not tab:
                continue
            gd = group_delta(base, tab)
            for g, val in gd.items():
                rows.append({'run': run_name, 'target': tgt, 'group': g, 'delta_ms': val})
    if rows and (pd is not None and sns is not None and plt is not None):
        df = pd.DataFrame(rows)
        sns.set_style('whitegrid')
        g = sns.catplot(data=df, x='group', y='delta_ms', hue='target', col='run', kind='bar', col_wrap=2, height=3, sharex=False, sharey=False)
        g.set_xticklabels(rotation=45)
        g.fig.suptitle('相对 modified_hf 的类别聚合开销（ms）', y=1.02)
        if args.save_dir:
            out_dir = Path(args.save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / 'grouped_overhead.png'
            plt.savefig(out_file, dpi=160, bbox_inches='tight')
            print('\nSaved figure to', out_file)
        else:
            plt.show()
    elif rows:
        print('\n[提示] 未安装 pandas/seaborn/matplotlib，已跳过绘图，仅输出文本结果。')


if __name__ == '__main__':
    main()
