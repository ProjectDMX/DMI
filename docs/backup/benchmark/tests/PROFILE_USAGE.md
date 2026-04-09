# Profile Decode Benchmark Usage

## Overview

The `profile_decode.py` script can run in two modes:

1. **Profiling mode** (default): Generates detailed trace files + measures wallclock time
2. **Measurement-only mode** (`--no-profile`): Only measures wallclock time (much faster)

## Why use `--no-profile`?

The PyTorch profiler adds **significant overhead** (often 8-10x slower) to execution time. This overhead is NOT included in the trace file's `user_annotation.dur` field, causing discrepancies:

- **Terminal output time**: Includes profiler overhead (real wallclock time)
- **Trace file time**: Excludes profiler overhead (partial time)

For async benchmarks, this creates additional confusion because:
- The trace records queue management overhead
- But doesn't record the actual async processing time
- Making async appear slower than sync in traces, even when it's actually faster

## Usage Examples

### Get accurate performance measurements (recommended for comparison)

```bash
python benchmark/tests/profile_decode.py \
  --batch-size 64 \
  --decode-steps 64 \
  --steps 1 \
  --profile-dir results/benchmark_no_profile \
  --no-profile \
  --collect-hidden \
  --collect-attention
```

**Output:**
```
Running benchmarks WITHOUT profiling (pure wallclock time measurement)

Timing results saved to: /path/to/results/benchmark_no_profile/timing_results.json

Timing results (decode duration per run):
-   transformer_lens: duration=5.0454s tokens/s=811.82
-        hf_modified: duration=2.7068s tokens/s=1513.24
-   hf_modified_hook: duration=4.9751s tokens/s=823.30
- hf_modified_hook_async: main_duration=4.4106s total_duration=4.4180s
                          main_token/s=928.68 total_token/s=927.11
```

✅ These times are **accurate** and **comparable** across baselines.
✅ Results are saved to `timing_results.json` for later analysis.

### Generate profiler traces for detailed analysis

```bash
python benchmark/tests/profile_decode.py \
  --batch-size 64 \
  --decode-steps 64 \
  --steps 1 \
  --profile-dir results/my_profile_run \
  --collect-hidden \
  --collect-attention
```

**Output:**
```
Running benchmarks WITH profiling (trace files will be generated)

Timing results (decode duration per run):
-   transformer_lens: duration=12.3456s tokens/s=331.51
-        hf_modified: duration=8.9012s tokens/s=460.15
...

Profiler traces written under:
  results/my_profile_run
```

⚠️ Note: The wallclock times include profiler overhead and are **NOT** comparable to `--no-profile` runs.

✅ Use the trace files to analyze:
- CPU/GPU operation breakdown
- Memory usage patterns
- Kernel launch patterns
- NOT for comparing total execution time between baselines

## Performance Comparison Workflow

1. **First**: Run with `--no-profile` to get accurate performance metrics
2. **Then**: Run with profiling enabled for the baseline(s) you want to analyze in detail
3. **Analyze**: Use the trace files for breakdown analysis, use terminal output for total time

## Key Differences

| Feature | Default (profiling) | `--no-profile` |
|---------|-------------------|----------------|
| Execution speed | 8-10x slower | Normal speed |
| Trace files | ✅ Generated | ❌ Not generated |
| Wallclock time | ⚠️ Includes profiler overhead | ✅ Accurate |
| Trace duration | ⚠️ Excludes profiler overhead | N/A |
| Use case | Detailed breakdown analysis | Performance comparison |

## Interpreting Results

### For `--no-profile` runs:
- `duration`: Accurate end-to-end wallclock time
- `main_duration` (async): Time excluding `resolve_all()`
- `total_duration` (async): Time including `resolve_all()`

**Example interpretation:**
```
- hf_modified_hook:       duration=4.9751s
- hf_modified_hook_async: main_duration=4.4106s total_duration=4.4180s
```
→ Async is **11.4% faster** than sync (4.41s vs 4.98s)
→ Async overhead is minimal (0.0074s difference between main and total)

### For profiling runs:
- Terminal `duration`: Wallclock time including profiler overhead
- Trace `user_annotation.dur`: Recorded time excluding profiler overhead
- **These will NOT match!**
- Use trace for breakdown, terminal for comparison

## Output Files

### `timing_results.json` (always generated)

Both profiling and no-profile modes generate this file with structured results:

```json
{
  "config": {
    "batch_size": 64,
    "prefill_tokens": 1,
    "decode_steps": 64,
    "steps": 1,
    "warmup": 1,
    "device": "cuda",
    "dtype": "fp32",
    "collect_hidden": true,
    "collect_attention": true,
    "cache_dtype": "none",
    "engine_queue_size": 256,
    "engine_delay_steps": 0,
    "profiling_enabled": false
  },
  "timings": {
    "transformer_lens": {
      "duration": 5.0454,
      "tokens_per_second": 811.82
    },
    "hf_modified_hook": {
      "duration": 4.9751,
      "tokens_per_second": 823.30
    },
    "hf_modified_hook_async": {
      "main_duration": 4.4106,
      "total_duration": 4.4180,
      "tokens_per_second_main": 928.68,
      "tokens_per_second_total": 927.11
    }
  },
  "total_decoded_tokens": 4096
}
```

This file can be used for:
- Automated analysis scripts
- Comparing results across different runs
- Generating charts and reports

### Profiler trace files (only with profiling enabled)

When profiling is enabled, additional `.json` trace files are generated in subdirectories:
```
results/my_profile_run/
├── timing_results.json          # Always generated
├── transformer_lens/            # Only with profiling
│   └── *.pt.trace.json
├── hf_modified_hook/
│   └── *.pt.trace.json
└── hf_modified_hook_async/
    └── *.pt.trace.json
```

## Recommendations

1. **Always use `--no-profile` for performance benchmarking**
2. Only enable profiling when you need detailed trace analysis
3. Never compare trace times between runs - use terminal output or `timing_results.json`
4. For async baselines, pay attention to main vs total duration
5. Use `timing_results.json` for programmatic analysis and visualization
