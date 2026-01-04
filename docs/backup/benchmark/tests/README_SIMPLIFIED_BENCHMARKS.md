# Simplified HF Modified Benchmarks

这个目录包含两个简化的 benchmark，专门用于测试 HF Modified (HookedGPT2Model) 的性能：

1. **`hf_modified_sync_only.py`** - 同步版本（无 monitoring cache）
2. **`hf_modified_async_only.py`** - 异步版本（使用 monitoring engine）

## 与原始 benchmark 的区别

- ✅ 只包含相关的 benchmark 配置
- ✅ 保留所有 NVTX 注释
- ✅ 保留相同的用法和参数结构
- ✅ 移除了所有无关的 baseline（transformer_lens, huggingface_api 等）

---

## 1. 同步版本 (hf_modified_sync_only.py)

### 功能
- 测试 `HookedGPT2Model` 的基础性能
- **不使用** `run_with_cache()`
- **不使用** monitoring engine
- 只进行普通的模型前向传播

### 使用方法

```bash
# 快速测试（无 profiling）
python benchmark/tests/hf_modified_sync_only.py \
  --batch-size 4 \
  --prefill-tokens 1 \
  --decode-steps 64 \
  --steps 3 \
  --warmup 1 \
  --device cuda \
  --dtype fp32 \
  --no-profile

# 带 profiling（生成 TensorBoard 跟踪文件）
python benchmark/tests/hf_modified_sync_only.py \
  --batch-size 4 \
  --prefill-tokens 1 \
  --decode-steps 64 \
  --steps 3 \
  --warmup 1 \
  --device cuda \
  --dtype fp32 \
  --profile-dir results/profile_sync

# 启用 NVTX 注释（用于 Nsight Systems）
python benchmark/tests/hf_modified_sync_only.py \
  --nvtx \
  --batch-size 4 \
  --decode-steps 64 \
  --no-profile
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--batch-size` | 4 | Batch 大小 |
| `--prefill-tokens` | 1 | Prefill 阶段的 token 数 |
| `--decode-steps` | 64 | Decode 步数 |
| `--steps` | 3 | Benchmark 迭代次数 |
| `--warmup` | 1 | Warmup 迭代次数 |
| `--device` | cuda/cpu | 设备类型 |
| `--dtype` | fp32 | 计算精度（fp32/fp16/bf16） |
| `--profile-dir` | results/profile_hf_modified_sync | Profiler 输出目录 |
| `--nvtx` | False | 启用 NVTX 注释 |
| `--no-profile` | False | 跳过 profiling，只测量时间 |

### 输出

```
Timing results:
- hf_modified: duration=0.3071s tokens/s=6.51
```

---

## 2. 异步版本 (hf_modified_async_only.py)

### 功能
- 测试 `HookedGPT2Model` + monitoring engine 的性能
- **使用** `run_with_cache()` 捕获激活值
- **使用** monitoring engine 异步处理
- 测量主流时间和总时间（包括后台任务）

### 使用方法

```bash
# 基础测试（收集 hidden states）
python benchmark/tests/hf_modified_async_only.py \
  --batch-size 4 \
  --prefill-tokens 1 \
  --decode-steps 64 \
  --steps 3 \
  --warmup 1 \
  --device cuda \
  --dtype fp32 \
  --collect-hidden \
  --no-profile

# 收集 attention + hidden states
python benchmark/tests/hf_modified_async_only.py \
  --batch-size 4 \
  --decode-steps 64 \
  --collect-hidden \
  --collect-attention \
  --no-profile

# 带 profiling + NVTX
python benchmark/tests/hf_modified_async_only.py \
  --nvtx \
  --batch-size 4 \
  --decode-steps 64 \
  --collect-hidden \
  --profile-dir results/profile_async

# 测试 monitoring engine 参数
python benchmark/tests/hf_modified_async_only.py \
  --collect-hidden \
  --engine-queue-size 1024 \
  --engine-delay-steps 2 \
  --cache-dtype fp16 \
  --no-profile
```

### 参数说明

除了同步版本的参数外，还包括：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--collect-hidden` | False | **必需**：收集 hidden states |
| `--collect-attention` | False | **必需**：收集 attention tensors |
| `--cache-dtype` | none | Monitoring engine 缓存精度 |
| `--engine-queue-size` | 0 | 队列大小（0=无界） |
| `--engine-delay-steps` | 0 | 延迟处理步数 |

**注意**：必须指定 `--collect-hidden` 和/或 `--collect-attention`，否则会报错。

### 输出

```
Timing results:
- hf_modified_hook_async:
    main_duration=0.3075s     # 主流时间（模型前向）
    total_duration=0.3077s    # 总时间（包括后台处理）
    main_token/s=6.50         # 基于主流时间的吞吐量
    total_token/s=6.50        # 真实吞吐量
```

**时间说明**：
- `main_duration`：只等待主 CUDA 流完成，**不等待** cache 流
- `total_duration`：等待所有后台监控任务完成（调用 `resolve_all()`）

---

## 对比两个版本

### 同步版本 (Sync)
```
代码路径: hf_modified_prefill/decode (line 194-268)
调用方式: hf_hooked_model(input_ids, ..., use_cache=True)
监控功能: ❌ 无
开销:     最小（只有模型前向）
```

### 异步版本 (Async)
```
代码路径: hf_modified_hook_async_prefill/decode (line 281-388)
调用方式:
  monitoring_engine.start_step()
  hf_hooked_model.run_with_cache(...)
  monitoring_engine.end_step()
监控功能: ✅ 收集激活值/注意力权重
开销:     主流开销小 (dispatch)，后台处理 (worker)
```

---

## 结果文件

两个 benchmark 都会生成 JSON 结果文件：

```json
{
  "config": {
    "batch_size": 4,
    "prefill_tokens": 1,
    "decode_steps": 64,
    "steps": 3,
    "warmup": 1,
    "device": "cuda",
    "dtype": "fp32",
    ...
  },
  "timings": {
    "hf_modified": {
      "duration": 0.3071,
      "tokens_per_second": 6.51
    }
  },
  "total_decoded_tokens": 768
}
```

---

## Profiling 输出

如果不使用 `--no-profile`，会生成 TensorBoard 跟踪文件：

```bash
# 查看 profiling 结果
tensorboard --logdir results/profile_hf_modified_sync/hf_modified
tensorboard --logdir results/profile_hf_modified_async/hf_modified_hook_async
```

---

## NVTX 注释层次

启用 `--nvtx` 后，在 Nsight Systems 中可以看到以下层次：

### 同步版本
```
benchmark_iter_0
├── modified_prefill
│   ├── modified_prefill_forward
│   ├── modified_prefill_post
│   │   └── modified_prefill_project
├── modified_decode (x64)
    ├── modified_decode_forward
    └── modified_decode_post
        └── modified_decode_project
```

### 异步版本
```
benchmark_iter_0
├── async_prefill
│   ├── async_prefill_forward
│   ├── async_prefill_end_step
│   └── async_prefill_post
├── async_decode (x64)
    ├── async_decode_start_step
    ├── async_decode_forward
    ├── async_decode_end_step
    └── async_decode_post
```

---

## 常见用例

### 1. 测量基准性能（无监控开销）
```bash
python benchmark/tests/hf_modified_sync_only.py --no-profile
```

### 2. 测量监控开销
```bash
# 运行两个 benchmark 并对比结果
python benchmark/tests/hf_modified_sync_only.py --no-profile
python benchmark/tests/hf_modified_async_only.py --collect-hidden --no-profile
```

### 3. 用 Nsight Systems 分析
```bash
nsys profile -o profile_sync python benchmark/tests/hf_modified_sync_only.py \
  --nvtx --no-profile --decode-steps 128

nsys profile -o profile_async python benchmark/tests/hf_modified_async_only.py \
  --nvtx --no-profile --decode-steps 128 --collect-hidden
```

### 4. 测试不同 queue 配置
```bash
# 无界队列
python benchmark/tests/hf_modified_async_only.py \
  --collect-hidden --engine-queue-size 0 --no-profile

# 有界队列（背压）
python benchmark/tests/hf_modified_async_only.py \
  --collect-hidden --engine-queue-size 128 --no-profile

# 延迟处理
python benchmark/tests/hf_modified_async_only.py \
  --collect-hidden --engine-delay-steps 4 --no-profile
```

---

## 故障排查

### 问题 1：CUDA OOM
```
解决方案：减小 batch-size 或使用 fp16/bf16
python ... --batch-size 1 --dtype fp16
```

### 问题 2：异步版本必需参数
```
错误: At least one of --collect-hidden or --collect-attention must be provided.
解决方案：添加 --collect-hidden 或 --collect-attention
```

### 问题 3：Monitoring engine 未使用
```
检查: hf_hooked_model.monitoring_engine 是否绑定
同步版本: monitoring_engine = None（正确）
异步版本: monitoring_engine = <MonitoringEngine>（必需）
```

---

## 与原始 profile_decode.py 的对应关系

| 原始 Benchmark | 对应的简化版本 |
|---------------|--------------|
| `hf_modified` (line 974-984) | `hf_modified_sync_only.py` |
| `hf_modified_hook_async` (line 1004-1025) | `hf_modified_async_only.py` |

---

## 贡献者注意事项

如果修改原始 `profile_decode.py` 中的 `hf_modified` 或 `hf_modified_hook_async` 函数，请同步更新这两个简化版本：
- 保持 NVTX 注释一致
- 保持参数和用法一致
- 更新本 README
