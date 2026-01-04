# TransformerLens Benchmark Suite

## 快速开始

### 1. 环境准备

创建并激活专用的 conda 环境：

```bash
# 创建环境
conda create -n TL python=3.10 -y

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TL

# 安装依赖
pip install torch transformer-lens GPUtil psutil numpy
```

### 2. 运行缓存性能测试

最简单的测试（快速验证）：
```bash
python benchmark/tests/cache_comparison.py --batch-sizes 1 4 --seq-lengths 128 --iterations 5
```

完整测试（全面评估）：
```bash
python benchmark/tests/cache_comparison.py \
    --batch-sizes 1 4 8 16 32 \
    --seq-lengths 128 256 512 1024 \
    --iterations 10
```

### 3. 运行批量推理基准测试

```bash
python benchmark/tests/batch_inference.py \
    --models gpt2 gpt2-medium \
    --batch-sizes 1 2 4 8 16 \
    --seq-lengths 128 256 512 \
    --iterations 10
```

### 4. TransformerLens vs HuggingFace 性能对比

对比 TransformerLens 和 HuggingFace 原生实现的性能差异：

快速测试：
```bash
python benchmark/tests/tl_vs_hf_comparison.py \
    --batch-sizes 1 4 \
    --seq-lengths 128 \
    --iterations 5
```

完整对比：
```bash
python benchmark/tests/tl_vs_hf_comparison.py \
    --model gpt2 \
    --batch-sizes 1 4 8 16 \
    --seq-lengths 128 256 512 \
    --iterations 10
```

这个测试会对比三种实现：
- **HuggingFace 原生**：直接使用 `AutoModelForCausalLM`
- **TransformerLens（无缓存）**：使用 `HookedTransformer` 普通推理
- **TransformerLens（带缓存）**：使用 `run_with_cache()`

### 5. Hook 开销 Profiling（支持 NVTX 标注）

使用 PyTorch Profiler 对比 TransformerLens 与 HuggingFace 推理时间线，并可选启用 NVTX
标注以定位 Hook 带来的额外开销：

```bash
python benchmark/tests/profile_inference.py \
    --batch-size 4 \
    --sequence-length 256 \
    --steps 3 \
    --warmup 2 \
    --nvtx
```

生成的 TensorBoard trace 位于 `results/profile_traces/`，可通过
`tensorboard --logdir results/profile_traces` 查看。

命令行会输出三个基线的汇总耗时/吞吐：
- `transformer_lens`：标准前向（无缓存）。
- `transformer_lens_cache`：`run_with_cache` 捕获全部 Hook 激活（保持在 GPU）。
- `huggingface`：Hugging Face GPT-2（强制 eager attention）。
- `huggingface_api`：调用 Hugging Face 原生接口输出 `hidden_states` 与 `attentions`。
- `huggingface_hook`：在 HF 模型上注册细粒度 hook（收集层内激活，留在 GPU）。
- `huggingface_hook_cpu`：同上，但每次捕获后立即搬到 CPU。

## 测试参数说明

### cache_comparison.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | gpt2 | 要测试的模型名称 |
| `--batch-sizes` | [1, 4, 8, 16] | 批量大小列表 |
| `--seq-lengths` | [128, 256, 512] | 序列长度列表 |
| `--iterations` | 10 | 每个配置的测试迭代次数 |
| `--device` | cuda | 运行设备 (cuda/cpu) |

### profile_inference.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 4 | Profiling 时的批量大小 |
| `--sequence-length` | 256 | 输入序列长度 |
| `--steps` | 3 | 在 profiler 中执行的前向次数 |
| `--warmup` | 2 | 进入 profiler 前的预热次数 |
| `--device` | 自动检测 | 运行设备（默认优先使用 GPU） |
| `--dtype` | fp32 | 推理使用的数据类型（fp32/fp16/bf16） |
| `--profile-dir` | results/profile_traces | Trace 输出目录 |
| `--nvtx` | False | 是否启用 TransformerLens NVTX 标注 |

### batch_inference.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--models` | [gpt2] | 要测试的模型列表 |
| `--batch-sizes` | [1, 2, 4, 8, 16] | 批量大小列表 |
| `--seq-lengths` | [128, 256, 512] | 序列长度列表 |
| `--iterations` | 10 | 每个配置的测试迭代次数 |
| `--output-dir` | results/batch_inference | 结果保存目录 |
| `--test-max-batch` | False | 是否测试最大批量大小 |

### tl_vs_hf_comparison.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | gpt2 | 要测试的模型名称 |
| `--batch-sizes` | [1, 4, 8] | 批量大小列表 |
| `--seq-lengths` | [128, 256] | 序列长度列表 |
| `--iterations` | 10 | 每个配置的测试迭代次数 |
| `--device` | cuda | 运行设备 (cuda/cpu) |

## 输出解读

### 缓存性能测试输出示例

```
============================================================
Batch Size: 4, Sequence Length: 256
============================================================

Metric                           No Cache      With Cache        Overhead
----------------------------------------------------------------------
Throughput (tokens/sec)           43653.9         28320.9           35.1%
Latency (ms/batch)                  11.73           18.08           54.1%
Peak Memory (MB)                    843.0          1260.6          417.6 MB

Summary
----------------------------------------------------------------------
Cache overhead: 35.1% slower, 417.6 MB extra memory
```

**关键指标说明**：
- **Throughput**: 每秒处理的 token 数量（越高越好）
- **Latency**: 每个批次的处理时间（越低越好）
- **Peak Memory**: 峰值内存使用（越低越好）
- **Overhead**: 使用缓存相对于不使用缓存的性能损失

### 批量推理测试输出

结果会保存到 CSV 和 JSON 文件中，包含：
- 不同批量大小和序列长度的吞吐量
- 内存使用情况
- 延迟统计
- 错误记录（如 OOM）

### TransformerLens vs HuggingFace 对比输出示例

```
================================================================================
Batch Size: 4, Sequence Length: 128
================================================================================

Implementation                  Load Time      Throughput     Latency       Memory
                                (seconds)    (tokens/sec)   (ms/batch)        (MB)
-----------------------------------------------------------------------------------------------
HuggingFace                          2.31        45234.5        11.30        694.1
TransformerLens (no cache)           3.45        38956.2        13.13        702.3
TransformerLens (with cache)         3.46        28320.9        18.08        810.4

Overhead Analysis (vs HuggingFace)
-----------------------------------------------------------------------------------------------

TransformerLens (no cache):
  Load time: +49.4% (+1.14s)
  Throughput: -13.9% slower
  Latency: +16.2% higher
  Memory: +8.2 MB more

TransformerLens (with cache):
  Load time: +49.8% (+1.15s)
  Throughput: -37.4% slower
  Latency: +60.0% higher
  Memory: +116.3 MB more
```

**关键发现**：
- TransformerLens 比 HuggingFace 慢约 14%（无缓存时）
- 启用缓存会带来额外 23% 的性能开销
- Hook 系统的灵活性是以性能为代价的

## 常见问题

### 1. CUDA Out of Memory

如果遇到内存不足错误，减小批量大小或序列长度：
```bash
python benchmark/tests/cache_comparison.py --batch-sizes 1 2 --seq-lengths 128 256
```

### 2. 测试特定模型

测试其他 HuggingFace 模型：
```bash
python benchmark/tests/cache_comparison.py --model gpt2-medium
python benchmark/tests/cache_comparison.py --model EleutherAI/gpt-neo-125M
```

### 3. CPU 测试

在 CPU 上运行（速度较慢）：
```bash
python benchmark/tests/cache_comparison.py --device cpu --batch-sizes 1 2 --seq-lengths 64 128
```

## 性能优化建议

基于测试结果，使用 TransformerLens 时的优化建议：

1. **选择性缓存**：仅在需要分析激活值时使用 `run_with_cache()`
2. **批量处理**：较大的批量大小有更好的吞吐量
3. **内存管理**：缓存会显著增加内存使用，注意 GPU 内存限制
4. **生产环境**：在生产环境中，普通推理（不带缓存）性能更好

## 扩展测试

如需添加更多测试场景，可以：

1. 继承 `BaseBenchmark` 类创建新的测试
2. 使用 `MetricsCollector` 收集自定义指标
3. 参考 `cache_comparison.py` 实现特定功能的测试
