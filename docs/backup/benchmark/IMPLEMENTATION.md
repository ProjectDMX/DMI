# TransformerLens Benchmark 实现文档

## 架构设计

### 整体架构

```
benchmark/
├── core/                      # 核心框架
│   ├── base_benchmark.py     # 基础测试类
│   ├── metrics.py            # 指标收集和管理
│   └── utils.py              # 工具函数
├── tests/                     # 具体测试实现
│   ├── batch_inference.py    # 批量推理测试
│   └── cache_comparison.py   # 缓存性能对比测试
├── configs/                   # 配置文件（待扩展）
├── analysis/                  # 分析工具（待扩展）
└── results/                   # 测试结果存储
```

## 核心组件实现

### 1. BaseBenchmark 基类

**设计思路**：
- 提供统一的测试框架，所有具体测试继承此类
- 处理模型加载、内存管理、结果保存等通用逻辑
- 支持批量测试不同配置组合

**关键方法**：
```python
class BaseBenchmark:
    def __init__(self, model_names, batch_sizes, sequence_lengths, ...):
        # 初始化测试参数和指标收集器
        
    def load_model(self, model_name):
        # 加载模型，使用缓存避免重复加载
        
    @abstractmethod
    def run_single_test(self, model, batch_size, sequence_length, **kwargs):
        # 抽象方法，子类必须实现具体测试逻辑
        
    def run(self, **kwargs):
        # 执行完整测试套件，遍历所有配置组合
```

### 2. MetricsCollector 指标收集器

**设计目标**：
- 统一管理所有性能指标
- 支持实时计算和批量导出
- 提供多种输出格式（JSON、CSV）

**核心数据结构**：
```python
@dataclass
class BenchmarkResult:
    # 基础信息
    model_name: str
    test_name: str
    batch_size: int
    sequence_length: int
    
    # 性能指标
    throughput_tokens_per_sec: float
    latency_ms_per_batch: float
    latency_ms_per_token: float
    
    # 内存指标
    gpu_memory_mb: float
    peak_gpu_memory_mb: float
    
    # Hook 配置（TransformerLens 特有）
    num_hooks: int
    hooks_enabled: bool
    cache_all_activations: bool
```

### 3. 缓存性能对比测试实现

**核心思路**：对比两种推理模式的性能差异

#### 3.1 无缓存推理测试
```python
def benchmark_without_cache(model, input_ids, num_iterations):
    # 1. 清理 GPU 内存
    clear_gpu_memory()
    
    # 2. 预热模型（3次小批量）
    warmup_model(model, input_ids[:1])
    
    # 3. 记录初始内存
    memory_before = get_gpu_memory()
    
    # 4. 执行测试
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(input_ids)
            torch.cuda.synchronize()  # 确保 GPU 完成计算
    end_time = time.perf_counter()
    
    # 5. 计算指标
    return calculate_metrics(...)
```

#### 3.2 带缓存推理测试
```python
def benchmark_with_cache(model, input_ids, num_iterations):
    # 类似流程，但使用 run_with_cache
    with torch.no_grad():
        for _ in range(num_iterations):
            output, cache = model.run_with_cache(input_ids)
            torch.cuda.synchronize()
            del cache  # 及时释放缓存，避免内存累积
```

### 4. 性能测量技术细节

#### 4.1 时间测量
```python
# 使用 perf_counter 获得高精度时间
start = time.perf_counter()

# 确保 GPU 同步（重要！）
torch.cuda.synchronize()

# 执行操作
operation()

# 再次同步
torch.cuda.synchronize()

end = time.perf_counter()
elapsed = end - start
```

#### 4.2 内存测量
```python
# GPU 内存
allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
peak = torch.cuda.max_memory_allocated() / 1024 / 1024

# 重置峰值统计
torch.cuda.reset_peak_memory_stats()
```

#### 4.3 吞吐量计算
```python
total_tokens = batch_size * sequence_length * num_iterations
throughput = total_tokens / elapsed_time  # tokens/sec
```

## 关键优化

### 1. 内存管理

**问题**：GPU 内存碎片和累积
**解决方案**：
- 每次测试前调用 `torch.cuda.empty_cache()`
- 使用 `gc.collect()` 清理 Python 对象
- 测试后立即删除大对象（如 cache）

### 2. 测量准确性

**问题**：GPU 异步执行导致时间测量不准
**解决方案**：
- 所有时间测量前后调用 `torch.cuda.synchronize()`
- 使用多次迭代取平均值
- 预热模型避免首次运行的开销

### 3. 批量测试稳定性

**问题**：OOM 错误中断整个测试
**解决方案**：
- try-catch 捕获单个测试的错误
- 记录错误但继续其他配置
- 自动检测最大可用批量大小

## 扩展点

### 1. 添加新的测试类型

继承 `BaseBenchmark` 并实现 `run_single_test`：

```python
class MyCustomBenchmark(BaseBenchmark):
    def run_single_test(self, model, batch_size, sequence_length, **kwargs):
        # 实现自定义测试逻辑
        # 返回 BenchmarkResult
        pass
```

### 2. 添加新的指标

扩展 `BenchmarkResult` 数据类：
```python
# 在 BenchmarkResult 中添加
custom_metric: float = 0.0

# 在测试中设置
self.metrics_collector.add_metric("custom_metric", value)
```

### 3. 支持更多模型

TransformerLens 支持的所有模型都可以测试：
- GPT-2 系列
- GPT-Neo/GPT-J
- OPT 系列
- Pythia 系列
- 等等

## 性能分析要点

### 缓存开销分析

根据实测结果，`run_with_cache()` 的开销主要来自：

1. **内存分配**：为每层的激活值分配存储空间
2. **数据复制**：将激活值从计算图复制到缓存字典
3. **Python 开销**：字典操作和 hook 调用的 Python 开销

### 优化建议

1. **生产环境**：避免使用 `run_with_cache()`，除非必要
2. **调试分析**：仅在需要分析特定层时选择性缓存
3. **批量处理**：增大批量大小可以摊薄固定开销
4. **内存预算**：缓存会使内存使用增加 50-100%

## 未来改进方向

1. **选择性缓存测试**：测试仅缓存特定层的性能
2. **Hook 影响测试**：测量不同数量 hook 的性能影响  
3. **编译模式测试**：对比 `torch.compile` 的加速效果
4. **多 GPU 测试**：支持模型并行和数据并行测试
5. **长序列测试**：测试 KV 缓存在长序列上的表现