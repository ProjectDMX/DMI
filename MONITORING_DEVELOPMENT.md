# vLLM 监控系统开发文档

## 目录
1. [概述](#概述)
2. [vLLM现有监控能力调查](#vllm现有监控能力调查)
3. [现状与理想差距分析](#现状与理想差距分析)
4. [监控系统架构设计](#监控系统架构设计)
5. [自动Buffer大小计算系统](#自动buffer大小计算系统)
6. [核心组件设计](#核心组件设计)
7. [需要修改的模块](#需要修改的模块)
8. [实现细节](#实现细节)
9. [配置系统](#配置系统)
10. [Benchmark方案](#benchmark方案)
11. [开发路线图](#开发路线图)
12. [附录：未来优化方向](#附录未来优化方向)

## 概述

### 项目目标
构建一个高性能的监控系统，用于记录vLLM模型推理过程中的所有中间状态和激活值。系统设计原则：
- **完整性优先**：首先确保能捕获所有需要的数据
- **自动化配置**：根据模型架构自动计算buffer大小
- **异步非阻塞**：使用GPU buffer和异步传输避免影响推理性能
- **可扩展性**：为未来的压缩和优化预留接口

### 核心要求
- 能够捕获模型推理过程中的所有中间状态（激活值、注意力权重、KV缓存等）
- 自动根据模型架构计算所需buffer大小，确保至少能存储一个完整层的激活值
- 使用异步传输机制，目标是对推理吞吐量的影响 < 5%
- 提供灵活的配置选项，支持不同的监控粒度

## vLLM现有监控能力调查

### 1. 现有监控基础设施

#### 1.1 Metrics系统 (`vllm/engine/metrics.py`)
```python
class Metrics:
    # 系统级指标
    - gauge_scheduler_running  # GPU上运行的请求数
    - gauge_scheduler_waiting  # 等待处理的请求数
    - gauge_gpu_cache_usage    # GPU KV缓存使用率
    
    # 性能指标
    - histogram_time_to_first_token   # 首token延迟
    - histogram_inter_token_latency   # token间延迟
    - histogram_e2e_time_request      # 端到端请求延迟
```
**局限性**：仅收集时间和计数指标，不保存模型内部状态

#### 1.2 ObservabilityConfig (`vllm/config/__init__.py:3129`)
```python
@dataclass
class ObservabilityConfig:
    """当前仅支持时间收集"""
    collect_model_forward_time: bool  # 收集前向传播时间
    collect_model_execute_time: bool  # 收集执行时间
```
**实现方式**：
- 使用`torch.cuda.Event(enable_timing=True)`进行GPU时间测量
- 通过`IntermediateTensors`在pipeline stages间传递时间信息
- 见`vllm/worker/model_runner.py:1664-1686`

#### 1.3 IntermediateTensors (`vllm/sequence.py:1094`)
```python
class IntermediateTensors:
    """Pipeline并行的中间状态容器"""
    tensors: dict[str, torch.Tensor]  # 可存储任意tensor
    
    # 已用于传递时间信息
    # 可扩展用于传递激活值
```
**优势**：已有跨节点传输机制，可直接扩展

#### 1.4 RequestMetrics (`vllm/sequence.py:82`)
```python
@dataclass
class RequestMetrics:
    arrival_time: float
    first_token_time: Optional[float]
    model_forward_time: Optional[float]  # 已支持
    model_execute_time: Optional[float]  # 已支持
```
**局限性**：仅记录时间，不记录中间状态

### 2. 关键观察点

#### 2.1 模型执行入口
`vllm/worker/model_runner.py:1672` - 统一的前向传播入口
```python
hidden_or_intermediate_states = model_executable(
    input_ids=model_input.input_tokens,
    positions=model_input.input_positions,
    intermediate_tensors=intermediate_tensors,
    **kwargs
)
```
**这是添加监控的最佳切入点**

#### 2.2 现有的时间监控模式
```python
# model_runner.py:1664-1687
if self.observability_config.collect_model_forward_time:
    model_forward_start = torch.cuda.Event(enable_timing=True)
    model_forward_end = torch.cuda.Event(enable_timing=True)
    model_forward_start.record()
    
# ... 模型执行 ...

if self.observability_config.collect_model_forward_time:
    model_forward_end.record()
    # 时间通过IntermediateTensors传递
    hidden_or_intermediate_states.tensors["model_forward_time"] = ...
```
**可以复用这个模式添加激活值收集**

## 现状与理想差距分析

### 功能对比表

| 功能类别 | 子功能 | vLLM现状 | 理想状态 | 实现难度 | 优先级 |
|---------|--------|----------|----------|---------|--------|
| **时间监控** | 请求级时间 | ✅ 完整支持 | ✅ 保持 | - | - |
| | 模型前向时间 | ✅ 支持 | ✅ 保持 | - | - |
| | 层级时间 | ❌ 不支持 | ✅ 每层耗时 | 低 | P2 |
| **激活值保存** | 层输入/输出 | ❌ 无 | ✅ 完整tensor | 中 | P0 |
| | 注意力输出 | ❌ 无 | ✅ 完整tensor | 中 | P0 |
| | MLP输出 | ❌ 无 | ✅ 完整tensor | 中 | P0 |
| | 残差连接 | ❌ 无 | ✅ 可选保存 | 低 | P1 |
| **注意力机制** | 注意力分数 | ❌ 无 | ✅ attention scores | 中 | P1 |
| | 注意力权重 | ❌ 无 | ✅ attention weights | 中 | P1 |
| | 注意力模式 | ❌ 无 | ✅ attention patterns | 中 | P2 |
| **KV缓存** | 缓存内容 | ⚠️ 仅管理 | ✅ 可导出内容 | 低 | P1 |
| | 缓存统计 | ⚠️ 仅使用率 | ✅ 详细统计 | 低 | P2 |
| **数据传输** | GPU→CPU | ❌ 无专门机制 | ✅ 异步传输 | 高 | P0 |
| | 缓冲管理 | ❌ 无 | ✅ 环形缓冲 | 中 | P0 |
| | 背压控制 | ❌ 无 | ✅ 自动降级 | 中 | P1 |
| **配置能力** | 自动配置 | ❌ 无 | ✅ 根据模型自动 | 中 | P0 |
| | 选择性监控 | ❌ 无 | ✅ 层/类型选择 | 低 | P1 |

### 实现策略

1. **利用现有机制**
   - 扩展`IntermediateTensors`用于激活值传递
   - 复用`ObservabilityConfig`模式添加新配置项
   - 基于现有的CUDA Event机制进行同步

2. **最小侵入原则**
   - 主要修改集中在`model_runner.py`
   - 通过配置系统控制是否启用
   - 不影响现有功能

## 监控系统架构设计

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      vLLM推理引擎                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               Model Forward Pass                      │  │
│  │                                                       │  │
│  │  Input → Layer1 → Layer2 → ... → LayerN → Output    │  │
│  │           ↓        ↓              ↓                  │  │
│  │      [Hook Points - 激活值捕获点]                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Monitoring Engine (监控引擎)                │  │
│  │                                                       │  │
│  │  • 自动Buffer大小计算                                 │  │
│  │  • 激活值收集器管理                                   │  │
│  │  • 异步传输调度                                       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   GPU Buffer System                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Auto-sized Ring Buffer (环形缓冲区)          │  │
│  │                                                       │  │
│  │  Size = max(2 * max_layer_size, user_config)         │  │
│  │                                                       │  │
│  │  [Slot 0] [Slot 1] [Slot 2] ... [Slot N]            │  │
│  │     ↓        ↓        ↓            ↓                 │  │
│  │  [Write]  [Transfer] [Transfer] [Available]          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
                     [异步传输 - CUDA Stream]
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    CPU Memory System                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Double Buffer (双缓冲)                   │  │
│  │                                                       │  │
│  │  Buffer A: [Writing from GPU]                        │  │
│  │  Buffer B: [Persisting to Disk]                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Persistence Layer                          │
│                                                              │
│  • HDF5/Parquet格式存储                                      │
│  • 元数据索引                                                │
│  • 时间戳对齐                                                │
└─────────────────────────────────────────────────────────────┘
```

## 自动Buffer大小计算系统

### 1. 核心计算器

```python
class AutoBufferCalculator:
    """
    自动计算监控系统所需的buffer大小
    确保至少能完整存储一个transformer层的所有激活值
    """
    
    def __init__(self, model_config: ModelConfig, parallel_config: ParallelConfig):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.dtype_size = 4  # FP32，后续可配置
        
    def calculate_minimum_buffer_size(self) -> Dict[str, int]:
        """
        计算最小buffer大小
        返回详细的内存需求分解
        """
        # 基础参数
        batch_size = getattr(self.model_config, 'max_batch_size', 1)
        seq_len = self.model_config.max_model_len
        hidden_dim = self.model_config.hidden_size
        num_heads = self.model_config.num_attention_heads
        head_dim = hidden_dim // num_heads
        
        # 计算单个transformer层的激活值大小
        layer_components = {}
        
        # 1. 层输入
        layer_components['layer_input'] = (
            batch_size * seq_len * hidden_dim * self.dtype_size
        )
        
        # 2. LayerNorm后的激活
        layer_components['post_layernorm'] = (
            batch_size * seq_len * hidden_dim * self.dtype_size
        )
        
        # 3. 注意力组件
        # QKV projections
        layer_components['qkv_projections'] = (
            batch_size * seq_len * hidden_dim * 3 * self.dtype_size
        )
        
        # Attention scores (before softmax)
        layer_components['attention_scores'] = (
            batch_size * num_heads * seq_len * seq_len * self.dtype_size
        )
        
        # Attention weights (after softmax)
        layer_components['attention_weights'] = (
            batch_size * num_heads * seq_len * seq_len * self.dtype_size
        )
        
        # Attention output
        layer_components['attention_output'] = (
            batch_size * seq_len * hidden_dim * self.dtype_size
        )
        
        # 4. MLP组件
        if hasattr(self.model_config, 'intermediate_size'):
            intermediate_dim = self.model_config.intermediate_size
        else:
            intermediate_dim = hidden_dim * 4  # 默认4x
            
        # MLP intermediate activations
        layer_components['mlp_intermediate'] = (
            batch_size * seq_len * intermediate_dim * self.dtype_size
        )
        
        # MLP output
        layer_components['mlp_output'] = (
            batch_size * seq_len * hidden_dim * self.dtype_size
        )
        
        # 5. 残差连接（2个）
        layer_components['residuals'] = (
            batch_size * seq_len * hidden_dim * 2 * self.dtype_size
        )
        
        # 计算总和
        total_per_layer = sum(layer_components.values())
        
        # 添加20%的安全边际
        safe_buffer_size = int(total_per_layer * 1.2)
        
        # 确保是CUDA对齐的大小（512字节对齐）
        aligned_size = ((safe_buffer_size + 511) // 512) * 512
        
        return {
            'min_buffer_size': aligned_size,
            'layer_breakdown': layer_components,
            'total_per_layer': total_per_layer,
            'recommended_size': aligned_size * 2,  # 双缓冲
            'max_reasonable_size': aligned_size * min(10, self.model_config.num_hidden_layers)
        }
    
    def calculate_for_specific_models(self) -> Dict[str, int]:
        """
        为特定模型架构提供精确计算
        """
        model_type = self.model_config.model_type.lower()
        
        if model_type == "llama":
            return self._calculate_llama_buffer()
        elif model_type == "mistral":
            return self._calculate_mistral_buffer()
        elif model_type == "mixtral":
            return self._calculate_mixtral_buffer()  # MoE需要特殊处理
        elif model_type == "gpt2":
            return self._calculate_gpt2_buffer()
        else:
            # 使用通用计算
            return self.calculate_minimum_buffer_size()
    
    def _calculate_llama_buffer(self) -> Dict[str, int]:
        """LLaMA特定的计算（考虑RMSNorm、SwiGLU等）"""
        base_calc = self.calculate_minimum_buffer_size()
        
        # LLaMA使用SwiGLU，需要额外的gate projection
        batch_size = getattr(self.model_config, 'max_batch_size', 1)
        seq_len = self.model_config.max_model_len
        intermediate_dim = self.model_config.intermediate_size
        
        additional_memory = {
            'gate_proj': batch_size * seq_len * intermediate_dim * self.dtype_size,
            'up_proj': batch_size * seq_len * intermediate_dim * self.dtype_size,
        }
        
        base_calc['layer_breakdown'].update(additional_memory)
        base_calc['total_per_layer'] += sum(additional_memory.values())
        base_calc['min_buffer_size'] = int(base_calc['total_per_layer'] * 1.2)
        
        return base_calc
    
    def _calculate_mixtral_buffer(self) -> Dict[str, int]:
        """Mixtral (MoE)特定的计算"""
        base_calc = self.calculate_minimum_buffer_size()
        
        # MoE额外需要：
        # 1. Router logits
        # 2. Expert选择的激活值
        # 3. 多个expert的输出
        
        batch_size = getattr(self.model_config, 'max_batch_size', 1)
        seq_len = self.model_config.max_model_len
        hidden_dim = self.model_config.hidden_size
        num_experts = self.model_config.num_local_experts
        top_k = self.model_config.num_experts_per_tok
        
        moe_memory = {
            'router_logits': batch_size * seq_len * num_experts * self.dtype_size,
            'selected_experts': batch_size * seq_len * top_k * hidden_dim * self.dtype_size,
            'expert_outputs': batch_size * seq_len * top_k * hidden_dim * self.dtype_size,
        }
        
        base_calc['layer_breakdown'].update(moe_memory)
        base_calc['total_per_layer'] += sum(moe_memory.values())
        base_calc['min_buffer_size'] = int(base_calc['total_per_layer'] * 1.2)
        
        return base_calc
```

### 2. 动态Buffer管理器

```python
class DynamicGPUBuffer:
    """
    动态GPU缓冲区管理
    自动调整大小，确保能容纳激活值
    """
    
    def __init__(self, model_config: ModelConfig, initial_size: Optional[int] = None):
        self.model_config = model_config
        self.calculator = AutoBufferCalculator(model_config)
        
        # 计算buffer大小
        calc_result = self.calculator.calculate_minimum_buffer_size()
        self.min_size = calc_result['min_buffer_size']
        self.recommended_size = calc_result['recommended_size']
        
        # 使用初始大小（如果提供）或推荐大小
        if initial_size and initial_size >= self.min_size:
            self.current_size = initial_size
        else:
            if initial_size and initial_size < self.min_size:
                logger.warning(
                    f"Requested buffer size {initial_size} is smaller than "
                    f"minimum required {self.min_size}. Using minimum."
                )
            self.current_size = self.recommended_size
        
        # 分配GPU内存
        self._allocate_buffer()
        
        # 统计信息
        self.stats = BufferStatistics()
        
    def _allocate_buffer(self):
        """分配GPU内存"""
        try:
            # 尝试分配连续内存
            self.buffer = torch.cuda.allocate_shared_memory(
                self.current_size,
                device=torch.cuda.current_device()
            )
            self.buffer_view = self.buffer.view(torch.uint8)
            logger.info(f"Allocated GPU buffer: {self.current_size / (1024**3):.2f} GB")
        except torch.cuda.OutOfMemoryError:
            # 如果失败，尝试分配最小大小
            if self.current_size > self.min_size:
                logger.warning(
                    f"Failed to allocate {self.current_size} bytes, "
                    f"falling back to minimum size {self.min_size}"
                )
                self.current_size = self.min_size
                self._allocate_buffer()
            else:
                raise RuntimeError(
                    f"Cannot allocate minimum required buffer of {self.min_size} bytes"
                )
    
    def can_fit(self, tensor_size: int) -> bool:
        """检查是否能容纳指定大小的tensor"""
        return self.write_position + tensor_size <= self.current_size
    
    def write_activation(self, name: str, tensor: torch.Tensor, metadata: Dict):
        """
        写入激活值到buffer
        如果空间不足，自动处理（刷新或扩容）
        """
        tensor_bytes = tensor.numel() * tensor.element_size()
        
        if not self.can_fit(tensor_bytes):
            # 策略1：如果单个tensor就超过buffer，必须扩容
            if tensor_bytes > self.current_size:
                self._resize_buffer(tensor_bytes * 1.5)
            # 策略2：否则先刷新现有数据
            else:
                self._flush_buffer()
                self.write_position = 0
        
        # 写入数据
        self._write_tensor_to_buffer(tensor, metadata)
        
        # 更新统计
        self.stats.record_write(name, tensor_bytes)
    
    def _resize_buffer(self, new_size: int):
        """动态调整buffer大小"""
        # 检查GPU可用内存
        available_memory = torch.cuda.get_device_properties(0).total_memory
        used_memory = torch.cuda.memory_allocated()
        free_memory = available_memory - used_memory
        
        if new_size > free_memory * 0.8:  # 最多使用80%的空闲内存
            raise RuntimeError(
                f"Cannot resize buffer to {new_size}, only {free_memory} available"
            )
        
        logger.info(f"Resizing buffer from {self.current_size} to {new_size}")
        
        # 保存现有数据
        old_data = self.buffer[:self.write_position].clone()
        
        # 释放旧buffer
        del self.buffer
        torch.cuda.empty_cache()
        
        # 分配新buffer
        self.current_size = new_size
        self._allocate_buffer()
        
        # 恢复数据
        self.buffer[:self.write_position] = old_data
        
        self.stats.resize_count += 1
```

### 3. 智能配置建议器

```python
class MonitoringConfigAdvisor:
    """
    根据模型和使用场景提供监控配置建议
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.calculator = AutoBufferCalculator(model_config)
        
    def recommend_config(self, use_case: str = "research") -> MonitoringConfig:
        """
        根据使用场景推荐配置
        """
        calc_result = self.calculator.calculate_minimum_buffer_size()
        
        configs = {
            "research": {
                "description": "完整数据捕获，用于研究分析",
                "buffer_size": calc_result['recommended_size'] * 5,
                "save_all_activations": True,
                "save_attention_weights": True,
                "save_kv_cache": True,
                "sampling_frequency": 1,  # 每个token
            },
            "debugging": {
                "description": "调试模式，关注关键层",
                "buffer_size": calc_result['recommended_size'] * 2,
                "save_all_activations": False,
                "save_attention_weights": True,
                "save_kv_cache": False,
                "sampling_frequency": 10,  # 每10个token
                "target_layers": "auto",  # 自动选择关键层
            },
            "production": {
                "description": "生产环境，最小开销",
                "buffer_size": calc_result['min_buffer_size'],
                "save_all_activations": False,
                "save_attention_weights": False,
                "save_kv_cache": False,
                "sampling_frequency": 100,  # 每100个token
                "statistics_only": True,  # 只保存统计信息
            },
            "profiling": {
                "description": "性能分析，平衡数据和开销",
                "buffer_size": calc_result['recommended_size'],
                "save_all_activations": True,
                "save_attention_weights": False,
                "save_kv_cache": False,
                "sampling_frequency": 50,
            }
        }
        
        config = configs.get(use_case, configs["research"])
        
        # 添加模型特定的建议
        if self.model_config.num_hidden_layers > 40:
            # 大模型建议
            config["sampling_layers"] = list(range(0, self.model_config.num_hidden_layers, 5))
            logger.info(f"Large model detected, sampling every 5th layer")
        
        return MonitoringConfig(**config)
    
    def estimate_memory_usage(self, config: MonitoringConfig) -> Dict[str, float]:
        """
        估算给定配置的内存使用
        """
        calc_result = self.calculator.calculate_minimum_buffer_size()
        
        estimates = {
            "gpu_buffer_mb": config.buffer_size / (1024**2),
            "cpu_buffer_mb": config.buffer_size * 2 / (1024**2),  # 双缓冲
            "disk_per_hour_gb": 0,
        }
        
        if config.save_all_activations:
            # 估算磁盘使用
            tokens_per_second = 100  # 假设
            bytes_per_token = calc_result['total_per_layer'] / self.model_config.max_model_len
            
            if config.sampling_frequency:
                bytes_per_token /= config.sampling_frequency
            
            bytes_per_hour = bytes_per_token * tokens_per_second * 3600
            estimates["disk_per_hour_gb"] = bytes_per_hour / (1024**3)
        
        return estimates
```

## 核心组件设计

### 1. MonitoringConfig类

```python
@dataclass
class MonitoringConfig:
    """
    监控系统配置
    支持自动计算和手动覆盖
    """
    
    # 基础配置
    enabled: bool = False
    output_dir: str = "./monitoring_data"
    
    # Buffer配置（自动计算或手动指定）
    buffer_size: Optional[int] = None  # None表示自动计算
    buffer_strategy: str = "auto"  # auto, fixed, dynamic
    
    # 监控内容配置
    save_activations: bool = True
    save_attention_weights: bool = False
    save_attention_scores: bool = False
    save_kv_cache: bool = False
    save_gradients: bool = False  # 仅训练模式
    
    # 采样配置
    sampling_frequency: int = 1  # 每N个token采样一次
    sampling_layers: Optional[List[int]] = None  # None表示所有层
    sampling_strategy: str = "uniform"  # uniform, adaptive, critical
    
    # 性能配置
    async_transfer: bool = True
    num_transfer_streams: int = 2
    cpu_buffer_factor: float = 2.0  # CPU buffer是GPU buffer的倍数
    
    # 高级配置
    enable_statistics: bool = True  # 收集统计信息
    enable_compression: bool = False  # 暂不实现
    profile_overhead: bool = False  # 分析监控开销
    
    def __post_init__(self):
        """初始化后处理，自动计算未指定的值"""
        if self.buffer_size is None and self.buffer_strategy == "auto":
            # 将在MonitoringEngine初始化时根据模型计算
            self.buffer_size = "auto"
```

### 2. MonitoringEngine类

```python
class MonitoringEngine:
    """
    监控引擎主类
    负责协调所有监控组件
    """
    
    def __init__(self, 
                 model_config: ModelConfig,
                 monitoring_config: MonitoringConfig,
                 parallel_config: Optional[ParallelConfig] = None):
        
        self.model_config = model_config
        self.monitoring_config = monitoring_config
        self.parallel_config = parallel_config
        
        # 自动计算buffer大小（如果需要）
        if monitoring_config.buffer_size == "auto":
            self._auto_configure_buffer()
        
        # 初始化组件
        self.gpu_buffer = DynamicGPUBuffer(
            model_config=model_config,
            initial_size=self.monitoring_config.buffer_size
        )
        
        self.async_transfer = AsyncTransferManager(
            num_streams=monitoring_config.num_transfer_streams
        )
        
        self.collectors = {}
        self.statistics = MonitoringStatistics()
        
        # 注册清理函数
        import atexit
        atexit.register(self.cleanup)
        
    def _auto_configure_buffer(self):
        """自动配置buffer大小"""
        calculator = AutoBufferCalculator(self.model_config, self.parallel_config)
        calc_result = calculator.calculate_minimum_buffer_size()
        
        # 根据可用GPU内存调整
        gpu_props = torch.cuda.get_device_properties(0)
        available_memory = gpu_props.total_memory - torch.cuda.memory_allocated()
        
        # 使用推荐大小，但不超过可用内存的25%
        recommended = calc_result['recommended_size']
        max_allowed = int(available_memory * 0.25)
        
        self.monitoring_config.buffer_size = min(recommended, max_allowed)
        
        logger.info(
            f"Auto-configured buffer size: {self.monitoring_config.buffer_size / (1024**3):.2f} GB "
            f"(min: {calc_result['min_buffer_size'] / (1024**3):.2f} GB, "
            f"recommended: {recommended / (1024**3):.2f} GB)"
        )
    
    def register_model(self, model: nn.Module):
        """
        在模型上注册监控钩子
        """
        hook_count = 0
        
        for name, module in model.named_modules():
            if self._should_monitor_module(name, module):
                collector = ActivationCollector(
                    module_name=name,
                    module_type=type(module).__name__,
                    gpu_buffer=self.gpu_buffer,
                    config=self.monitoring_config
                )
                
                # 注册前向钩子
                handle = module.register_forward_hook(collector.forward_hook)
                collector.hook_handle = handle
                
                self.collectors[name] = collector
                hook_count += 1
        
        logger.info(f"Registered {hook_count} monitoring hooks")
        
    def _should_monitor_module(self, name: str, module: nn.Module) -> bool:
        """
        判断是否应该监控该模块
        """
        # 检查层索引
        if self.monitoring_config.sampling_layers is not None:
            layer_idx = self._extract_layer_index(name)
            if layer_idx not in self.monitoring_config.sampling_layers:
                return False
        
        # 检查模块类型
        monitor_types = []
        
        if self.monitoring_config.save_activations:
            monitor_types.extend(['Linear', 'LayerNorm', 'RMSNorm'])
        
        if self.monitoring_config.save_attention_weights:
            monitor_types.extend(['Attention', 'MultiheadAttention'])
        
        module_type = type(module).__name__
        return any(t in module_type for t in monitor_types)
    
    def start_monitoring(self):
        """开始监控"""
        self.async_transfer.start()
        self.monitoring_active = True
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        self.flush_all_buffers()
        self.async_transfer.stop()
        logger.info("Monitoring stopped")
    
    def flush_all_buffers(self):
        """刷新所有缓冲区"""
        # 刷新GPU buffer
        if self.gpu_buffer.write_position > 0:
            self.async_transfer.transfer_batch(
                self.gpu_buffer.get_current_data()
            )
            self.gpu_buffer.reset()
        
        # 等待所有传输完成
        self.async_transfer.wait_all_transfers()
        
        # 刷新CPU buffers到磁盘
        self.async_transfer.flush_to_disk()
```

### 3. ActivationCollector类

```python
class ActivationCollector:
    """
    激活值收集器
    作为PyTorch hook使用
    """
    
    def __init__(self, 
                 module_name: str,
                 module_type: str,
                 gpu_buffer: DynamicGPUBuffer,
                 config: MonitoringConfig):
        
        self.module_name = module_name
        self.module_type = module_type
        self.gpu_buffer = gpu_buffer
        self.config = config
        
        self.token_count = 0
        self.hook_handle = None
        
    def forward_hook(self, module, input, output):
        """
        前向传播钩子
        注意：这个函数需要非常高效，避免影响推理性能
        """
        # 检查采样频率
        self.token_count += 1
        if self.token_count % self.config.sampling_frequency != 0:
            return
        
        # 准备元数据
        metadata = {
            'module_name': self.module_name,
            'module_type': self.module_type,
            'token_count': self.token_count,
            'timestamp': time.perf_counter(),
        }
        
        # 处理不同类型的输出
        if isinstance(output, torch.Tensor):
            self._save_tensor(output, 'output', metadata)
        elif isinstance(output, tuple):
            for i, tensor in enumerate(output):
                if isinstance(tensor, torch.Tensor):
                    self._save_tensor(tensor, f'output_{i}', metadata)
        
        # 如果需要保存输入
        if self.config.save_inputs:
            if isinstance(input, torch.Tensor):
                self._save_tensor(input, 'input', metadata)
            elif isinstance(input, tuple):
                for i, tensor in enumerate(input):
                    if isinstance(tensor, torch.Tensor):
                        self._save_tensor(tensor, f'input_{i}', metadata)
    
    def _save_tensor(self, tensor: torch.Tensor, name: str, metadata: dict):
        """
        保存tensor到GPU buffer
        """
        # 添加tensor信息到元数据
        metadata.update({
            'tensor_name': name,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
        })
        
        # 写入到GPU buffer
        self.gpu_buffer.write_activation(
            name=f"{self.module_name}.{name}",
            tensor=tensor,
            metadata=metadata
        )
```

### 4. AsyncTransferManager类

```python
class AsyncTransferManager:
    """
    异步传输管理器
    负责GPU到CPU的数据传输
    """
    
    def __init__(self, num_streams: int = 2):
        self.num_streams = num_streams
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream_idx = 0
        
        # CPU双缓冲
        self.cpu_buffers = [
            CPUBuffer(size=1024*1024*1024)  # 1GB each
            for _ in range(2)
        ]
        self.current_cpu_buffer_idx = 0
        
        # 传输队列
        self.transfer_queue = queue.Queue()
        self.transfer_thread = None
        self.stop_event = threading.Event()
        
    def start(self):
        """启动异步传输线程"""
        self.transfer_thread = threading.Thread(
            target=self._transfer_worker,
            daemon=True
        )
        self.transfer_thread.start()
        
    def stop(self):
        """停止异步传输"""
        self.stop_event.set()
        if self.transfer_thread:
            self.transfer_thread.join(timeout=5)
    
    def transfer_batch(self, gpu_data: torch.Tensor):
        """
        将一批数据加入传输队列
        """
        # 选择下一个CUDA流
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
        
        # 创建传输任务
        transfer_task = TransferTask(
            gpu_data=gpu_data,
            stream=stream,
            timestamp=time.perf_counter()
        )
        
        # 加入队列
        self.transfer_queue.put(transfer_task)
    
    def _transfer_worker(self):
        """
        后台传输工作线程
        """
        while not self.stop_event.is_set():
            try:
                # 获取传输任务
                task = self.transfer_queue.get(timeout=0.1)
                
                # 执行异步传输
                self._execute_transfer(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transfer error: {e}")
    
    def _execute_transfer(self, task: TransferTask):
        """
        执行GPU到CPU的异步传输
        """
        # 获取CPU buffer
        cpu_buffer = self.cpu_buffers[self.current_cpu_buffer_idx]
        
        # 使用指定的CUDA流进行传输
        with torch.cuda.stream(task.stream):
            # 分配pinned memory（如果需要）
            if not cpu_buffer.is_pinned:
                cpu_buffer.make_pinned()
            
            # 异步拷贝
            cpu_buffer.data.copy_(task.gpu_data, non_blocking=True)
            
            # 记录事件用于同步
            event = torch.cuda.Event()
            event.record(task.stream)
            
            # 保存元数据
            cpu_buffer.metadata = task.metadata
            cpu_buffer.event = event
        
        # 切换CPU buffer
        self.current_cpu_buffer_idx = 1 - self.current_cpu_buffer_idx
        
        # 异步等待传输完成并持久化
        self._schedule_persistence(cpu_buffer)
    
    def _schedule_persistence(self, buffer: CPUBuffer):
        """
        调度数据持久化到磁盘
        """
        # 等待传输完成
        buffer.event.wait()
        
        # 持久化到磁盘
        self._persist_to_disk(buffer)
```

### 5. 数据持久化层

```python
class DataPersistence:
    """
    数据持久化管理
    支持HDF5和Parquet格式
    """
    
    def __init__(self, output_dir: str, format: str = "hdf5"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        
        # 创建输出文件
        self.current_file = None
        self.file_counter = 0
        self.create_new_file()
        
    def create_new_file(self):
        """创建新的输出文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"activations_{timestamp}_{self.file_counter:04d}.{self.format}"
        filepath = self.output_dir / filename
        
        if self.format == "hdf5":
            self.current_file = h5py.File(filepath, 'w')
            self._setup_hdf5_structure()
        elif self.format == "parquet":
            self.current_file = ParquetWriter(filepath)
            self._setup_parquet_structure()
        
        self.file_counter += 1
        logger.info(f"Created new output file: {filepath}")
    
    def write_activation(self, name: str, data: np.ndarray, metadata: dict):
        """
        写入激活值数据
        """
        if self.format == "hdf5":
            self._write_hdf5(name, data, metadata)
        elif self.format == "parquet":
            self._write_parquet(name, data, metadata)
        
        # 检查文件大小，必要时创建新文件
        if self._should_rotate_file():
            self.close_current_file()
            self.create_new_file()
    
    def _write_hdf5(self, name: str, data: np.ndarray, metadata: dict):
        """写入HDF5格式"""
        # 创建或获取组
        group_name = name.rsplit('.', 1)[0] if '.' in name else 'root'
        if group_name not in self.current_file:
            group = self.current_file.create_group(group_name)
        else:
            group = self.current_file[group_name]
        
        # 创建数据集
        dataset_name = f"{name}_{metadata['token_count']}"
        dataset = group.create_dataset(
            dataset_name,
            data=data,
            compression='gzip',
            compression_opts=1  # 快速压缩
        )
        
        # 添加元数据
        for key, value in metadata.items():
            dataset.attrs[key] = value
    
    def _should_rotate_file(self) -> bool:
        """检查是否需要创建新文件"""
        # 文件大小限制：1GB
        max_size = 1024 * 1024 * 1024
        
        if self.format == "hdf5":
            current_size = self.current_file.id.get_filesize()
            return current_size > max_size
        
        return False
```

## 需要修改的模块

### 1. 添加监控配置 (`vllm/config/__init__.py`)

```python
# 在 VllmConfig 类中添加
@dataclass
class VllmConfig:
    # ... 现有字段 ...
    monitoring_config: Optional[MonitoringConfig] = None
    
    def __post_init__(self):
        # ... 现有初始化 ...
        
        # 初始化监控配置
        if self.monitoring_config and self.monitoring_config.enabled:
            # 自动配置buffer大小
            if self.monitoring_config.buffer_size == "auto":
                calculator = AutoBufferCalculator(self.model_config)
                calc_result = calculator.calculate_minimum_buffer_size()
                self.monitoring_config.buffer_size = calc_result['recommended_size']
```

### 2. 集成到ModelRunner (`vllm/worker/model_runner.py`)

```python
class ModelRunner:
    def __init__(self, vllm_config: VllmConfig, ...):
        # ... 现有初始化 ...
        
        # 初始化监控引擎
        self.monitoring_engine = None
        if vllm_config.monitoring_config and vllm_config.monitoring_config.enabled:
            self.monitoring_engine = MonitoringEngine(
                model_config=vllm_config.model_config,
                monitoring_config=vllm_config.monitoring_config,
                parallel_config=vllm_config.parallel_config
            )
            logger.info("Monitoring engine initialized")
    
    def load_model(self):
        # ... 现有模型加载 ...
        
        # 注册监控钩子
        if self.monitoring_engine:
            self.monitoring_engine.register_model(self.model)
            self.monitoring_engine.start_monitoring()
    
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        
        # 监控：批次开始
        if self.monitoring_engine:
            self.monitoring_engine.on_batch_start(model_input)
        
        # ... 现有执行逻辑 ...
        
        # 在 line 1672 左右，模型前向传播
        with set_forward_context(model_input.attn_metadata, 
                                self.vllm_config, virtual_engine):
            
            # 如果启用监控，包装模型执行
            if self.monitoring_engine:
                with self.monitoring_engine.monitor_forward_pass():
                    hidden_or_intermediate_states = model_executable(
                        input_ids=model_input.input_tokens,
                        inputs_embeds=model_input.inputs_embeds,
                        positions=model_input.input_positions,
                        intermediate_tensors=intermediate_tensors,
                        **kwargs
                    )
            else:
                # 原有执行方式
                hidden_or_intermediate_states = model_executable(...)
        
        # 监控：批次结束
        if self.monitoring_engine:
            self.monitoring_engine.on_batch_end()
        
        # ... 继续原有逻辑 ...
        
        return output
```

### 3. 扩展ObservabilityConfig (`vllm/config/__init__.py`)

```python
@dataclass
class ObservabilityConfig:
    # ... 现有字段 ...
    
    # 新增监控相关配置
    collect_activations: Optional[str] = None  # "all", "sample", None
    collect_attention_weights: bool = False
    collect_kv_cache: bool = False
    monitoring_backend: str = "builtin"  # builtin, custom
    
    @cached_property
    def collect_model_internals(self) -> bool:
        """是否收集模型内部状态"""
        return (self.collect_activations is not None or 
                self.collect_attention_weights or 
                self.collect_kv_cache)
```

### 4. 扩展IntermediateTensors用于监控数据传递

```python
# vllm/sequence.py
class IntermediateTensors:
    """扩展以支持监控数据"""
    
    def __init__(self, tensors: dict):
        self.tensors = tensors
        self.monitoring_data = {}  # 新增：监控数据
        
    def add_monitoring_data(self, layer_name: str, data: torch.Tensor):
        """添加监控数据"""
        if layer_name not in self.monitoring_data:
            self.monitoring_data[layer_name] = []
        self.monitoring_data[layer_name].append(data)
    
    def get_monitoring_data(self) -> dict:
        """获取所有监控数据"""
        return self.monitoring_data
```

## 实现细节

### 1. 高效的Hook实现

```python
def create_efficient_hook(monitoring_engine: MonitoringEngine, layer_name: str):
    """
    创建高效的监控钩子
    最小化对推理性能的影响
    """
    
    def hook(module, input, output):
        # 快速路径：如果不需要监控，立即返回
        if not monitoring_engine.should_collect_this_step():
            return
        
        # 使用no_grad避免梯度计算
        with torch.no_grad():
            # 异步记录CUDA事件
            event = torch.cuda.Event()
            event.record()
            
            # 准备元数据（轻量级）
            metadata = {
                'layer': layer_name,
                'event': event,
                'shape': output.shape if isinstance(output, torch.Tensor) else None
            }
            
            # 异步拷贝到buffer（不等待）
            if isinstance(output, torch.Tensor):
                monitoring_engine.gpu_buffer.write_async(
                    tensor=output,
                    metadata=metadata,
                    stream=monitoring_engine.get_next_stream()
                )
    
    return hook
```

### 2. 内存对齐和优化

```python
class MemoryOptimizedBuffer:
    """
    内存优化的buffer实现
    """
    
    def __init__(self, size: int):
        # 确保512字节对齐（CUDA优化）
        self.aligned_size = ((size + 511) // 512) * 512
        
        # 使用CUDA统一内存
        self.buffer = torch.cuda.caching_allocator_alloc(
            self.aligned_size,
            device=torch.cuda.current_device(),
            stream=torch.cuda.current_stream()
        )
        
        # 预分配元数据存储
        self.metadata_buffer = []
        self.metadata_capacity = 10000
        
    def write_aligned(self, tensor: torch.Tensor):
        """
        对齐写入，优化内存访问
        """
        # 计算对齐的大小
        tensor_size = tensor.numel() * tensor.element_size()
        aligned_tensor_size = ((tensor_size + 127) // 128) * 128  # 128字节对齐
        
        # 确保有足够空间
        if self.write_pos + aligned_tensor_size > self.aligned_size:
            return False
        
        # 执行对齐拷贝
        self.buffer[self.write_pos:self.write_pos + tensor_size] = tensor.flatten()
        self.write_pos += aligned_tensor_size  # 使用对齐后的大小
        
        return True
```

### 3. 批处理优化

```python
class BatchedTransfer:
    """
    批量传输优化
    减少传输开销
    """
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.pending_transfers = []
        
    def add_transfer(self, data: torch.Tensor, metadata: dict):
        """添加到待传输批次"""
        self.pending_transfers.append((data, metadata))
        
        # 达到批次大小时触发传输
        if len(self.pending_transfers) >= self.batch_size:
            self.flush_batch()
    
    def flush_batch(self):
        """批量传输"""
        if not self.pending_transfers:
            return
        
        # 合并所有tensor
        tensors = [t for t, _ in self.pending_transfers]
        metadatas = [m for _, m in self.pending_transfers]
        
        # 计算总大小
        total_size = sum(t.numel() * t.element_size() for t in tensors)
        
        # 分配连续内存
        combined_buffer = torch.empty(total_size, dtype=torch.uint8, device='cuda')
        
        # 批量拷贝
        offset = 0
        for tensor in tensors:
            size = tensor.numel() * tensor.element_size()
            combined_buffer[offset:offset+size] = tensor.view(-1, dtype=torch.uint8)
            offset += size
        
        # 单次异步传输
        self._async_transfer_combined(combined_buffer, metadatas)
        
        # 清空批次
        self.pending_transfers.clear()
```

## 配置系统

### 1. 配置文件格式

```yaml
# monitoring_config.yaml
monitoring:
  # 基础配置
  enabled: true
  output_dir: "./monitoring_data"
  output_format: "hdf5"  # hdf5, parquet
  
  # 自动配置
  auto_configure: true  # 自动计算buffer大小
  use_case: "research"  # research, debugging, production, profiling
  
  # 手动配置（可选，覆盖自动配置）
  buffer:
    size: null  # null表示自动计算
    strategy: "dynamic"  # fixed, dynamic, adaptive
    
  # 监控内容
  collection:
    activations: true
    attention_weights: false
    attention_scores: false
    kv_cache: false
    statistics: true
    
  # 采样策略
  sampling:
    frequency: 1  # 每N个token
    layers: null  # null表示所有层，或指定[0, 5, 10, 15]
    strategy: "uniform"  # uniform, adaptive, critical
    
  # 性能优化
  performance:
    async_transfer: true
    num_streams: 2
    batch_size: 32
    compression: false  # 暂不实现
    
  # 调试选项
  debug:
    profile_overhead: false
    validate_data: false
    verbose_logging: false
```

### 2. 环境变量支持

```bash
# 覆盖配置文件
export VLLM_MONITORING_ENABLED=1
export VLLM_MONITORING_BUFFER_SIZE=auto
export VLLM_MONITORING_OUTPUT_DIR=/data/monitoring
export VLLM_MONITORING_SAMPLING_FREQ=10
```

### 3. 命令行参数

```python
# 在 vllm/engine/arg_utils.py 中添加
parser.add_argument(
    '--enable-monitoring',
    action='store_true',
    help='Enable activation monitoring'
)
parser.add_argument(
    '--monitoring-config',
    type=str,
    help='Path to monitoring configuration file'
)
parser.add_argument(
    '--monitoring-buffer-size',
    type=str,
    default='auto',
    help='GPU buffer size for monitoring (auto, or size in GB)'
)
```

## Benchmark方案

### 1. 性能测试框架

```python
class MonitoringBenchmark:
    """
    监控系统性能测试
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {}
        
    def run_comprehensive_benchmark(self):
        """运行完整的基准测试"""
        
        # 1. 基线测试（无监控）
        baseline = self.run_baseline()
        
        # 2. 不同监控配置的测试
        configs = [
            ("minimal", {"sampling_frequency": 100, "save_statistics_only": True}),
            ("sampling", {"sampling_frequency": 10, "save_activations": True}),
            ("full", {"sampling_frequency": 1, "save_all": True}),
        ]
        
        for name, config in configs:
            result = self.run_with_monitoring(config)
            self.results[name] = self.calculate_overhead(baseline, result)
        
        # 3. 生成报告
        self.generate_report()
    
    def run_baseline(self) -> BenchmarkResult:
        """运行基线测试"""
        config = create_engine_config(
            model=self.model_name,
            monitoring_enabled=False
        )
        
        engine = LLMEngine(config)
        
        # 预热
        self.warmup(engine, num_iterations=10)
        
        # 测试
        start_time = time.perf_counter()
        tokens_generated = 0
        
        for _ in range(100):
            output = engine.generate(
                prompts=["Test prompt"] * 32,  # batch_size=32
                sampling_params=SamplingParams(max_tokens=128)
            )
            tokens_generated += sum(len(o.token_ids) for o in output)
        
        elapsed_time = time.perf_counter() - start_time
        
        return BenchmarkResult(
            throughput=tokens_generated / elapsed_time,
            latency=elapsed_time / 100,
            memory_used=torch.cuda.max_memory_allocated()
        )
    
    def calculate_overhead(self, baseline: BenchmarkResult, 
                          monitored: BenchmarkResult) -> dict:
        """计算监控开销"""
        return {
            "throughput_loss": (
                (baseline.throughput - monitored.throughput) / 
                baseline.throughput * 100
            ),
            "latency_increase": (
                (monitored.latency - baseline.latency) / 
                baseline.latency * 100
            ),
            "memory_overhead": monitored.memory_used - baseline.memory_used,
        }
    
    def generate_report(self):
        """生成性能报告"""
        report = f"""
        Monitoring Performance Report
        ============================
        Model: {self.model_name}
        
        Results:
        --------
        """
        
        for config_name, overhead in self.results.items():
            report += f"""
        Configuration: {config_name}
        - Throughput Loss: {overhead['throughput_loss']:.2f}%
        - Latency Increase: {overhead['latency_increase']:.2f}%
        - Memory Overhead: {overhead['memory_overhead'] / (1024**3):.2f} GB
            """
        
        print(report)
        
        # 保存到文件
        with open(f"benchmark_{self.model_name}.txt", "w") as f:
            f.write(report)
```

### 2. 自动化测试脚本

```bash
#!/bin/bash
# run_monitoring_benchmark.sh

# 测试不同模型
models=("gpt2" "llama-7b" "llama-13b")
configs=("minimal" "sampling" "full")

for model in "${models[@]}"; do
    echo "Testing model: $model"
    
    # 基线测试
    python -m vllm.benchmarks.benchmark_monitoring \
        --model $model \
        --no-monitoring \
        --output baseline_${model}.json
    
    # 监控测试
    for config in "${configs[@]}"; do
        python -m vllm.benchmarks.benchmark_monitoring \
            --model $model \
            --monitoring-config configs/${config}.yaml \
            --output monitoring_${model}_${config}.json
    done
    
    # 生成对比报告
    python -m vllm.benchmarks.generate_report \
        --baseline baseline_${model}.json \
        --monitoring monitoring_${model}_*.json \
        --output report_${model}.html
done

# 汇总报告
python -m vllm.benchmarks.summary_report \
    --reports report_*.html \
    --output summary.html
```

## 开发路线图

### Phase 0: 基础准备（Week 1）
- [x] 调查vLLM现有监控能力
- [ ] 设计自动buffer计算系统
- [ ] 创建项目结构和基础类

### Phase 1: 核心功能实现（Week 2-3）
- [ ] 实现AutoBufferCalculator
- [ ] 实现DynamicGPUBuffer
- [ ] 实现MonitoringEngine基础框架
- [ ] 实现ActivationCollector

### Phase 2: 集成到vLLM（Week 4-5）
- [ ] 修改VllmConfig添加监控配置
- [ ] 集成到ModelRunner
- [ ] 添加命令行参数支持
- [ ] 实现配置文件解析

### Phase 3: 异步传输系统（Week 6-7）
- [ ] 实现AsyncTransferManager
- [ ] 实现CPU双缓冲机制
- [ ] 优化GPU-CPU传输
- [ ] 实现背压控制

### Phase 4: 数据持久化（Week 8）
- [ ] 实现HDF5存储后端
- [ ] 实现Parquet存储后端
- [ ] 实现文件轮转机制
- [ ] 添加元数据管理

### Phase 5: 测试和优化（Week 9-10）
- [ ] 编写单元测试
- [ ] 实现benchmark框架
- [ ] 性能调优
- [ ] 内存泄漏检查
- [ ] 文档编写

### Phase 6: 高级功能（Future）
- [ ] 实现自适应采样
- [ ] 添加在线分析功能
- [ ] 实现分布式监控
- [ ] 开发可视化工具

## 附录：未来优化方向

### A. 压缩技术（Phase 2实现）

虽然初期实现不考虑压缩，但预留了接口。未来可以实现：

1. **量化压缩**
   - FP32 → FP16/BF16（50%压缩率）
   - INT8量化（75%压缩率）
   - 混合精度策略

2. **稀疏性压缩**
   - Top-K稀疏化
   - 结构化稀疏（2:4稀疏）
   - 动态阈值稀疏

3. **张量压缩**
   - 低秩分解（SVD）
   - Tucker分解
   - Tensor-Train分解

4. **无损压缩**
   - Zstandard（高压缩率）
   - LZ4（高速压缩）
   - Snappy（超低延迟）

5. **差分编码**
   - 时间差分（只保存变化）
   - 层间差分（相似层共享）

### B. 智能采样策略

1. **自适应采样**
   - 根据激活值变化率调整
   - 关键token的识别
   - 异常检测触发

2. **重要性采样**
   - 注意力权重引导
   - 梯度幅度引导
   - 不确定性引导

### C. 分布式监控

1. **多GPU协调**
   - 分布式buffer管理
   - 跨节点数据聚合
   - 时间同步机制

2. **流水线并行支持**
   - Stage间数据传递
   - 统一时间线构建

### D. 在线分析

1. **实时统计**
   - 激活值分布
   - 注意力模式
   - 异常检测

2. **实时可视化**
   - Web界面
   - Tensorboard集成
   - Weights & Biases集成

---

*文档版本: v2.0*
*最后更新: 2024*
*作者: vLLM Monitoring Team*