# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作提供指导。

## 项目概述

vLLM 是一个高性能的大语言模型（LLM）推理和服务引擎。它通过 PagedAttention 实现高效的 KV 缓存内存管理，具有业界领先的服务吞吐量，支持连续批处理、优化的 CUDA/HIP 内核以及各种量化方法。该项目由 vLLM 团队开发，现已成为 PyTorch 基金会的一部分。

## 构建和开发命令

### 环境配置
```bash
# 从 PyPI 安装 vLLM
pip install vllm

# 从源码构建（推荐用于开发）
pip install -e .

# 安装开发依赖
pip install -r requirements/dev.txt
pip install -r requirements/test.txt
pip install -r requirements/lint.txt

# CUDA 开发环境
pip install -r requirements/cuda.txt

# 仅 CPU 开发环境  
pip install -r requirements/cpu.txt
```

### 测试
```bash
# 运行所有测试
pytest tests/

# 运行特定类别的测试
pytest tests/core/          # 核心调度器测试
pytest tests/engine/        # 引擎测试
pytest tests/models/        # 模型测试
pytest tests/kernels/       # 内核测试
pytest tests/distributed/   # 分布式测试

# 使用特定标记运行测试
pytest -m "not distributed"  # 跳过分布式测试
pytest -m core_model         # 运行核心模型测试

# 运行基准测试
python benchmarks/benchmark_throughput.py --model <model_name>
python benchmarks/benchmark_serving.py --model <model_name>
```

### 代码质量
```bash
# 安装 pre-commit 钩子（提交前自动运行 linting）
pip install -r requirements/lint.txt
pre-commit install

# 手动格式化和 linting
ruff check vllm/
ruff format vllm/
mypy vllm/

# 类型检查
tools/mypy.sh
```

### 构建 CUDA/ROCM 扩展
```bash
# 通过环境变量设置编译选项
export MAX_JOBS=8  # 并行编译任务数
export NVCC_THREADS=2  # 每个任务的 NVCC 线程数
export CMAKE_BUILD_TYPE=RelWithDebInfo  # 或 Debug

# 清理并重新构建
python setup.py clean --all
python setup.py build_ext --inplace
```

## 架构和关键组件

### 核心组件

1. **引擎层 (`vllm/engine/`)**: 中央编排层
   - `llm_engine.py`: 用于离线推理的同步引擎
   - `async_llm_engine.py`: 用于在线服务的异步引擎
   - `arg_utils.py`: 参数解析和配置

2. **调度器 (`vllm/core/scheduler.py`)**: 请求调度和内存管理
   - 实现连续批处理
   - 管理抢占（交换/重计算模式）
   - 处理前缀缓存

3. **注意力层 (`vllm/attention/`)**: 注意力机制实现
   - 后端选择（FlashAttention、FlashInfer、xFormers）
   - PagedAttention 实现
   - KV 缓存管理

4. **工作进程 (`vllm/worker/`)**: 模型执行层
   - `model_runner.py`: GPU 模型执行
   - `worker.py`: 工作进程管理
   - `cache_engine.py`: KV 缓存操作

5. **模型执行器 (`vllm/model_executor/`)**: 模型加载和执行
   - models 目录包含所有支持的模型实现
   - 量化支持（GPTQ、AWQ、FP8、INT8 等）
   - `csrc/` 中的自定义 CUDA 内核

6. **分布式 (`vllm/distributed/`)**: 多 GPU 支持
   - 张量、流水线和数据并行
   - 自定义 all-reduce 实现
   - 用于分离式服务的 KV 缓存传输

### 请求处理流程

1. 请求到达 API 服务器（`vllm/entrypoints/`）
2. 引擎处理并验证输入
3. 调度器分配资源并批处理请求
4. 工作进程执行模型前向传播
5. 采样器生成 token
6. 输出处理器处理解码和停止条件
7. 响应返回给客户端

### 内存管理

- **PagedAttention**: KV 缓存存储在非连续块中
- **块管理器**: 分配/释放缓存块
- **交换空间**: 用于被抢占序列的 CPU 内存
- **前缀缓存**: 重用共享前缀的已计算 KV 缓存

## V1 架构（Alpha 版）

vLLM V1 引入了重大架构改进：
- 优化的执行循环，减少开销
- 零开销前缀缓存
- 增强的多模态支持
- 更清晰的代码结构

V1 特定代码位于 `vllm/v1/` 目录中。

## 添加新模型

1. 在 `vllm/model_executor/models/` 中创建模型文件
2. 在 `vllm/model_executor/models/registry.py` 中注册模型
3. 在 `tests/models/` 中添加测试
4. 更新支持模型的文档

## 关键环境变量

- `VLLM_TARGET_DEVICE`: 目标设备（cuda、rocm、cpu、tpu、xpu）
- `MAX_JOBS`: 并行编译任务数
- `VLLM_CPU_KVCACHE_SPACE`: CPU 交换空间大小（GiB）
- `VLLM_GPU_MEMORY_UTILIZATION`: 使用的 GPU 内存比例（0-1）
- `VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT`: 启用抢占测试
- `VLLM_TRACE_FUNCTION`: 启用函数调用跟踪

## 性能优化

- 使用 `--enable-cuda-graph` 进行 CUDA 图优化
- 适当配置 `--max-num-seqs` 和 `--max-model-len`
- 为共享提示前缀启用 `--enable-prefix-caching`
- 在内存受限的部署中使用量化
- 使用 `vllm/profiler/` 工具进行性能分析

## 常见问题及解决方案

1. **内存溢出错误**: 减少 `--max-model-len` 或 `--gpu-memory-utilization`
2. **生成速度慢**: 检查是否使用了最优的注意力后端
3. **导入错误**: 确保所有依赖项正确安装
4. **CUDA 错误**: 验证 CUDA 版本与 PyTorch 的兼容性

## 贡献指南

- 遵循现有的代码模式和约定
- 为新功能添加全面的测试
- 更新 API 变更的文档
- 提交 PR 前运行 pre-commit 钩子
- 检查 GitHub PR 上的 CI 状态