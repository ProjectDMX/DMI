# vLLM-Prometheus 监控系统开发对话总结

## 项目目标
- 为 HF 主流模型实现高效的激活值监控
- 最终集成到 vLLM 推理系统中
- 支持 CUDA Graph 模式下的零性能开销监控

## 当前进度 (performant-monitor 分支)

### 已完成
1. **C++ 核心实现**：
   - `GraphNativeDelegate`: metadata 解析和 native backend 提交
   - `parse_shadow_block`: 从 GPU buffer 解析 tensor metadata
   - CUDA pointer 验证和生命周期管理

2. **GraphSafeEngine**: 
   - Step-oriented 监控引擎
   - 支持同步/异步收集模式
   - 与 GraphSlotConsumer 集成

3. **测试覆盖**: 
   - CUDA Graph capture/replay 验证
   - Metadata 正确性测试
   - Native delegate 集成测试

### 发现的关键问题
- **KV Cache 动态增长问题**: 手动 CUDA Graph 与 `resize_()` 操作不兼容
- 验证脚本显示 Graph 模式生成乱码，正常模式正确

## 技术路线评估结果

### 核心冲突
- 当前实现: 手动 CUDA Graph + KV Cache resize
- 问题: CUDA Graph 固定内存地址，resize 导致地址变化

### 路线选择
**建议采用: Custom Op + torch.compile (过渡方案)**

理由:
1. **快速模型支持**: 2周内支持 Llama/Mistral/Qwen 等主流模型
2. **避免过早优化**: 不在 GPT-2 KV Cache 问题上浪费时间
3. **保留核心价值**: C++ delegate 和 shadow parser 仍然有用
4. **为 vLLM 做准备**: 积累多模型监控经验

### Phase 1 实施计划 (1-2个月)
```python
# torch.compile 模式快速验证
model.forward = torch.compile(model.forward, mode="reduce-overhead")

@torch.ops.custom_op("monitoring::record") 
def record_tensor(tensor, buffer, slot): pass
```

### Phase 2 长期目标 (3-6个月)  
- 基于经验重新设计 vLLM 兼容架构
- 实现 VLLMMonitoringPlugin
- 支持 PagedAttention 和异构批处理

## 下一步行动
1. 停止修复当前 CUDA Graph 的 KV Cache 问题
2. 设计 custom op 接口支持 torch.compile
3. 在 Llama-2 上验证监控效果
4. 收集性能数据和使用反馈

## 重要文件位置
- 开发日志: `docs/dev_log/graph_monitor_replan.md`
- 核心实现: `monitoring/graph_engine.py`, `monitoring/graph_monitor.py`  
- C++ 代码: `monitoring/csrc/graph_native_delegate.*`
- 测试: `tests_monitoring/test_graph_*.py`
- Benchmark: `benchmark/tests/profile_decode.py`

---
*最后更新: 2026-02-22*