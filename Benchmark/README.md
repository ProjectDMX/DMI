Benchmark Scripts
=================

`Benchmark/` 目录收录了用于性能与内存测试的脚本，当前核心文件为
`gpt2_hidden_state_benchmark.py`，可比较 GPT-2 在不同隐藏状态 / 注意力抓取策略下的
推理表现与资源占用。

功能概览
--------
- **baseline**：纯推理路径，不抓取中间张量。
- **pytorch_hook**：通过自定义 hook 捕获注意力或隐藏状态，可与 Hugging Face 的
  `output_attentions` / `output_hidden_states` 输出对齐验证。
- **pytorch_hook_cpu**：在上述基础上，每次捕获后立即迁移到 CPU，评估设备间拷贝带来的
  额外开销（仍受 `--collect_attentions` / `--collect_hidden_states` 控制）。
- **pytorch_hook_dense**：在 block 内附加更多细粒度子模块 hook，例如残差流
  (`resid_pre`/`resid_post`)、层归一化输出、MLP 输入与输出、拆分后的 `q/k/v` 以及注意力
  模式等，以逼近 TransformerLens 的激活粒度。
- **hf_reference**：使用 Hugging Face 原生前向输出作为真值，帮助校验自定义 hook 的结果
  是否一致。
- 支持 `--profile` 切换到 PyTorch Profiler，生成 CPU/GPU 时间与内存分析，并将 trace 写入
  `./tb_traces/<label>`（可用 TensorBoard 查看）。
- 支持 `--save_dir` 将捕获的张量落盘，命名格式为 `label_feature.pt`。

常用参数
--------
- `--collect_attentions`：抓取并验证各层注意力概率。
- `--collect_hidden_states`：抓取并验证各层隐藏状态。
- `--profile`：启用 PyTorch Profiler，替代简单计时统计。
- `--save_dir PATH`：将捕获张量保存到指定目录。
- 其余参数（模型名称、批大小、序列长度、重复次数等）见脚本内 argparse 定义。

运行示例
--------
```
# 仅比较性能（默认在 GPU 上运行）
python Benchmark/gpt2_hidden_state_benchmark.py --batch_size 4 --sequence_length 128

# 捕获注意力与隐藏状态，并保存结果到 ./artifacts
python Benchmark/gpt2_hidden_state_benchmark.py \
    --collect_attentions \
    --collect_hidden_states \
    --save_dir ./artifacts

# 启用 profiler，输出 TensorBoard trace，聚焦注意力抓取路径
python Benchmark/gpt2_hidden_state_benchmark.py --profile --collect_attentions
```

脚本执行结束后，会输出各基线的耗时（或 profiler 表），若启用了验证，还会打印首批次
张量的形状、数据类型和所在设备。所有捕获张量在各基线运行完成后会自动迁移到 CPU 并
清理 GPU 缓存，避免跨基线累积导致的显存溢出。
