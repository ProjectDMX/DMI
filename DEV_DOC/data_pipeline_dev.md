# GPU→CPU 数据通道开发文档（MVP）

## 目标（MVP）
- 在不影响主推理流的前提下，通过 GPU 端预留 staging 缓冲启用高效 DMA，将采集数据（激活、Top‑K、元数据）从 GPU 合并转运到 CPU，并通过“实时流”提供订阅消费（不落盘）。

## 组件分层（MVP 最小集）
- GPUStager：GPU 端合并碎片为连续大块，典型三缓冲（3×64–128 MB，可配）。
- D2HTransfer：多 CUDA streams 执行 `cudaMemcpyAsync` 从 GPUStager 到 CPU pinned 环形缓冲（DMA）。
- CpuPinnedRing：2–4 槽，1–5 GB 可配；支持零拷贝切片给订阅端。
- RealtimeSink：gRPC streaming / WebSocket / IPC（三选一，MVP 任选其一实现）。

## 数据流
- 采集回调仅提交设备指针/元数据 → GPUStager 合并达到大小/时间阈值 → D2H 多流拷贝 → CpuPinnedRing 入队 → RealtimeSink 推送。

## 并发与同步
- GPUStager 在计算流完成后以 CUDA events 公布批次就绪；
- D2H 与推送分别由独立线程/任务负责；跨阶段以无锁队列/环形缓冲传递批次描述符。

## 直接内存访问（DMA）说明
- 使用 `cudaMemcpyAsync(Device→Host)` 到 CPU pinned 内存触发 DMA；
- 通过合并小张量到连续 staging 区显著降低调用次数、提高有效带宽；
- 建议使用 2–4 条 copy streams 与 2–3 组 staging 区以重叠计算/传输。

## 背压
- 当 ring 水位高/推送端积压：
  - 通知 MonitoringEngine：降层 → 降采样 → 仅 Top‑K → 暂停采集。

## 接口（草案）
```python
class GPUStager: ...
class D2HTransfer: ...
class CpuPinnedRing: ...
class RealtimeSink: ...  # 统一 push(payload) 接口
```

## 参数建议（默认）
- `gpu_staging_bytes=3*64–128MB`，`cpu_ring_bytes=2GB`，`copy_streams=2–4`，
  `staging_trigger_bytes=8–32MB`，`staging_trigger_ms=2–5`，
  `backpressure_high_watermark=0.7`，`max_d2h_bw≈20GB/s`。

## DoD
- 顺序一致、无丢帧（在 ring 窗口内）；在 7B/4090 上 D2H 达到高带宽、对推理吞吐影响 < 5%。

## 实施状态（代码位置）
- 配置入口（GPU staging/阈值、ring、D2H 流）已通过 `MonitoringConfig` 打通：
  - 定义：vllm/config/__init__.py:3215
  - CLI：vllm/engine/arg_utils.py:853, vllm/engine/arg_utils.py:857, vllm/engine/arg_utils.py:863
  - 注入引擎配置：vllm/engine/arg_utils.py:1430, vllm/engine/arg_utils.py:1456
- 代码实现：GPUStager/D2H/CpuPinnedRing/RealtimeSink 暂未落地，下一步将按接口骨架逐项实现（先 no‑op，后逐步启用 DMA 与流控）。
