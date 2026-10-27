---
title: P5 CUDA 浮点 YOLOX
phase: P5
days: D21-D25
planned_hours: 15-20
actual_hours:
status: 未开始
start:
end:
tags:
  - study/yolox
  - phase/P5
  - cuda
---

# P5 · CUDA 浮点 YOLOX（D21–D25）

> [!info] 阶段定位
> 把 P4 的 kernel 组装成真正的 GPU 推理引擎，跑通 YOLOX-s float32。这一阶段的关键不是“写更多 kernel”，而是“调度、内存、布局、对拍”。

> [!note] 机器：NVIDIA
> 在 P4 的同一台机器上继续；D25 结束前 push + tag `p5-done`。如果 P7 回 Mac，CUDA 集成仍留在这台机器验证。详见 [[11-双机Git工作流]]。

## 阶段目标

- 在 GPU 上实现图执行器：按拓扑序调度、权重常驻、tensor 复用。
- 跑通 279 节点全模型，并与 NumPy/ORT 对齐。
- 优化 top 热点，建立 GPU 性能基线。
- 端到端出框，和 P2 的 CPU 结果对比。

## 硬验收标准

- [ ] GPU float32 输出 vs NumPy < 1e-3。
- [ ] 单图延迟 < 100 ms（RTX 4090 级别，按实际卡调整）。
- [ ] 显存 < 2 GB。
- [ ] 端到端 demo 能画框。
- [ ] 有一张 top 10 kernel 耗时表。

> [!warning] 先别追求打败 ORT
> 手写 im2col+GEMM 很难打赢 cuDNN/TensorRT。这一阶段的目标是“正确 + 可解释 + 比 NumPy 快很多”，不是“生产级最快”。

## 前置条件

- P4 的 kernel 和 GPU tensor/memory/registry 已通过单测。
- P1/P2 的 NumPy 输出、检测结果、参考图已保存。
- 熟悉 cudaMemcpy、stream、同步。

## 每日任务

### D21 · GPU 图执行器

- [ ] 把 P0 的 graph/拓扑序搬过来；节点按序调度。
- [ ] 权重上传一次，常驻显存；输入/输出 buffer 复用。
- [ ] 实现 tensor 生命周期：谁分配、谁复用、何时释放。
- [ ] 先跑通 stem 几层，和 NumPy 对拍。
- [ ] 加每层计时和显存统计。
- [ ] 处理 layout：NCHW 还是 NHWC，先固定一种。

**完成标志**：stem + 几个 Conv 在 GPU 上结果正确。

### D22 · 全模型跑通

- [ ] 按拓扑序跑完 279 节点；处理 Slice/Concat/Resize/Transpose/Reshape。
- [ ] 逐层 vs NumPy：先看 Conv，再看 Add/Concat，再看 Resize。
- [ ] 定位第一个偏差层；检查 pad/stride、layout、同步。
- [ ] 输出 (1,8400,85)，和 NumPy 对比 max/mean diff。
- [ ] 记录每层耗时，找 top 10。

**完成标志**：全模型输出正确；误差 < 1e-3。

### D23 · 性能优化

- [ ] 优化 top 5 热点：GEMM tile、shared memory、register blocking。
- [ ] 尝试 NHWC + `float4` 向量化 load/store。
- [ ] 减少 kernel launch 次数：融合 conv+bias+SiLU。
- [ ] 用 stream 重叠 H2D/D2H 和计算（如果端到端包含传输）。
- [ ] 用 ncu 看 memory throughput、occupancy、stall reason。
- [ ] 记录每一步优化的收益，避免“凭感觉优化”。

**完成标志**：延迟明显下降；有一张优化前后对比表。

### D24 · 前后处理 + demo

- [ ] 预处理在 CPU 或 GPU：letterbox、RGB、0–255。
- [ ] 输出拷回 CPU，做 decode/NMS（或写 GPU NMS，作为加分）。
- [ ] 跑 20 张图，和 P2 的 NumPy/ORT 检测结果对比。
- [ ] 画框、保存可视化。
- [ ] 端到端计时：预处理 + 推理 + 后处理。

**完成标志**：GPU demo 能出合理框；检测结果与 CPU 一致。

### D25 · 基准 + 缓冲

- [ ] benchmark：batch=1，warmup 10，正式 50，取 median/P90。
- [ ] 对比：NumPy、ORT CPU、ORT GPU（如果有）、你的 CUDA float。
- [ ] 记录显存峰值、每层耗时、GFLOPS。
- [ ] 写 `docs/P5-cuda-float.md`。
- [ ] 缓冲：补测试、整理代码。

**完成标志**：有一张可复现的 GPU 性能表；知道 P6 该优化哪些 kernel。

## 关键知识点与自测

| 概念 | 自测问题 |
|---|---|
| 调度 | 为什么要按拓扑序？什么时候可以并行？ |
| 内存 | 权重常驻、activation 复用、临时 buffer 怎么管？ |
| 布局 | NCHW 和 NHWC 对卷积/向量化有什么影响？ |
| 传输 | H2D/D2H 什么时候是瓶颈？怎么重叠？ |
| 同步 | 什么时候需要 `cudaDeviceSynchronize`？ |
| 计时 | 为什么 benchmark 要 warmup？median 还是 mean？ |
| 对拍 | GPU 输出不对，怎么逐层定位？ |

## 实验记录

| 日期 | 实验 | 配置 | 结果 | 结论 |
|---|---|---|---|---|
| | | | | |

### 端到端性能

| 实现 | 预处理 | 推理 | 后处理 | 总延迟 | 显存 |
|---|---|---|---|---|---|
| NumPy | | | | | |
| ORT CPU | | | | | |
| CUDA float | | | | | |

### Top kernel

| Kernel | 耗时 | 占比 | 优化动作 | 收益 |
|---|---|---|---|---|
| | | | | |

## 学习笔记

### GPU 执行器和 CPU 执行器的区别

> 内存、调度、同步、布局，各写一条。

### 最大的性能瓶颈

> 是计算、显存带宽、kernel launch 还是传输？

### 哪个优化最划算

> 记录投入产出比最高的一项。

## 问题与解决

| 问题 | 假设 | 动作 | 结果 |
|---|---|---|---|
| | | | |

## 阶段复盘

- 计划任务完成：__ / __
- 实际用时：__ 小时
- 自评（1–5）：
- 最大的卡点：
- 最大的收获：
- 对 P6 的影响：
- 需要 AI 帮我判断的 3 个问题：
  1.
  2.
  3.

## 提交给 AI 的复盘材料

- [ ] 本文件的「阶段复盘」已填
- [ ] GPU 型号/显存/CUDA 版本
- [ ] 端到端误差
- [ ] 性能表
- [ ] 3 个具体问题

## 变更记录

| 日期 | 变更 | 原因 |
|---|---|---|
| | | |

## 导航

- 上一阶段：[[05-P4-CUDA基础]]
- 下一阶段：[[07-P6-CUDA-INT8]]
- 总览：[[00-总览与动态路线]]
