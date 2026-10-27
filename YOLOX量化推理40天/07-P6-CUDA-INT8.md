---
title: P6 CUDA INT8
phase: P6
days: D26-D30
planned_hours: 15-20
actual_hours:
status: 未开始
start:
end:
tags:
  - study/yolox
  - phase/P6
  - cuda
  - quantization
---

# P6 · CUDA INT8（D26–D30）

> [!info] 阶段定位
> 把 P3 的 int8 数学搬到 GPU，用 dp4a 做 int8 卷积，跑通全模型。这是整个 40 天里最难、也最能体现“量化推理框架”含金量的一段。P6 是加分项：做不完不影响 P1–P5 的价值。

> [!note] 机器：NVIDIA
> 继续用 P4/P5 的机器；D30 结束前 push + tag `p6-done`。如果机器时间有限，优先保证 P5 float 完整，再做“单层 int8 kernel + 全模型尽量”。详见 [[11-双机Git工作流]]。

## 阶段目标

- 实现 int8 GEMM/卷积：dp4a、int32 累加、bias、requant。
- 把 P3 的量化方案完整复现到 GPU。
- 处理 Add/Concat 的 scale 对齐、MaxPool/Resize、SiLU。
- 对比 int8 vs float32 的精度和速度。

## 硬验收标准

- [ ] int8 GEMM/卷积 vs NumPy int8：bit 级一致，或 max abs diff < 1e-5。
- [ ] 全模型 int8 vs NumPy int8：max abs diff < 1e-5。
- [ ] 相对 CUDA float32 加速 ≥ 1.5×。
- [ ] mAP drop 可接受（建议 < 2），或检测一致率 > 95%。
- [ ] 有一份 int8 报告：kernel、精度、速度、失败案例。

> [!warning] 先复现，再优化
> 第一版用 `__dp4a` + im2col/implicit GEMM，先保证和 NumPy int8 一致。不要一上来上 Tensor Core INT8、INT4、混合精度搜索。

## 前置条件

- P5 的 CUDA float32 全模型已对齐。
- P3 的量化方案、scales、逐层参考已保存。
- 理解 int8 打包、int32 累加、requant。
- GPU 支持 dp4a（sm_61+；RTX 30/40 都支持）。

## 每日任务

### D26 · dp4a GEMM

- [ ] 写 `__dp4a` 版 GEMM：A/B 为 int8，累加 int32。
- [ ] 处理 K 不是 4 的倍数：补零或尾处理。
- [ ] int8 布局：NCHW 还是 NHWC？dp4a 更适合 K 连续。
- [ ] 和 NumPy int8 GEMM 对拍。
- [ ] 测 GFLOPS/TOPS，和 float GEMM 对比。
- [ ] 记录 per-channel scale 的存放方式。

**完成标志**：int8 GEMM 正确；知道 dp4a 的吞吐上限。

### D27 · int8 卷积

- [ ] int8 im2col（或 implicit GEMM）+ dp4a。
- [ ] bias 用 int32；requant 用 `s_x*s_w/s_out`。
- [ ] 先 float multiplier，再定点 multiplier（int32 + shift）。
- [ ] fused conv+bias+requant+SiLU（SiLU 可用 LUT 或 fp32）。
- [ ] 和 P3 的 NumPy int8 单层对拍。
- [ ] 检查溢出和 clamp。

**完成标志**：单层 int8 conv 与 NumPy int8 一致；知道每个 scale 的来源。

### D28 · 全模型 int8

- [ ] 74 个 Conv+SiLU 走 int8；9 个 head 预测卷积先保持 fp32 或单独 per-channel。
- [ ] Add/Concat：rescale 到同一 scale；写 requant kernel。
- [ ] MaxPool/Resize：同 scale，直接复用 int8 buffer。
- [ ] 输出反量化到 fp32，再做 decode/NMS。
- [ ] 跑 20 张图，和 NumPy int8 对比输出/检测。
- [ ] 记录每层耗时和显存。

**完成标志**：int8 全模型能出框；与 NumPy int8 结果一致。

### D29 · 精度 + 速度调优

- [ ] 指标：vs NumPy int8、vs CUDA float、mAP/检测一致率。
- [ ] 调优：int8x4 向量化、shared memory、double buffer、减少 requant。
- [ ] 用 ncu 看 dp4a 利用率、memory throughput、stall。
- [ ] 对比不同布局/分块；记录收益。
- [ ] 如果精度掉太多：查 scale 对齐、SiLU、首尾层、校准。
- [ ] 记录“优化前/后”的表。

**完成标志**：有可解释的速度/精度 trade-off；知道下一步该优化什么。

### D30 · 混合精度 + 报告

- [ ] 混合精度：首尾层/head 预测层 fp32，其余 int8；对比精度/速度。
- [ ] 写 `docs/P6-cuda-int8.md`：方案、kernel、精度、速度、失败案例。
- [ ] 整理 benchmark：float vs int8，NumPy vs CUDA。
- [ ] 缓冲：补测试、清理代码。
- [ ] 如果时间不够，明确记录“未完成项”和原因，不要假装完成。

**完成标志**：有一份可复现的 int8 报告；知道 P7 该框架化哪些部分。

## 关键知识点与自测

| 概念 | 自测问题 |
|---|---|
| dp4a | 一条 dp4a 做多少次乘加？输入/输出是什么类型？ |
| 累加 | 为什么用 int32？最坏情况会溢出吗？ |
| 打包 | int8 在内存里怎么排？为什么要 4 个一组？ |
| requant | 定点 multiplier + shift 怎么实现？ |
| per-channel | scale 存在哪？kernel 怎么索引？ |
| scale 对齐 | Add/Concat 在 GPU 上怎么 rescale？ |
| SiLU | LUT 还是 fp32？LUT 的输入 scale 怎么定？ |
| 混合精度 | 哪些层必须 fp32？为什么？ |

## 实验记录

| 日期 | 实验 | 配置 | 结果 | 结论 |
|---|---|---|---|---|
| | | | | |

### 精度对比

| 方案 | vs NumPy int8 | vs float | mAP | 延迟 | 显存 |
|---|---|---|---|---|---|
| CUDA float | — | 1e-3 | | | |
| CUDA int8 | | | | | |
| 混合精度 | | | | | |

### Kernel 性能

| Kernel | 耗时 | TOPS/带宽 | 优化 | 收益 |
|---|---|---|---|---|
| | | | | |

## 学习笔记

### dp4a 到底做了什么

> 用一个小例子写清楚：4 个 int8 乘加，累加到 int32。

### 量化误差在 GPU 上多出来的部分

> 和 NumPy int8 比，GPU 版本多了哪些误差来源？（累加顺序、requant、LUT）

### 最值得的优化

> 投入产出比最高的一项是什么？

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
- 对 P7 的影响：
- 需要 AI 帮我判断的 3 个问题：
  1.
  2.
  3.

## 提交给 AI 的复盘材料

- [ ] 本文件的「阶段复盘」已填
- [ ] 量化方案 + scales
- [ ] vs NumPy int8 / vs float 的精度
- [ ] 性能表
- [ ] 未完成项及原因
- [ ] 3 个具体问题

## 变更记录

| 日期 | 变更 | 原因 |
|---|---|---|
| | | |

## 导航

- 上一阶段：[[06-P5-CUDA浮点YOLOX]]
- 下一阶段：[[08-P7-框架化]]
- 总览：[[00-总览与动态路线]]
