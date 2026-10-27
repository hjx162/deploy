---
title: P3 NumPy 模拟量化
phase: P3
days: D11-D15
planned_hours: 15-20
actual_hours:
status: 未开始
start:
end:
tags:
  - study/yolox
  - phase/P3
  - quantization
---

# P3 · NumPy 模拟量化（D11–D15）

> [!info] 阶段定位
> 在 CPU 上把量化数学跑通，再上 GPU。这一阶段的目标不是“把模型压小”，而是搞清楚**误差从哪来、怎么控、怎么验收**。做完后你会有：一套量化方案、一组 scale、一个 int8 全模型、一张精度/速度对比表。

> [!note] 机器：Mac（M5 Pro）
> 这是 Mac 阶段的最后一阶段。D15 除了量化收尾，还要完成 Git 推送和 N 卡机器交接；详见 [[11-双机Git工作流]]。

## 阶段目标

- 理解 affine quantization：scale、zero-point、对称/非对称、per-tensor/per-channel。
- 实现校准器，收集激活范围，保存 scales。
- 实现 int8 卷积：im2col、int32 累加、bias、requant。
- 跑通 int8 全模型，处理 Add/Concat 的 scale 对齐、SiLU、MaxPool/Resize。
- 评估精度：cosine、检测一致率、mAP drop。

## 硬验收标准

- [ ] quant/dequant 单测通过（含 clamp、round、zero-point）。
- [ ] 单层 int8 conv vs float：cosine > 0.999。
- [ ] 全模型 int8 vs float：输出 cosine > 0.99。
- [ ] 检测一致率 > 95%，或 mAP drop < 2。
- [ ] 有一份量化报告：方案、scales、精度、速度、失败案例。
- [ ] N 卡机器能复现 P1/P3 基线（同一输入，误差在容差内）；不通过不开始 CUDA。

> [!warning] 先不要追求极致压缩
> 第一版用 per-channel 权重 + per-tensor 激活 + 首尾层/head 预测层 fp32，先跑通再优化。不要一上来上 INT4、QAT、混合精度搜索。

## 前置条件

- P2 的 float32 检测链路和 mAP 基线已建立。
- 有 100 张左右校准图（可以是 COCO val 子集或自己的图）。
- 理解卷积的 int32 累加范围。

## 每日任务

### D11 · 量化基础

- [ ] 写 `quantize(x, scale, zp)` / `dequantize(q, scale, zp)`。
- [ ] 对称 int8：`q = clamp(round(x/s), -127, 127)`，`zp=0`。
- [ ] 非对称 uint8/int8：`q = clamp(round(x/s)+zp, qmin, qmax)`。
- [ ] per-tensor vs per-channel：权重按输出通道算 scale。
- [ ] 做误差实验：不同 scale 下，量化/反量化的 MSE、cosine。
- [ ] 观察 outlier 对 min-max 的影响。

**完成标志**：能手算一个 1×4 向量的量化/反量化；知道 zero-point 的作用。

### D12 · 校准

- [ ] 选 100 张校准图，跑 float32 模型，收集每层激活的 min/max。
- [ ] 实现 min-max、percentile（99.9/99.99）、KL 三种校准。
- [ ] 保存每层 scale，画激活直方图。
- [ ] 对比三种方法的精度/范围；选一个默认方案。
- [ ] 注意：校准集和测试集要分开，不要用测试图校准。

**完成标志**：每层有 scale；能解释为什么选 percentile 或 min-max。

### D13 · int8 卷积

- [ ] 权重 per-channel 量化；激活 per-tensor 量化。
- [ ] im2col 后做 int32 累加：`acc = Σ (q_x - zp_x)(q_w - zp_w)`。
- [ ] 对称权重时 `zp_w=0`，公式简化。
- [ ] bias 量化：`q_b = round(b / (s_x * s_w))`，存 int32。
- [ ] requant：`q_out = clamp(round(acc * s_x * s_w / s_out) + zp_out)`。
- [ ] 先实现 float multiplier 版本，再做定点 multiplier（int32 + shift）。
- [ ] 和 float 卷积对拍：单层 cosine、max abs diff。
- [ ] 检查 int32 溢出：最坏 `127*127*K`，K 最大约 4608，约 74M，安全。

**完成标志**：单层 int8 conv 输出与 float 的 cosine > 0.999；知道每一步的 scale 从哪来。

### D14 · 全模型 int8

- [ ] 74 个 Conv+SiLU：Conv 用 int8，SiLU 用 LUT 或反量化到 fp32。
- [ ] 9 个 head 预测卷积：第一版保持 fp32，或单独 per-channel。
- [ ] Add：两个输入 rescale 到同一 scale，再相加。
- [ ] Concat：所有输入 rescale 到同一 scale，再拼接。
- [ ] MaxPool/Resize：单调算子，输入输出同 scale，不需要 requant。
- [ ] 输出反量化到 fp32，再做 decode/NMS。
- [ ] 跑 20 张图，和 float 检测结果对比。

**完成标志**：int8 全模型能出框；检测结果没有系统性偏移。

### D15 · 精度评估 + Mac 交接

- [ ] 指标：输出 cosine、检测一致率、mAP、每层误差传播。
- [ ] 如果 mAP drop > 2：先查校准范围，再查 Add/Concat scale，再查 SiLU，再查首尾层。
- [ ] 尝试改进：per-channel 权重、percentile 校准、首尾层 fp32、head 预测层 fp32、SiLU LUT。
- [ ] 记录每个改动的收益/代价。
- [ ] 写 `docs/P3-quant-report.md`：方案、scales、精度、速度、失败案例。
- [ ] Mac 收尾：清理临时文件；确认 `.venv/`、`__pycache__/`、`模型/*.pth` 没有进 Git 索引。
- [ ] 补 `requirements.txt`、README、`docs/environment.md`；记录 Mac 侧的延迟/误差。
- [ ] commit + push + tag `p3-done`（`git push --tags`）。
- [ ] 在 N 卡机器 clone/pull，重建 venv，跑通 `tools/inspect_onnx.py` 和 P1/P3 基线；记录差异。
- [ ] 如果 N 卡机器基线不一致，先修环境/路径，不要开始 CUDA。
- [ ] 缓冲：补测试或整理代码。

**完成标志**：有一组可复现的 int8 结果；代码已推送并打 tag；N 卡机器能复现 P1/P3 基线。

## 关键知识点与自测

| 概念 | 自测问题 |
|---|---|
| affine | `q = round(x/s)+zp` 中 s 和 zp 的物理意义？ |
| 对称/非对称 | 权重为什么常用对称？激活为什么常用非对称？ |
| per-channel | 为什么权重 per-channel 比 per-tensor 准？ |
| 校准 | min-max、percentile、KL 各自适合什么分布？ |
| int32 | 为什么累加用 int32？什么时候会溢出？ |
| requant | `s_x * s_w / s_out` 怎么落到整数运算？ |
| scale 对齐 | Add/Concat 为什么必须同 scale？ |
| SiLU | 为什么 SiLU 对量化敏感？LUT 怎么做？ |

## 实验记录

| 日期 | 实验 | 配置 | 结果 | 结论 |
|---|---|---|---|---|
| | | | | |

### 逐层量化误差

| 层 | float cosine | int8 cosine | max abs diff | 备注 |
|---|---|---|---|---|
| | | | | |

### 精度对比

| 方案 | cosine | 检测一致率 | mAP | 延迟 |
|---|---|---|---|---|
| float32 | 1.0 | 100% | | |
| int8 per-tensor | | | | |
| int8 per-channel | | | | |
| + 首尾 fp32 | | | | |

## 学习笔记

### 我理解的量化公式

> 用一个小例子手算：x=[-1.0, 0.0, 1.0]，s=0.01，zp=0。

### 误差最大的层

> 哪一层掉得最多？为什么？是 outlier、SiLU 还是 scale 对齐？

### 如果只能保留一个改进

> 你会选 per-channel、percentile 还是首尾 fp32？为什么？

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
- 对 P4 的影响：
- 需要 AI 帮我判断的 3 个问题：
  1.
  2.
  3.

## 提交给 AI 的复盘材料

- [ ] 本文件的「阶段复盘」已填
- [ ] 量化方案（对称/非对称、per-tensor/per-channel、校准方法）
- [ ] 逐层 cosine 表
- [ ] 检测一致率/mAP
- [ ] 3 个具体问题

## 变更记录

| 日期 | 变更 | 原因 |
|---|---|---|
| | | |

## 导航

- 上一阶段：[[03-P2-前后处理与精度]]
- 下一阶段：[[05-P4-CUDA基础]]
- 双机工作流：[[11-双机Git工作流]]
- 总览：[[00-总览与动态路线]]
