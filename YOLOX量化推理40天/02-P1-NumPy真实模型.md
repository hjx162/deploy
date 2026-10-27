---
title: P1 NumPy 真实模型
phase: P1
days: D3-D6
planned_hours: 12-16
actual_hours:
status: 未开始
start:
end:
tags:
  - study/yolox
  - phase/P1
---

# P1 · NumPy 真实模型（D3–D6）

> [!info] 阶段定位
> 这是整个项目的地基：用纯 NumPy 把 279 节点、26.69 GFLOPs 的 YOLOX-s 跑通，并对齐 ORT。完成后你就有了一个可解释、可调试的参考实现，后面量化和 CUDA 都以它为准。

> [!note] 机器：Mac（M5 Pro）
> 本阶段全部在 Mac 上完成；不会用到 CUDA。双机同步见 [[11-双机Git工作流]]。

> [!note] 手搓区
> `核心框架/`、`算子演示/`、`验证工具/` 里的代码是你之前的草稿，不是第三方库。P1 的通用算子、im2col、执行器要你自己重写/扩展；现有 `numpy_ops.py` 只读参考，不要直接调用或先跑它。

## 阶段目标

- 实现 10 类算子的通用版本，而不是只够 stem 用的特化版。
- 用 im2col + GEMM 把卷积从“不可用”变成“可跑”。
- 加载真实 ONNX 权重，跑通全图。
- 与 ORT 逐层/端到端对齐，能定位第一个偏差层。

## 硬验收标准

- [ ] 10 类算子单测通过（含 bias/group/dilation/非对称 pad/任意 axis）。
- [ ] 全图 279 节点执行，无 NaN/Inf。
- [ ] 输出 vs ORT max abs diff < 1e-3，shape (1,8400,85)。
- [ ] NumPy 单图 1–3 s（M5 Pro），内存 < 4 GB。
- [ ] 有一张逐层误差表，能指出第一个偏差层。

> [!warning] 不要用朴素 conv 跑 640×640
> 26.69 GFLOPs 的七重 Python 循环是几十分钟到几小时级别。朴素版只保留给小尺寸单测，真实执行走 im2col + `np.matmul`。

## 前置条件

- P0 的 loader、拓扑排序、参考输入/输出都已通过。
- 理解 NCHW、卷积输出形状公式、广播。
- 知道 `np.matmul` 会走 BLAS。

## 每日任务

### D3 · 通用算子 + 单测

- [ ] 在新目录 `src/ops/` 下自己实现；旧 `numpy_ops.py` 只读参考，不直接修改或调用。
- [ ] Conv：NCHW、bias、group、dilation、pads `[top,left,bottom,right]`、stride；自己先写朴素版。
- [ ] Slice：starts/ends/axes/steps；处理 `INT64_MAX`；支持负 axis。
- [ ] Concat：任意 axis；重点测 `axis=1` 和 `axis=-1`。
- [ ] Add / Mul：支持广播。
- [ ] Sigmoid：向量化，注意大负数溢出。
- [ ] MaxPool：kernel 5/9/13、pads、stride 1、`ceil_mode=0`。
- [ ] Resize：nearest、`coordinate_transformation_mode=asymmetric`、scales `[1,1,2,2]`。
- [ ] Reshape：`[1,85,-1]`；Transpose：`perm=[0,2,1]`。
- [ ] 每个算子至少 2 个用例：一个手算小例子，一个和 ORT 对拍。

**完成标志**：10 类算子都有单测；失败用例能缩到最小形状。

### D4 · im2col + 向量化

- [ ] 写 `im2col`：输入 (N,C,H,W)，输出 (N*H_out*W_out, C*kH*kW)，处理 pad/stride/dilation。
- [ ] 把你 D3 自己写的卷积改成 `col @ W.reshape(C_out, -1).T + bias`。
- [ ] 用 `np.ascontiguousarray` 避免 matmul 前的隐式拷贝。
- [ ] 向量化 Sigmoid/Add/Mul/Concat/Slice，去掉逐元素 Python 循环。
- [ ] 在小层（如 1×3×32×32）上对比你自己新写的 naive vs im2col vs ORT。
- [ ] 在真实层（如 stem 3×3、80×80 的 3×3）上测耗时和内存。

**完成标志**：单层误差 < 1e-5；im2col 版本比你自己写的 naive 快至少一个数量级；知道 im2col 的峰值内存。

### D5 · 真实权重 + 全图对拍

- [ ] 从 ONNX initializer 加载 float32 权重；int64 常量进单独的常量表。
- [ ] 按拓扑序执行 279 节点；统一值表按 tensor name 存取。
- [ ] 逐层对比 ORT：max abs diff、relative diff；先看 Conv/Add/Concat。
- [ ] 定位第一个偏差 > 1e-3 的节点，缩到单层复现。
- [ ] 检查 dtype：全程 float32，不要混入 float64。
- [ ] 检查 pad/stride/dilation/group 的每一个属性。

**完成标志**：第一个偏差层被解释并修掉；全图无 NaN/Inf。

### D6 · 端到端 + 性能基线

- [ ] 640×640 输入跑完整模型；输出 (1,8400,85)。
- [ ] 和 ORT 比 max abs diff / mean abs diff；目标 < 1e-3。
- [ ] 记录单图延迟（warmup + 5 次平均）、峰值内存。
- [ ] 跑 3 张不同输入，确认不是偶然对齐。
- [ ] 提交第一个可复现版本（git commit）。
- [ ] 写一页 `docs/P1-float-baseline.md`：结构、耗时、误差、已知问题。

**完成标志**：别人按 README 能复现你的输出和误差。

## 关键知识点与自测

| 概念 | 自测问题 |
|---|---|
| 卷积形状 | 输出 H/W 公式怎么写？pad 不对称时呢？ |
| im2col | 为什么展开后可以用矩阵乘？峰值内存怎么估？ |
| GEMM | 为什么 `np.matmul` 比 Python 循环快？ |
| 布局 | NCHW 下 `im2col` 的列顺序是什么？ |
| 数值 | 为什么全程用 float32？累加顺序会带来多大误差？ |
| 对拍 | 如何在不改模型的情况下 dump 中间层？ |

## 实验记录

| 日期 | 命令/实验 | 结果 | 结论 |
|---|---|---|---|
| | | | |

### 逐层误差表

| 节点 | 形状 | max abs diff | 备注 |
|---|---|---|---|
| | | | |

## 学习笔记

### 我理解的 im2col

> 画一个小例子：3×3 输入、2×2 卷积核，展开后矩阵长什么样。

### 第一个偏差层是怎么找到的

> 记录排查过程，比结论更有价值。

### 性能瓶颈在哪

> 哪一层最慢？为什么？是计算、内存还是拷贝？

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
- 对 P2 的影响：
- 需要 AI 帮我判断的 3 个问题：
  1.
  2.
  3.

## 提交给 AI 的复盘材料

- [ ] 本文件的「阶段复盘」已填
- [ ] 逐层误差表
- [ ] 端到端误差和延迟
- [ ] `src/` 关键文件路径
- [ ] 3 个具体问题

## 变更记录

| 日期 | 变更 | 原因 |
|---|---|---|
| | | |

## 导航

- 上一阶段：[[01-P0-真值与工程收口]]
- 下一阶段：[[03-P2-前后处理与精度]]
- 总览：[[00-总览与动态路线]]
