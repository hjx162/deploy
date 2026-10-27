---
title: P4 CUDA 基础
phase: P4
days: D16-D20
planned_hours: 15-20
actual_hours:
status: 未开始
start:
end:
tags:
  - study/yolox
  - phase/P4
  - cuda
---

# P4 · CUDA 基础（D16–D20）

> [!info] 阶段定位
> 从“会写 NumPy”切到“会写 GPU kernel”。这一阶段不追求完整模型，先把 thread/block、内存层次、GEMM、卷积、融合 kernel 练熟。P5 才能把 YOLOX 搬上去。

> [!note] 机器：NVIDIA（远程/本地）
> 从这一阶段开始需要 N 卡。D16 先 clone 仓库、重建 venv、复现 P1/P3 基线，再写第一个 kernel；交接细节见 [[11-双机Git工作流]]。

## 阶段目标

- 掌握 CUDA 编程模型：grid/block/thread、warp、内存层次。
- 写出正确且不太慢的 vector add、GEMM、卷积。
- 理解 coalescing、shared memory、occupancy、bank conflict。
- 建立 GPU 侧的 tensor/memory/op 基础设施。

## 硬验收标准

- [ ] vector add 正确；测出显存带宽。
- [ ] GEMM 在 cuBLAS 2× 以内（FP32）。
- [ ] 卷积（direct 或 im2col+GEMM）vs NumPy < 1e-4。
- [ ] fused conv+bias+SiLU、maxpool、resize、concat/slice kernel 都有单测。
- [ ] 有一张 ncu/nsys profiling 表。
- [ ] N 卡机器上复现 P1/P3 基线（同一输入，误差在容差内）。

## 前置条件

- 可用 NVIDIA GPU（建议 RTX 3090/4090/A10 及以上）。
- CUDA Toolkit 12.x、nvcc、nsys/ncu。
- C++ 基础：指针、数组、`extern "C"`、Makefile/CMake。
- P1 的 NumPy 算子可作为对拍参考。

## 每日任务

### D16 · 环境交接 + 第一个 kernel

**上午：在 N 卡机器上复现基线**

- [ ] `git clone` / `git pull`，`cd deploy`；确认 `模型/yolox_s.onnx` 存在。
- [ ] 重建 venv，`pip install -r requirements.txt`；不要复制 Mac 的 `.venv`（架构不兼容）。
- [ ] `nvidia-smi`、`nvcc --version`；记录 GPU 型号、驱动、CUDA、SM 数、显存。
- [ ] 跑 `tools/inspect_onnx.py`、`tests/test_ops.py`、P1/P3 基线；对比 Mac 的误差/延迟。
- [ ] 基线不通过：先修路径/依赖/模型，不要写 CUDA。

**下午：第一个 kernel**

- [ ] 写 vector add：1D grid、block 256；处理非整除边界。
- [ ] 加错误检查：`cudaGetLastError`、`cudaDeviceSynchronize`。
- [ ] 测显存带宽：读+写，和理论带宽对比。
- [ ] 学 nsys/ncu 基本用法：kernel 时间、occupancy、memory throughput。
- [ ] 记录：GPU 型号、SM 数、显存、CUDA 版本。

**完成标志**：P1/P3 基线复现通过；vector add 正确；有一张带宽/耗时表。

### D17 · GEMM

- [ ] naive GEMM：每个 thread 算一个输出。
- [ ] shared memory tiling：block tile 16×16/32×32。
- [ ] register blocking：每个 thread 算多个输出。
- [ ] 处理和 cuBLAS 的差距；记录 GFLOPS。
- [ ] 观察 bank conflict、occupancy、shared memory 用量。
- [ ] 写 `bench_gemm.py` 或 C++ benchmark。

**完成标志**：GEMM 正确；性能在 cuBLAS 2× 以内；能解释瓶颈。

### D18 · 卷积

- [ ] direct conv：每个 thread 算一个输出像素/通道。
- [ ] im2col + GEMM：先写 im2col kernel，再复用 GEMM。
- [ ] 处理 pad/stride/dilation/group（YOLOX 只需要 group=1）。
- [ ] 和 NumPy 对拍：小尺寸 + 真实层（3×3、1×1）。
- [ ] 测 im2col 的额外显存和耗时。
- [ ] 记录 direct vs im2col 的适用场景。

**完成标志**：卷积 vs NumPy < 1e-4；知道哪个方案适合 YOLOX。

### D19 · 融合与轻量算子

- [ ] fused conv+bias+SiLU：减少一次 global memory 往返。
- [ ] maxpool：5/9/13、pad、stride 1。
- [ ] resize：nearest、2×、asymmetric。
- [ ] concat/slice：按 axis 处理；先做 channel 轴和最后轴。
- [ ] 每个 kernel 都有小尺寸单测。
- [ ] 对比融合前后的耗时。

**完成标志**：P5 需要的算子都有 GPU 版本。

### D20 · GPU 基础设施 + 缓冲

- [ ] `GpuTensor`：data ptr、shape、dtype、device、owner。
- [ ] 内存管理：`cudaMalloc`/`cudaFree` 封装；避免频繁分配。
- [ ] op registry：按 op_type 分派 kernel。
- [ ] 错误处理：CUDA_CHECK 宏；错误信息带文件/行号。
- [ ] profiling：每个 kernel 的耗时表。
- [ ] 缓冲：补测试或整理代码。

**完成标志**：P5 可以直接复用这套 tensor/memory/registry。

## 关键知识点与自测

| 概念 | 自测问题 |
|---|---|
| 编程模型 | grid/block/thread 的关系？warp 是多少线程？ |
| 内存层次 | global/shared/register 的延迟和容量？ |
| coalescing | 为什么相邻 thread 要读相邻地址？ |
| shared memory | 为什么 tiling 能加速 GEMM？bank conflict 怎么产生？ |
| occupancy | 什么限制了 occupancy？寄存器、shared memory 还是 block 数？ |
| 卷积 | direct 和 im2col+GEMM 的取舍？ |
| 融合 | 为什么 conv+bias+SiLU 融合能省时间？ |
| 调试 | kernel 输出不对时，怎么缩小到最小复现？ |

## 实验记录

| 日期 | 实验 | 配置 | 结果 | 结论 |
|---|---|---|---|---|
| | | | | |

### Kernel 性能表

| Kernel | 形状/配置 | 耗时 | GFLOPS/带宽 | 备注 |
|---|---|---|---|---|
| | | | | |

## 学习笔记

### 我理解的 CUDA 执行模型

> 画一个 grid → block → warp → thread 的图。

### GEMM 优化的每一步收益

> naive → shared → register，分别快了多少？为什么？

### 最容易错的边界

> 非整除、pad、stride、越界读。

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
- 对 P5 的影响：
- 需要 AI 帮我判断的 3 个问题：
  1.
  2.
  3.

## 提交给 AI 的复盘材料

- [ ] 本文件的「阶段复盘」已填
- [ ] GPU 型号/CUDA 版本
- [ ] GEMM/卷积性能表
- [ ] 关键 kernel 代码路径
- [ ] 3 个具体问题

## 变更记录

| 日期 | 变更 | 原因 |
|---|---|---|
| | | |

## 导航

- 上一阶段：[[04-P3-NumPy模拟量化]]
- 下一阶段：[[06-P5-CUDA浮点YOLOX]]
- 双机工作流：[[11-双机Git工作流]]
- 总览：[[00-总览与动态路线]]
