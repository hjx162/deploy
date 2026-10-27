---
title: 双机 Git 工作流（Mac ↔ NVIDIA）
type: workflow
status: 未开始
created: 2026-10-27
updated: 2026-10-27
tags:
  - study/yolox
  - workflow/git
  - workflow/remote
---

# 双机 Git 工作流（Mac ↔ NVIDIA）

> [!info] 目的
> Mac 负责 P0–P3（NumPy float + int8），N 卡机器负责 P4–P6（CUDA）。两台机器只通过 GitHub 同步代码和文档，不同步 `.venv`、编译产物和模型缓存。

## 机器分工与时间线

| 阶段 | 天数 | 机器 | 主要产出 |
|---|---|---|---|
| P0–P3 | D1–D15 | Mac（M5 Pro） | ONNX 解析、NumPy float 全模型、前后处理、NumPy int8 |
| P4–P6 | D16–D30 | NVIDIA 机器 | CUDA 基础、GPU float 全模型、CUDA int8 |
| P7–P8 | D31–D40 | 任一（建议 P7 的 CUDA 集成留在 N 卡机器） | 图优化、内存规划、API、benchmark、总结 |

**切换点**：D15 结束前完成 Mac 侧收尾并推送；D16 在 N 卡机器上先复现基线，再开始 CUDA。

## 仓库现状（2026-10-27）

- 远程仓库：`https://github.com/hjx162/deploy.git`
- `git ls-remote` 没有返回分支，说明仓库目前是空的。
- 本地项目根目录：`/Users/hjx/work/deploy/study`。
- 学习笔记已迁移到 `study/YOLOX量化推理40天/`；仓库根目录就是 `study/`，代码、笔记、模型在同一个仓库里。
- Obsidian vault 根目录也变成了 `study/`；在 Obsidian 里重新 **Open folder as vault**，选择 `/Users/hjx/work/deploy/study`。
- 本地仓库已初始化：分支 `main`，首次提交已完成，`origin` 已指向上面的远程仓库；尚未 push。

## 一次性初始化（Mac）

```bash
cd /Users/hjx/work/deploy/study
git init
git branch -M main
git remote add origin https://github.com/hjx162/deploy.git
```

在 `study/` 建 `.gitignore`：

```gitignore
.DS_Store

# Python
.venv/
**/__pycache__/
**/*.pyc

# 模型：ONNX 要同步；pth 不进 Git（推理不需要）
模型/*.pth

# 编译产物 / 临时文件
build/
**/*.o
**/*.so
**/*.log
**/*.tmp

# Obsidian 的机器相关状态
.obsidian/workspace.json
.obsidian/workspace-mobile.json
.obsidian/cache
.obsidian/plugins/*/data.json

# 子目录里的旧 vault 配置（当前 vault 根目录的 .obsidian 才需要同步）
YOLOX量化推理40天/.obsidian/
```

如果两台机器的换行/编码不一致，再加 `.gitattributes`：

```gitattributes
* text=auto
*.onnx binary
*.pth binary
*.npy binary
```

在 `study/` 下建 `requirements.txt`（以 Mac 上验证过的版本为基准）：

```text
numpy==2.0.2
onnx==1.19.1
onnxruntime==1.19.2
opencv-python>=4.8
pycocotools>=2.0.7
```

首次提交：

```bash
cd /Users/hjx/work/deploy/study
git add .gitignore .gitattributes .
git status --short
git commit -m "chore: initial YOLOX study project"
git push -u origin main
```

> [!warning] 不要提交
> `.venv/`、`__pycache__/`、`.DS_Store`、`模型/*.pth`、`build/`、`*.o`、`*.so`、任何 token/密钥。
> `.venv` 是 macOS arm64 的，复制到 Linux/N 卡机器上不能用，必须在目标机器重建。

## 模型与数据文件策略

| 文件 | 大小 | 建议 |
|---|---|---|
| `模型/yolox_s.onnx` | 34 MB | 直接提交；GitHub 单文件限制 100 MB |
| `模型/yolox_s.pth` | 69 MB | 不提交；推理不需要。要用就 Git LFS 或手动拷贝 |
| `tests/data/ref_input.npy` | 约 4.9 MB | 建议提交，跨机器复现用 |
| `tests/data/ref_output.npy` | 约 2.9 MB | 建议提交，跨机器复现用 |
| `DAG学习/*.json` | < 30 KB | 直接提交 |

如果以后二进制文件变多，再启用 Git LFS：

```bash
git lfs install
git lfs track "*.onnx" "*.pth" "*.npy"
git add .gitattributes
```

> [!note] 公开仓库
> 如果不想公开模型权重，把 GitHub 仓库设为 private；私有仓库在 N 卡机器上需要 SSH key 或 `gh auth login`，不要把 PAT 写进 remote URL。

## N 卡机器首次拉取

```bash
git clone https://github.com/hjx162/deploy.git
cd deploy
python3 --version          # 建议 3.10/3.11
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

先确认硬件和 CUDA：

```bash
nvidia-smi
nvcc --version
```

再复现 Mac 的基线（这一步不通过，不要开始写 CUDA）：

```bash
python tools/inspect_onnx.py
python tests/test_ops.py
python run_yolox.py --input tests/data/ref_input.npy
```

记录到 `docs/environment.md`：

```markdown
## Mac
- 机型：MacBook Pro / Apple M5 Pro
- macOS：
- Python：3.9.6
- numpy / onnx / onnxruntime：
- ORT 延迟：

## NVIDIA
- GPU：
- 驱动：
- CUDA：
- Python：
- numpy / onnx / onnxruntime：
- ORT 延迟：
- 复现 P1/P3 的误差：
```

## 每次切换机器的清单

> [!important] 离开一台机器前
> - [ ] `git status` 干净（没有未提交改动）
> - [ ] `git add` + `git commit`
> - [ ] `git push`
> - [ ] 阶段结束：`git tag p?-done && git push --tags`
> - [ ] 记下当前进度和下一步（写到阶段文档）

> [!important] 到另一台机器后
> - [ ] `git pull --ff-only`
> - [ ] `git fetch --tags`
> - [ ] 重建/激活 venv，`pip install -r requirements.txt`
> - [ ] 确认模型文件存在，哈希一致（`shasum -a 256` / `sha256sum`）
> - [ ] 跑 `tools/inspect_onnx.py`，节点数/算子数一致
> - [ ] 跑一个最小单测
> - [ ] 跑 P1/P3 基线，误差在容差内
> - [ ] 再开始新阶段

## 分支与 tag 策略

 solo 学习不需要复杂分支。建议：

- 平时在 `main` 上提交。
- 每个阶段结束打 tag：`p0-done`、`p1-done`、…、`p6-done`、`v1.0`。
- 实验性改动开短分支：`exp/winograd`、`exp/int8-lut`；验证完合并或丢弃。
- 不要对 `main` 做 `push --force`；不要用 `reset --hard` 解决同步问题。

```bash
# 阶段结束
git tag p3-done
git push --tags

# 另一台机器查看
git fetch --tags
git tag -l
```

## 环境固定

- Python：Mac 3.9.6；N 卡机器建议 3.10/3.11。`requirements.txt` 里的版本要两边都能装。
- CUDA：记录 Toolkit 版本（如 12.4）、驱动版本、GPU 架构（如 `sm_89`）。
- 编译：CMake 里用 `CMAKE_CUDA_ARCHITECTURES=native`（CMake ≥ 3.24），或显式 `sm_89`/`sm_86`。
- 如果 N 卡机器是 Windows：优先 WSL2 + Ubuntu；原生 Windows 需要 Visual Studio + CUDA，路径和编译问题更多。
- 不要依赖 Mac 特有的库或路径；代码里用 `pathlib`，不要写 `/workspace/env_init`。

## 常见坑

| 坑 | 现象 | 处理 |
|---|---|---|
| 把 `.venv` 提交了 | 仓库巨大，Linux 上不能用 | 删除并加入 `.gitignore`，目标机器重建 |
| 中文路径/空格 | 远程 shell/编译脚本出错 | 代码目录尽量用 ASCII；至少避免空格 |
| 硬编码绝对路径 | 换机器就 FileNotFound | 全部改成 `Path(__file__).parent` / 项目相对路径 |
| 模型没同步 | N 卡机器上 `FileNotFound` | 确认 ONNX 已提交；或用 `scp`/LFS |
| pth 超过 100 MB | push 被 GitHub 拒绝 | 提交 ONNX 即可；pth 用 LFS 或手动拷贝 |
| 两边同时改同一文件 | merge 冲突 | 切换前 push；到另一台先 pull；不要两边并行编辑 |
| CUDA 架构不匹配 | 编译报错/运行 no kernel image | 用 `native` 或查 GPU 的 `sm_XX` |
| 未提交就换机器 | 代码丢失/版本不一致 | 离开前 `git status` 必须干净 |

## 如果 N 卡机器晚到

不要空等。Mac 上可以先做 P7 中不依赖 CUDA 的部分：

- 图优化：常量折叠、死节点消除、融合规划（先在 NumPy 后端验证）。
- 内存规划：生命周期分析、buffer 复用（NumPy 后端）。
- Python API：`Model(path, backend="numpy")`、`run()`。
- Benchmark harness：NumPy / ORT 的延迟和精度表。
- README、环境文档、阶段复盘。

等 N 卡机器可用后，按 P4 → P5 → P6 推进，再把 CUDA 后端接进已有 API。这样 GPU 时间花在 kernel 和模型上，而不是补工程管线。

## 与阶段文档的对应

- P3 D15：Mac 收尾 + push + tag `p3-done`。
- P4 D16：N 卡机器 clone + 重建环境 + 复现 P1/P3 基线 + 第一个 kernel。
- P5 D25：push + tag `p5-done`。
- P6 D30：push + tag `p6-done`。
- P7 D35：push + tag `p7-done`。
- P8 D40：push + tag `v1.0`。

## 导航

- 总览：[[00-总览与动态路线]]
- [[04-P3-NumPy模拟量化]]
- [[05-P4-CUDA基础]]
- [[06-P5-CUDA浮点YOLOX]]
- [[07-P6-CUDA-INT8]]
- [[08-P7-框架化]]
- [[09-P8-收官与拓展]]
