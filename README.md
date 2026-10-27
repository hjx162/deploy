# YOLOX-s 量化推理学习项目

以 `模型/yolox_s.onnx` 为唯一模型源，用纯 NumPy 手写 YOLOX-s 推理框架，再扩展到 NumPy int8 模拟量化和 CUDA float/int8。

当前进度：学习起点。`核心框架/` 里是 stem 玩具图；真实 ONNX 全模型从 P0/P1 开始。

## 目录

| 目录 | 内容 |
|---|---|
| `YOLOX量化推理40天/` | Obsidian 学习笔记、40 天路线、阶段复盘 |
| `核心框架/` | 图/算子/执行器雏形（stem 玩具图） |
| `DAG学习/` | YOLOX DAG、279 节点算子图、拓扑排序 |
| `验证工具/` | 算子单测、与 ORT 对拍（待改造） |
| `模型/` | `yolox_s.onnx`；`yolox_s.pth` 不进 Git |
| `算子演示/` | CSPLayer 抽象视角 vs 算子视角 |

## 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`.venv/` 是本机环境，不进 Git。换到 N 卡机器后必须重建，不要复制。

## 快速检查

```bash
python 核心框架/numpy_framework.py
PYTHONPATH=核心框架 python 验证工具/test_ops.py
```

这两条只验证 stem 玩具图和基础算子，不代表真实 YOLOX 已跑通。

## 模型事实

- 输入 `inputs` (1,3,640,640)，RGB、0–255、letterbox。
- 输出 `outputs` (1,8400,85)，80×80 → 40×40 → 20×20，原始网格值。
- 279 节点、10 类算子、176 initializers、8.96M 参数、26.69 GFLOPs。
- ORT CPU（M5 Pro）约 54 ms。

## 学习路线

- P0–P3（D1–D15，Mac）：ONNX 解析、NumPy float、前后处理、NumPy int8。
- P4–P6（D16–D30，NVIDIA）：CUDA 基础、GPU float、CUDA int8。
- P7–P8（D31–D40）：框架化、benchmark、demo、总结。

入口：`YOLOX量化推理40天/00-总览与动态路线.md`。

Git 双机同步：`YOLOX量化推理40天/11-双机Git工作流.md`。

## 备注

- 推理不需要 `.pth`；ONNX 已包含权重。
- 不要提交 `.venv/`、`__pycache__/`、`.DS_Store`、`模型/*.pth`、`build/`。
