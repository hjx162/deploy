"""
终极验证: NumPy 手搓框架 vs PyTorch 相同结构, 输出对比。
用小尺寸(1,3,32,32), 验证框架逻辑与 torch 完全一致。

对比结构(stem 部分):
  input -> slice_focus -> conv(12->32,3x3,s1) -> bn -> silu -> conv(32->64,3x3,s2) -> bn -> silu
"""
import numpy as np
import sys
sys.path.insert(0, '/workspace/env_init')
import numpy_ops as ops

# ---------- PyTorch 参考实现(仅用于验证对比) ----------
import torch
import torch.nn as nn

def build_torch_model():
    """与 NumPy 框架完全相同的结构。"""
    class Focus(nn.Module):
        def forward(self, x):
            patch_top_left = x[..., ::2, ::2]
            patch_bot_left = x[..., 1::2, ::2]
            patch_top_right = x[..., ::2, 1::2]
            patch_bot_right = x[..., 1::2, 1::2]
            return torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)

    class Stem(nn.Module):
        def __init__(self):
            super().__init__()
            self.focus = Focus()
            self.conv1 = nn.Conv2d(12, 32, 3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(64)
        def forward(self, x):
            x = self.focus(x)
            x = torch.nn.functional.silu(self.bn1(self.conv1(x)))
            x = torch.nn.functional.silu(self.bn2(self.conv2(x)))
            return x
    return Stem()

# ---------- 用相同权重 ----------
torch.manual_seed(42)
np.random.seed(42)

model = build_torch_model()
model.eval()

# 把 torch 权重转成 numpy 格式, 喂给 NumPy 框架
def get_weights(model):
    w = {}
    # conv1: 12->32
    w['stem_conv'] = {
        'W': model.conv1.weight.detach().numpy(),
        'B': np.zeros(32, dtype=np.float32),  # bias=False, BN 吸收 bias
    }
    w['stem_bn'] = {
        'gamma': model.bn1.weight.detach().numpy(),
        'beta': model.bn1.bias.detach().numpy(),
        'mean': model.bn1.running_mean.numpy(),
        'var': model.bn1.running_var.numpy(),
    }
    # conv2: 32->64
    w['d2_conv'] = {
        'W': model.conv2.weight.detach().numpy(),
        'B': np.zeros(64, dtype=np.float32),
    }
    w['d2_bn'] = {
        'gamma': model.bn2.weight.detach().numpy(),
        'beta': model.bn2.bias.detach().numpy(),
        'mean': model.bn2.running_mean.numpy(),
        'var': model.bn2.running_var.numpy(),
    }
    return w

# ---------- 执行对比 ----------
x = np.random.randn(1, 3, 32, 32).astype(np.float32)

# 1. NumPy 框架
import numpy_yolox_demo as demo
weights = get_weights(model)
tensors, order = demo.run_inference(demo.GRAPH, weights, torch.tensor(x).numpy())
numpy_out = tensors['output']

# 2. PyTorch
with torch.no_grad():
    torch_out = model(torch.tensor(x))

# 3. 对比
torch_out_np = torch_out.numpy()
print('=' * 60)
print('NumPy 手搓框架 vs PyTorch 输出对比')
print('=' * 60)
print(f'NumPy 输出形状: {numpy_out.shape}')
print(f'PyTorch 输出形状: {torch_out_np.shape}')

diff = np.abs(numpy_out - torch_out_np)
print(f'\n最大绝对误差: {diff.max():.6f}')
print(f'平均绝对误差: {diff.mean():.6f}')

# 容差判断(不同实现累加顺序不同, 允许小误差)
ok = np.allclose(numpy_out, torch_out_np, rtol=1e-3, atol=1e-3)
print(f'\n是否对齐(rtol=1e-3, atol=1e-3): {"✅ 完全对齐" if ok else "❌ 有偏差"}')
if not ok:
    # 看看哪里差最多
    idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f'最大误差位置: {idx}, NumPy={numpy_out[idx]:.4f}, torch={torch_out_np[idx]:.4f}')
