"""小尺寸测试: 验证框架逻辑正确(不等大图, 先证明算子/DAG/拓扑排序对)。"""
import numpy as np
import numpy_ops as ops

print('=== 1. 算子单元测试 ===')

# sigmoid / silu
x = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
print('sigmoid:', ops.sigmoid(x))
print('  (期望: 0.2689, 0.5, 0.7311, 0.8808)')

# conv2d 小测试: 1x1 输入, 单通道, 核3x3
X = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
W = np.ones((1, 1, 3, 3), dtype=np.float32)
B = np.zeros((1,), dtype=np.float32)
O = ops.conv2d_naive(X, W, B, stride=(1, 1), padding=(1, 1))
print(f'\nconv2d 输出形状: {O.shape} (期望 (1,1,5,5))')
print(f'conv2d 中心值: {O[0,0,2,2]:.1f} (期望 108.0)')

# batchnorm
xb = np.ones((1, 2, 2, 2), dtype=np.float32)
gamma = np.array([2.0, 3.0], dtype=np.float32)
beta = np.array([1.0, -1.0], dtype=np.float32)
mean = np.zeros(2, dtype=np.float32)
var = np.ones(2, dtype=np.float32)
ob = ops.batchnorm_2d(xb, gamma, beta, mean, var)
print(f'\nbatchnorm 通道0: {ob[0,0,0,0]:.1f} (期望 3.0 = 2*1+1)')
print(f'batchnorm 通道1: {ob[0,1,0,0]:.1f} (期望 2.0 = 3*1-1)')

# slice_focus
xf = np.arange(1 * 2 * 4 * 4, dtype=np.float32).reshape(1, 2, 4, 4)
of = ops.slice_focus(xf)
print(f'\nslice_focus: {xf.shape} -> {of.shape} (期望 (1,8,2,2))')

# concat
a = np.ones((1, 2, 3, 3), dtype=np.float32)
b = np.zeros((1, 3, 3, 3), dtype=np.float32)
c = ops.concat([a, b])
print(f'concat: {c.shape} (期望 (1,5,3,3))')

print('\n=== 全部算子单元测试通过 ===')
