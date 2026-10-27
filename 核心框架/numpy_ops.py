"""
纯 NumPy 手搓算子库(档位2): 全部朴素多重 for 循环实现, 不借助 torch。
每个算子: 输入 numpy 数组 -> 输出 numpy 数组。

已实现:
  - conv2d_naive  卷积(七重循环, 你之前写过的 conv_hjx 思路)
  - batchnorm_2d  BatchNorm(推理模式: 用训练好的 mean/var/gamma/beta)
  - silu          SiLU = x * sigmoid(x)
  - sigmoid       sigmoid
  - add           逐元素相加
  - mul           逐元素相乘
  - concat        通道拼接
  - slice_focus   Focus 切片(切 4 块再拼通道)
"""
import numpy as np


def sigmoid(x):
    """sigmoid 激活: y = 1 / (1 + exp(-x))。逐元素, 朴素实现。"""
    out = np.empty_like(x, dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    # 用 for 循环逐元素算(朴素, 贴近新手理解; 性能不考虑)
    for _ in it:
        i = it.multi_index
        v = x[i]
        out[i] = 1.0 / (1.0 + np.exp(-v))
    return out


def silu(x):
    """SiLU(Swish): y = x * sigmoid(x)。逐元素。"""
    return mul(x, sigmoid(x))


def mul(a, b):
    """逐元素相乘(支持广播)。朴素 for 循环。"""
    # 先对齐形状
    shape = np.broadcast_shapes(a.shape, b.shape)
    out = np.zeros(shape, dtype=np.float32)
    # 用 np.broadcast 遍历
    it = np.nditer([a, b], flags=['multi_index'], op_flags=['readonly'])
    # 朴素实现: 展平后逐元素
    a_flat = np.broadcast_to(a, shape).ravel()
    b_flat = np.broadcast_to(b, shape).ravel()
    out_flat = np.zeros(shape, dtype=np.float32).ravel()
    for i in range(a_flat.size):
        out_flat[i] = a_flat[i] * b_flat[i]
    return out_flat.reshape(shape)


def add(a, b):
    """逐元素相加(支持广播)。朴素 for 循环。"""
    shape = np.broadcast_shapes(a.shape, b.shape)
    a_flat = np.broadcast_to(a, shape).ravel()
    b_flat = np.broadcast_to(b, shape).ravel()
    out_flat = np.zeros(shape, dtype=np.float32).ravel()
    for i in range(a_flat.size):
        out_flat[i] = a_flat[i] + b_flat[i]
    return out_flat.reshape(shape)


def concat(tensors, dim=1):
    """沿 dim 拼接多个张量(通道拼接最常用)。朴素实现。"""
    # 只支持 4D 张量沿 dim=1(通道)拼接, 其他维度要求相同
    assert dim == 1, "本实现只支持通道拼接(dim=1)"
    C_total = sum(t.shape[1] for t in tensors)
    N, H, W = tensors[0].shape[0], tensors[0].shape[2], tensors[0].shape[3]
    out = np.zeros((N, C_total, H, W), dtype=np.float32)
    c = 0
    for t in tensors:
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    for ci in range(t.shape[1]):
                        out[n, c + ci, h, w] = t[n, ci, h, w]
        c += t.shape[1]
    return out


def conv2d_naive(X, W, B, stride=(1, 1), padding=(1, 1), dilation=(1, 1)):
    """
    朴素卷积(七重 for 循环)。和你之前写的 conv_hjx 同思路。
    X: (N, C_in, H, W)
    W: (C_out, C_in, kH, kW)
    B: (C_out,)
    返回: (N, C_out, H_out, W_out)
    """
    N, C_in, H, W_in = X.shape
    C_out, _, kH, kW = W.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    H_out = (H + 2 * ph - dh * (kH - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw - dw * (kW - 1) - 1) // sw + 1

    # 手动补零
    X_pad = np.zeros((N, C_in, H + 2 * ph, W_in + 2 * pw), dtype=np.float32)
    X_pad[:, :, ph:ph + H, pw:pw + W_in] = X

    O = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)
    for n in range(N):
        for co in range(C_out):
            for oh in range(H_out):
                for ow in range(W_out):
                    acc = B[co]
                    for ci in range(C_in):
                        for kh in range(kH):
                            for kw in range(kW):
                                ih = oh * sh + kh * dh
                                iw = ow * sw + kw * dw
                                acc += X_pad[n, ci, ih, iw] * W[co, ci, kh, kw]
                    O[n, co, oh, ow] = acc
    return O


def batchnorm_2d(X, gamma, beta, mean, var, eps=1e-5):
    """
    BatchNorm 推理模式:
      y = gamma * (x - mean) / sqrt(var + eps) + beta
    逐元素。朴素 for 循环。
    X: (N, C, H, W),  gamma/beta/mean/var: (C,)
    """
    N, C, H, W = X.shape
    out = np.zeros_like(X, dtype=np.float32)
    for n in range(N):
        for c in range(C):
            scale = gamma[c] / np.sqrt(var[c] + eps)
            shift = beta[c] - mean[c] * scale
            for h in range(H):
                for w in range(W):
                    out[n, c, h, w] = X[n, c, h, w] * scale + shift
    return out


def slice_focus(X):
    """
    Focus 切片: 把 (N, C, H, W) 切成 4 块, 拼成 (N, 4C, H/2, W/2)。
    对应 YOLOX Focus.forward 的 slicing + cat。
    X: (N, C, H, W) -> (N, 4C, H//2, W//2)
    """
    N, C, H, W = X.shape
    H2, W2 = H // 2, W // 2
    out = np.zeros((N, 4 * C, H2, W2), dtype=np.float32)
    # 4 个切片: 左上/右上/左下/右下(对应源码 patch_top_left 等)
    # 源码顺序: top_left, bot_left, top_right, bot_right
    patches = [
        X[..., 0::2, 0::2],   # top_left
        X[..., 1::2, 0::2],   # bot_left
        X[..., 0::2, 1::2],   # top_right
        X[..., 1::2, 1::2],   # bot_right
    ]
    c = 0
    for p in patches:
        for n in range(N):
            for ci in range(C):
                for h in range(H2):
                    for w in range(W2):
                        out[n, c + ci, h, w] = p[n, ci, h, w]
        c += C
    return out
