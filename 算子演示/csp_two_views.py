# 演示:CSPLayer 的"一个节点 vs 展开成算子"
# 左边 = 我们 37 节点 DAG 的抽象视角(1 个节点)
# 右边 = Netron/推理引擎的真实视角(展开成 N 个算子)

csp_expanded = [
    # (算子, 作用, 对应源码)
    ("Conv_70",   "conv1: 1x1 卷积(主路)",           "self.conv1"),
    ("Sigmoid_71", "SiLU 的一部分",                   "act"),
    ("Mul_72",    "SiLU = x*sigmoid(x)",              "act"),
    ("Sigmoid_91", "conv2 的 SiLU(旁路)",             "self.conv2"),
    ("Mul_92",    "旁路激活",                         "act"),
    ("Add_93",    "Bottleneck 残差相加(shortcut)",    "y = y + x"),
    ("Concat_94", "两条路拼接",                       "torch.cat"),
    ("Conv_95",   "conv3: 1x1 融合卷积",              "self.conv3"),
    ("Sigmoid_96", "conv3 的 SiLU",                   "act"),
    ("Mul_97",    "激活",                             "act"),
    ("Conv_98",   "Bottleneck.conv1: 1x1",            "self.conv1"),
    ("Sigmoid_99", "激活",                            "act"),
    ("Mul_100",   "激活",                             "act"),
    ("Conv_101",  "Bottleneck.conv2: 3x3",            "self.conv2"),
    ("Sigmoid_102", "激活",                           "act"),
    ("Mul_103",   "激活",                             "act"),
]

print("=" * 60)
print("同一个 d2_csp,两种视角")
print("=" * 60)
print()
print("【37 节点 DAG 的视角】—— 抽象层")
print("  d2_csp  = 一个节点")
print("  输入 (1,64,160,160) -> 输出 (1,64,160,160)")
print()
print("【Netron / 推理引擎的视角】—— 展开层")
print("  d2_csp  = 16 个算子:")
print()
for i, (name, desc, src) in enumerate(csp_expanded, 1):
    print(f"    {i:2d}. {name:12s} {desc:32s} ({src})")
print()
print("=" * 60)
print("关键:这两个视角都对,只是粒度不同")
print("=" * 60)
