# ============================================================
# 完整示例：真实 YOLOX-s 的 DAG 节点表 + Kahn 拓扑排序
# 配置：exps/default/yolox_s.py  depth=0.33, width=0.50
#   base_channels = int(0.50*64) = 32, base_depth = max(round(0.33*3),1) = 1
# 输入：(1, 3, 640, 640)
# ============================================================

# ------------------------------------------------------------
# 一、节点表（edges 字典）
# 规则：edges[A] = [B, C] 表示 "A 的输出喂给 B 和 C"
# 节点命名：d2/d3/d4/d5 = dark2..5；p4/p5/p3 = PAFPN；head3/4/5 = 三个尺度解耦头
# 源码位置：darknet.py / network_blocks.py / yolo_pafpn.py / yolo_head.py
# ------------------------------------------------------------
edges = {
    # ===== Backbone: CSPDarknet（darknet.py 167-179 行 forward，串行主链）=====
    'input':      ['stem_focus'],                   # 输入图像
    'stem_focus': ['d2_conv'],                      # Focus: 切4块+拼通道+Conv -> (1,32,320,320)
    'd2_conv':    ['d2_csp'],                       # BaseConv(32,64,3,s2)     -> (1,64,160,160)
    'd2_csp':     ['d3_conv'],                      # CSPLayer(64,64,n=1)      -> (1,64,160,160)
    'd3_conv':    ['d3_csp'],                       # BaseConv(64,128,3,s2)    -> (1,128,80,80)
    'd3_csp':     ['d4_conv', 'cat1'],              # CSPLayer(128,128,n=3)    -> (1,128,80,80) ★dark3 出口分叉
    'd4_conv':    ['d4_csp'],                       # BaseConv(128,256,3,s2)   -> (1,256,40,40)
    'd4_csp':     ['d5_conv', 'cat0'],              # CSPLayer(256,256,n=3)    -> (1,256,40,40) ★dark4 出口分叉
    'd5_conv':    ['d5_spp'],                       # BaseConv(256,512,3,s2)   -> (1,512,20,20)
    'd5_spp':     ['d5_csp'],                       # SPPBottleneck(512,512)   -> (1,512,20,20)
    'd5_csp':     ['lateral0'],                     # CSPLayer(512,512,n=1)    -> (1,512,20,20) ★dark5 出口

    # ===== Neck: PAFPN（yolo_pafpn.py 92-116 行 forward，先降后升）=====
    'lateral0':   ['up0', 'cat3'],                  # BaseConv(512,256,1)      -> (1,256,20,20) ★分叉
    'up0':        ['cat0'],                         # Upsample x2 nearest     -> (1,256,40,40)
    'cat0':       ['c3_p4'],                        # cat(up0, d4_csp)        -> (1,512,40,40) ★汇聚
    'c3_p4':      ['reduce1'],                      # CSPLayer(512,256,n=1)   -> (1,256,40,40)
    'reduce1':    ['up1', 'cat2'],                  # BaseConv(256,128,1)      -> (1,128,40,40) ★分叉
    'up1':        ['cat1'],                         # Upsample x2 nearest     -> (1,128,80,80)
    'cat1':       ['c3_p3'],                        # cat(up1, d3_csp)        -> (1,256,80,80) ★汇聚
    'c3_p3':      ['bu2', 'head3'],                 # CSPLayer(256,128,n=1)   -> (1,128,80,80) ★出口
    'bu2':        ['cat2'],                         # BaseConv(128,128,3,s2)  -> (1,128,40,40)
    'cat2':       ['c3_n3'],                        # cat(bu2, reduce1)       -> (1,256,40,40) ★汇聚
    'c3_n3':      ['bu1', 'head4'],                 # CSPLayer(256,256,n=1)   -> (1,256,40,40) ★出口
    'bu1':        ['cat3'],                         # BaseConv(256,256,3,s2)  -> (1,256,20,20)
    'cat3':       ['c3_n4'],                        # cat(bu1, lateral0)      -> (1,512,20,20) ★汇聚
    'c3_n4':      ['head5'],                        # CSPLayer(512,512,n=1)   -> (1,512,20,20) ★出口

    # ===== Head: 解耦头（yolo_head.py 142-161 行，每个尺度一分三）=====
    'head3':      ['h3_cls', 'h3_reg'],             # stem(1x1->128)          -> (1,128,80,80) ★分叉
    'h3_cls':     ['h3_out'],                       # 2xConv3x3 + Conv(128,80) -> (1,80,80,80)  80类
    'h3_reg':     ['h3_out'],                       # 2xConv3x3 + Conv(128,4)+Conv(128,1) -> 4+1
    'h3_out':     [],
    'head4':      ['h4_cls', 'h4_reg'],             # stem                    -> (1,128,40,40)
    'h4_cls':     ['h4_out'],                       #                          -> (1,80,40,40)
    'h4_reg':     ['h4_out'],                       #                          -> 4+1
    'h4_out':     [],
    'head5':      ['h5_cls', 'h5_reg'],             # stem                    -> (1,128,20,20)
    'h5_cls':     ['h5_out'],                       #                          -> (1,80,20,20)
    'h5_reg':     ['h5_out'],                       #                          -> 4+1
    'h5_out':     [],
}

# 每个节点输出形状（YOLOX-s, 输入 640x640）
shape_of = {
    'input': '(1,3,640,640)', 'stem_focus': '(1,32,320,320)',
    'd2_conv': '(1,64,160,160)', 'd2_csp': '(1,64,160,160)',
    'd3_conv': '(1,128,80,80)',  'd3_csp': '(1,128,80,80)',
    'd4_conv': '(1,256,40,40)',  'd4_csp': '(1,256,40,40)',
    'd5_conv': '(1,512,20,20)',  'd5_spp': '(1,512,20,20)', 'd5_csp': '(1,512,20,20)',
    'lateral0': '(1,256,20,20)', 'up0': '(1,256,40,40)', 'cat0': '(1,512,40,40)',
    'c3_p4': '(1,256,40,40)',    'reduce1': '(1,128,40,40)', 'up1': '(1,128,80,80)',
    'cat1': '(1,256,80,80)',     'c3_p3': '(1,128,80,80)',
    'bu2': '(1,128,40,40)',      'cat2': '(1,256,40,40)', 'c3_n3': '(1,256,40,40)',
    'bu1': '(1,256,20,20)',      'cat3': '(1,512,20,20)', 'c3_n4': '(1,512,20,20)',
    'head3': '(1,128,80,80)', 'h3_cls': '(1,80,80,80)', 'h3_reg': '4+1通道', 'h3_out': '输出3',
    'head4': '(1,128,40,40)', 'h4_cls': '(1,80,40,40)', 'h4_reg': '4+1通道', 'h4_out': '输出3',
    'head5': '(1,128,20,20)', 'h5_cls': '(1,80,20,20)', 'h5_reg': '4+1通道', 'h5_out': '输出3',
}

# ------------------------------------------------------------
# 二、topo_sort：Kahn 拓扑排序（完整实现）
# ------------------------------------------------------------
def topo_sort(edges):
    """Kahn 拓扑排序：每次挑一个"所有前置都算完了"的节点执行"""
    indeg = {n: 0 for n in edges}          # ① 账本初始化：每节点记 0
    for n in edges:
        for m in edges[n]:
            indeg[m] += 1                  # ② 数"指向我的边"→ 入度（前置数）
    ready = [n for n, d in indeg.items() if d == 0]  # ③ 入度为 0 的：现在就能算
    order = []                             # ④ 结果列表
    while ready:                           # ⑤ 主循环：只要有能算的
        n = ready.pop(0)                   # ⑥ 取出一个可执行节点
        order.append(n)                    # ⑦ 记录到结果
        for m in edges[n]:                 # ⑧ 遍历它的下游
            indeg[m] -= 1                  # ⑨ 下游前置数 -1
            if indeg[m] == 0:              # ⑩ 清零 → 万事俱备
                ready.append(m)            # ⑪ 解锁下游，加入可执行集合
    return order                           # ⑫ 返回拓扑序

# ------------------------------------------------------------
# 三、验证：这个顺序是不是合法的拓扑序
# ------------------------------------------------------------
def verify(order, edges):
    """拓扑序的定义：1) 全部节点出现且不重复；2) 每条边 u->v，u 在 v 前面"""
    if len(order) != len(edges) or set(order) != set(edges):
        return False, '节点缺失/重复/有环'
    pos = {n: i for i, n in enumerate(order)}
    for u, vs in edges.items():
        for v in vs:
            if pos[u] > pos[v]:
                return False, f'边 {u}->{v} 顺序反了'
    return True, 'OK：每条边的起点都在终点之前'

# ------------------------------------------------------------
# 四、运行
# ------------------------------------------------------------
if __name__ == '__main__':
    print(f'节点表共 {len(edges)} 个节点，{sum(len(v) for v in edges.values())} 条边\n')
    print('拓扑排序执行顺序：')
    order = topo_sort(edges)
    for i, n in enumerate(order, 1):
        print(f'  {i:2d}. {n:12s} -> {shape_of[n]}')
    ok, msg = verify(order, edges)
    print(f'\n验证：{msg}')
