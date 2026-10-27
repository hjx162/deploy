# 真实 YOLOX-s DAG（块级粒度，每个节点标注真实形状 + 源码位置）
# 配置：exps/default/yolox_s.py  depth=0.33, width=0.50
#   base_channels = int(0.50*64) = 32,  base_depth = max(round(0.33*3),1) = 1
# 输入：(1, 3, 640, 640)
#
# 节点命名规则：d2/d3/d4/d5=dark2..5；p4/p5/p3=PAFPN；head3/4/5=三个尺度解耦头
# 源码：darknet.py / network_blocks.py / yolo_pafpn.py / yolo_head.py

edges = {
    # ===== Backbone: CSPDarknet（darknet.py 167-179 行 forward 串行）=====
    'input':     ['stem_focus'],                      # (1,3,640,640)
    'stem_focus':['d2_conv'],                         # Focus: slicex4->cat->Conv -> (1,32,320,320)
    'd2_conv':   ['d2_csp'],                          # BaseConv(32,64,3,s2)  -> (1,64,160,160)
    'd2_csp':    ['d3_conv'],                         # CSPLayer(64,64,n=1)   -> (1,64,160,160)
    'd3_conv':   ['d3_csp'],                          # BaseConv(64,128,3,s2) -> (1,128,80,80)
    'd3_csp':    ['d4_conv','cat1'],                  # CSPLayer(128,128,n=3) -> (1,128,80,80) ★dark3 出口分叉
    'd4_conv':   ['d4_csp'],                          # BaseConv(128,256,3,s2)-> (1,256,40,40)
    'd4_csp':    ['d5_conv','cat0'],                  # CSPLayer(256,256,n=3) -> (1,256,40,40) ★dark4 出口分叉
    'd5_conv':   ['d5_spp'],                          # BaseConv(256,512,3,s2)-> (1,512,20,20)
    'd5_spp':    ['d5_csp'],                          # SPPBottleneck(512,512) -> (1,512,20,20)
    'd5_csp':    ['lateral0'],                        # CSPLayer(512,512,n=1) -> (1,512,20,20) ★dark5 出口

    # ===== Neck: PAFPN（yolo_pafpn.py 92-116 行 forward）=====
    'lateral0':  ['up0','cat3'],                      # BaseConv(512,256,1)   -> (1,256,20,20) ★分叉
    'up0':       ['cat0'],                            # Upsample x2 nearest   -> (1,256,40,40)
    'cat0':      ['c3_p4'],                           # cat(up0, d4_csp)      -> (1,512,40,40) ★汇聚
    'c3_p4':     ['reduce1'],                         # CSPLayer(512,256,n=1) -> (1,256,40,40)
    'reduce1':   ['up1','cat2'],                      # BaseConv(256,128,1)   -> (1,128,40,40) ★分叉
    'up1':       ['cat1'],                            # Upsample x2 nearest   -> (1,128,80,80)
    'cat1':      ['c3_p3'],                           # cat(up1, d3_csp)      -> (1,256,80,80) ★汇聚
    'c3_p3':     ['bu2','head3'],                     # CSPLayer(256,128,n=1) -> (1,128,80,80)=pan_out2 ★出口
    'bu2':       ['cat2'],                            # BaseConv(128,128,3,s2)-> (1,128,40,40)
    'cat2':      ['c3_n3'],                           # cat(bu2, reduce1)     -> (1,256,40,40) ★汇聚
    'c3_n3':     ['bu1','head4'],                     # CSPLayer(256,256,n=1) -> (1,256,40,40)=pan_out1 ★出口
    'bu1':       ['cat3'],                            # BaseConv(256,256,3,s2)-> (1,256,20,20)
    'cat3':      ['c3_n4'],                           # cat(bu1, lateral0)    -> (1,512,20,20) ★汇聚
    'c3_n4':     ['head5'],                           # CSPLayer(512,512,n=1) -> (1,512,20,20)=pan_out0 ★出口

    # ===== Head: Decoupled Head（yolo_head.py 142-161 行）=====
    # 每个尺度：stem(1x1->128) 后分三支 cls/reg/obj（解耦头！）
    'head3':     ['h3_cls','h3_reg'],                 # stem BaseConv(128,128,1)->(1,128,80,80) ★分叉
    'h3_cls':    ['h3_out'],                          # 2xConv3x3 + Conv2d(128,80)  -> (1,80,80,80)  80类
    'h3_reg':    ['h3_out'],                          # 2xConv3x3 + Conv2d(128,4)+Conv2d(128,1) -> 4+1
    'h3_out':    [],
    'head4':     ['h4_cls','h4_reg'],                 # stem ->(1,128,40,40)
    'h4_cls':    ['h4_out'],                          # -> (1,80,40,40)
    'h4_reg':    ['h4_out'],                          # -> 4+1
    'h4_out':    [],
    'head5':     ['h5_cls','h5_reg'],                 # stem ->(1,128,20,20)
    'h5_cls':    ['h5_out'],                          # -> (1,80,20,20)
    'h5_reg':    ['h5_out'],                          # -> 4+1
    'h5_out':    [],
}

# 每个节点输出形状（YOLOX-s, 输入 640x640, 通道 32/64/128/256/512）
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
    'head3': '(1,128,80,80)', 'h3_cls': '(1,80,80,80)', 'h3_reg': '4+1 通道', 'h3_out': '输出3',
    'head4': '(1,128,40,40)', 'h4_cls': '(1,80,40,40)', 'h4_reg': '4+1 通道', 'h4_out': '输出3',
    'head5': '(1,128,20,20)', 'h5_cls': '(1,80,20,20)', 'h5_reg': '4+1 通道', 'h5_out': '输出3',
}

def topo_sort(edges):
    indeg = {n: 0 for n in edges}
    for n in edges:
        for m in edges[n]:
            indeg[m] += 1
    ready = [n for n, d in indeg.items() if d == 0]
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in edges[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    return order

def verify(order, edges):
    if len(order) != len(edges) or set(order) != set(edges):
        return False, '节点缺失/有环'
    pos = {n: i for i, n in enumerate(order)}
    for u, vs in edges.items():
        for v in vs:
            if pos[u] > pos[v]:
                return False, f'边 {u}->{v} 顺序反了'
    return True, 'OK'

if __name__ == '__main__':
    order = topo_sort(edges)
    ok, msg = verify(order, edges)
    print(f'真实 YOLOX-s DAG：共 {len(edges)} 个节点，拓扑序验证：{msg}\n')
    print('执行顺序（节点 → 输出形状）：')
    for i, n in enumerate(order, 1):
        print(f'  {i:2d}. {n:12s} -> {shape_of[n]}')
