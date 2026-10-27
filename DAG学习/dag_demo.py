# 迷你 YOLOX DAG 演示：节点=算子，边="我的输出喂给谁"
# 数据结构：邻接表。edges[A] = [B, C] 表示 A 的输出是 B 和 C 的输入
#
# 按真实源码 /workspace/YOLOX/yolox/models/darknet.py 修正：
#   forward 是纯串行 stem->dark2->dark3->dark4->dark5，
#   分叉发生在出口：dark3/dark4/dark5 三个特征同时交给 PAFPN（yolo_pafpn.py 94-95 行）
edges = {
    'input':    ['stem'],                    # 输入图 -> stem
    'stem':     ['dark2'],                   # 串行！stem 只喂给 dark2
    'dark2':    ['dark3'],                   # dark2 只喂给 dark3
    'dark3':    ['dark4', 'pafpn_c3'],       # 出口分叉：dark3 特征给 dark4 继续下采样，同时给 PAFPN
    'dark4':    ['dark5', 'pafpn_c4'],       # 同上：dark4 特征给 dark5，同时给 PAFPN
    'dark5':    ['pafpn_c5'],                # dark5 只给 PAFPN（最深层，没有更深的 stage）
    'pafpn_c3': ['pafpn_fuse'],              # PAFPN 里三个尺度汇聚（真实是上采样+Concat）
    'pafpn_c4': ['pafpn_fuse'],
    'pafpn_c5': ['pafpn_fuse'],
    'pafpn_fuse': ['head'],                  # 融合后的特征给解耦头
    'head':     [],                          # 终点，没有下游
}

def topo_sort(edges):
    """Kahn 拓扑排序：每次挑一个"所有前置都算完了"的节点执行"""
    indeg = {n: 0 for n in edges}          # indeg[n] = n 还有几个前置没算完
    for n in edges:
        for m in edges[n]:
            indeg[m] += 1                  # 每有一条边指向 m，m 的前置数 +1

    ready = [n for n, d in indeg.items() if d == 0]  # 前置为 0 的：现在就能算
    order = []
    while ready:
        n = ready.pop(0)                   # 挑一个能算的
        order.append(n)
        for m in edges[n]:                 # 它算完了，下游节点的前置数 -1
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)            # 前置清零 → 它也能算了
    return order

def verify(order, edges):
    """验证一个顺序是不是合法的拓扑序（拓扑序的定义）：
    1) 所有节点都出现、且不重复
    2) 每条边 u->v，u 必须排在 v 前面"""
    if len(order) != len(edges) or set(order) != set(edges):
        return False, '节点缺失/重复/有环（环上的节点永远清不掉前置）'
    pos = {n: i for i, n in enumerate(order)}
    for u, vs in edges.items():
        for v in vs:
            if pos[u] > pos[v]:
                return False, f'边 {u}->{v} 顺序反了：{u} 排在 {v} 后面'
    return True, 'OK：每条边的起点都在终点之前'

if __name__ == '__main__':
    print('一、真实 YOLOX 结构的拓扑序：')
    order = topo_sort(edges)
    for i, n in enumerate(order, 1):
        print(f'  {i:2d}. {n}')
    ok, msg = verify(order, edges)
    print(f'验证：{msg}')

    print()
    print('二、故意加一条边 head->input 制造环：')
    bad_edges = dict(edges)
    bad_edges['head'] = ['input']            # 环：input->...->head->input
    order2 = topo_sort(bad_edges)
    print(f'  排出来的节点数 = {len(order2)} / 总共 {len(bad_edges)} 个')
    ok2, msg2 = verify(order2, bad_edges)
    print(f'  验证：{msg2}')
