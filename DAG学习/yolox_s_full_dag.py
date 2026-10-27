#!/usr/bin/env python3
"""YOLOX-s 完整算子级 DAG(279 节点) — 从 Netron SVG 解析的真实边, 同款 yolox_dag_full.py 格式。
生成: 节点表(edges 字典) + Kahn 拓扑排序 + 验证 + 运行输出
"""
import json
import sys

# ---------- 数据来源 ----------
DATA = '/workspace/env_init/yolox_s_full_edges.json'

def load_data():
    with open(DATA) as f:
        d = json.load(f)
    return d

# ---------- 拓扑排序(Kahn, 与 yolox_dag_full.py 相同) ----------
def topo_sort(edges):
    """Kahn 拓扑排序：每次挑一个"所有前置都算完了"的节点执行"""
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
    """拓扑序定义: 1) 全部节点出现且不重复 2) 每条边 u->v, u 在 v 前"""
    if len(order) != len(edges) or set(order) != set(edges):
        return False, f'节点缺失/重复/有环 (排出{len(order)}/{len(edges)})'
    pos = {n: i for i, n in enumerate(order)}
    for u, vs in edges.items():
        for v in vs:
            if pos[u] > pos[v]:
                return False, f'边 {u}->{v} 顺序反了'
    return True, 'OK：每条边的起点都在终点之前'

# ---------- 主程序 ----------
if __name__ == '__main__':
    d = load_data()
    edges = d['edges']
    node_pos = d['node_pos']
    print(f'节点表共 {len(edges)} 个节点，{sum(len(v) for v in edges.values())} 条边\n')

    print('拓扑排序执行顺序：')
    order = topo_sort(edges)
    for i, n in enumerate(order, 1):
        x, y = node_pos[n]
        print(f'  {i:3d}. {n:15s} (y={y:.0f})')
    ok, msg = verify(order, edges)
    print(f'\n验证：{msg}')
