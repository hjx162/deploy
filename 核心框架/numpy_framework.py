"""
纯 NumPy 手搓推理框架(v7, 完全仿 ONNX Runtime 架构, 图格式与真实 ONNX 同构)。

文件分离(标准做法):
  yolox_graph.json   ← 图定义(与真实 onnx.ModelProto.graph 同构: node列表含权重输入 + initializer)
  numpy_ops.py       ← 算子库(纯 NumPy, 朴素 for 循环)
  numpy_framework.py ← 本框架(读图 -> 拓扑排序 -> 执行)

v7 改动: 按 ONNX Runtime 真实执行模型重构——
  删掉 load_graph 里的 upstream/all_outputs/init_names(冗余的"上游节点名"概念),
  graph 变成对 json 的原样搬运; run_inference 改"统一值表":
  权重预载入 tensors, 执行节点时按 node.inputs 的张量名查表取值,
  不区分数据/权重(谁是数据谁是权重是 ONNX 算子规范的事, 如 Conv: 输入0=X, 输入1=W)。
v6 改动: 合并 compute_indeg 进 topo_sort(单函数, 教科书写法), 加 debug 开关打印入度表。
v5 改动: topo_sort 改为自包含修复版——入度只数"图内节点产出的输入",
  权重/图输入(images)不算边; 不再依赖 load_graph 预过滤。

用法:
  python numpy_framework.py
"""
import json
import numpy as np
import numpy_ops as ops

# ============ 1. 读图(模拟 ONNX Runtime 加载 ONNX) ============
def load_graph(path='yolox_graph.json'):
    """读图文件, 转成内部表示: {node_name: {op, params, inputs, outputs}}。

    与真实 onnx.ModelProto.graph 同构: node 的 inputs/outputs 就是张量名列表。
    不预存 upstream——依赖关系只在 topo_sort 排序时按需推导(ONNX Runtime 同款设计)。
    """
    with open(path) as f:
        proto = json.load(f)

    graph = {}

    # 图输入也作为节点(张量源)
    input_name = proto['input']['name']
    graph[input_name] = {
        'op': 'Input',
        'params': {'shape': proto['input']['shape']},
        'inputs': [],
        'outputs': [input_name],
    }

    for node in proto['node']:
        graph[node['name']] = {
            'op': node['op_type'],
            'params': node.get('attribute', {}),   # ONNX 叫 attribute
            'inputs': node.get('inputs', []),      # 张量名列表(含权重名, 与 json 一致)
            'outputs': node.get('outputs', []),
        }
    return graph, proto

# ============ 2. 拓扑排序(Kahn) ============
def topo_sort(graph, debug=False):
    """Kahn 拓扑排序: 每次挑一个"所有前置都算完了"的节点执行。

    入度只数"图内节点产出的输入"!
      graph[n]['inputs'] 是张量名列表(含权重), 权重(stem_conv_w等)不是节点,
      不参与"等谁先算完"这件事, 所以必须跳过。
    做法: 先建反查表 producer{张量名: 产出它的节点名},
          输入张量若在 producer 里才算一条图内边。

    debug=True 时打印每个节点的入度表(教学用)。
    """
    # 1) 反查表: 输出的张量名 <- 节点名
    producer = {}
    for n in graph:
        for t in graph[n]['outputs']:
            producer[t] = n # 输出 映射到 上一个节点的名称

    # 2) 入度: 只数图内边(权重/外部输入不在 producer 里, 自动跳过)
    #    按"上游节点"计数: 同一节点被同一消费者用多次(如 concat(x,x)),
    #    入度只 +1, 与下方主循环的减1(按 producer 匹配, 每个消费者一次)对称。
    indeg = {n: 0 for n in graph}
    for n in graph:
        seen = set()                     # 本节点已数过的上游节点
        for t in graph[n]['inputs']:
            p = producer.get(t)          # 这个输入张量是哪个节点产的
            if p is not None and p not in seen:
                seen.add(p)
                indeg[n] += 1            # 同 producer 只 +1

    if debug:
        print('    每个节点的入度(只数图内边; 权重/图片不是节点, 不数):')
        for n, d in indeg.items():
            mark = '  ← 起点, 先算它' if d == 0 else ''
            print(f'    {n:15s} indeg = {d}{mark}')

    # 3) Kahn 主循环
    ready = [n for n, d in indeg.items() if d == 0]  # 前置为 0 的: 现在就能算
    order = []
    while ready:
        n = ready.pop(0)             # 挑一个能算的
        order.append(n)
        for m in graph:              # n 算完了 → 消费它输出的节点入度 -1
            for t in graph[m]['inputs']:
                if producer.get(t) == n:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        ready.append(m)
                    break            # 一个节点只被 n 喂一次, 减一次即可
    return order

# ============ 3. 权重初始化(形状从 initializer 声明来) ============
def init_weights(rng, proto):
    """根据 initializer 的形状声明生成随机权重。返回 {权重名: ndarray}"""
    w = {}
    for init in proto.get('initializer', []):
        name = init['name']
        shape = init.get('shape')
        if shape is None:
            continue  # 无权重(如 Slice)
        # BN 的 var 必须是正数(否则 sqrt 为 nan)
        if name.endswith('_var'):
            w[name] = rng.uniform(0.5, 1.5, shape).astype(np.float32)
        else:
            w[name] = rng.standard_normal(shape).astype(np.float32) * 0.1
    return w

# ============ 4. 执行引擎 ============
def run_inference(graph, weights, x):
    """执行引擎(ONNX Runtime 风格): 统一值表, 按张量名取值/存值。

    tensors = 权重(initializer)预载入 + 运行中节点输出写入。
    执行节点时按 node['inputs'] 里的张量名查表取输入, 算完把
    node['outputs'] 里的张量名写进表。不区分数据/权重——
    谁是数据谁是权重是 ONNX 算子规范的事(如 Conv: 输入0=X, 输入1=W)。
    """
    order = topo_sort(graph)
    tensors = dict(weights)          # 统一值表: 权重预载入
    for name in order:
        node = graph[name]
        op = node['op']
        ins = [tensors[t] for t in node['inputs']]   # 按名查表(数据+权重一视同仁)
        if op == 'Input':
            out = x
        elif op == 'Output':
            out = ins[0]
        elif op == 'Slice':
            out = ops.slice_focus(ins[0])
        elif op == 'Conv':
            a = node['params']
            stride = a['strides'][0]
            pad = a['pads'][0]
            out = ops.conv2d_naive(
                ins[0], ins[1],
                np.zeros(ins[1].shape[0], dtype=np.float32),  # 无 bias(BN 吸收)
                stride=(stride, stride),
                padding=(pad, pad),
            )
        elif op == 'BatchNormalization':
            eps = node['params'].get('epsilon', 1e-5)
            out = ops.batchnorm_2d(
                ins[0], ins[1], ins[2], ins[3], ins[4], eps)
        elif op == 'SiLU':
            out = ops.silu(ins[0])
        else:
            raise ValueError(f'未知算子: {op}')
        # 输出写入"张量名"下(不是节点名): 统一值表的 key 就是张量名
        # Output 节点 outputs 为空, 用节点名(= 图输出名, 如 'output')
        for t in (node['outputs'] or [name]):
            tensors[t] = out
    return tensors, order

# ============ 5. 主程序 ============
if __name__ == '__main__':
    print('=' * 60)
    print('纯 NumPy 推理框架 v7(图与真实 ONNX 同构)')
    print('=' * 60)

    # 1. 读图
    graph, proto = load_graph()
    print(f'\n[1] 加载图: {len(graph)} 个节点(含图输入)')
    print(f'    输入: {proto["input"]["shape"]}')
    print(f'    initializer(权重): {len(proto.get("initializer", []))} 个')

    # 2. 拓扑排序(debug=True 打印入度表)
    print(f'\n[2] 拓扑排序:')
    order = topo_sort(graph, debug=True)
    print(f'    执行顺序:')
    for i, n in enumerate(order, 1):
        print(f'    {i:2d}. {n}')

    # 3. 权重 + 输入
    rng = np.random.default_rng(42)
    weights = init_weights(rng, proto)
    x = rng.standard_normal((1, 3, 32, 32)).astype(np.float32)

    # 4. 执行
    print(f'\n[3] 执行推理...')
    tensors, order = run_inference(graph, weights, x)
    print(f'[4] 各节点输出张量形状(统一值表, key=张量名):')
    for n in order:
        for t in (graph[n]['outputs'] or [n]):
            if t in tensors:
                print(f'    {t:15s} -> {tensors[t].shape}')

    out = tensors['output']
    print(f'\n✅ 推理完成! 输出: {out.shape}, mean={out.mean():.4f}')
    print('(图格式与真实 ONNX 同构: 换成 onnx.load() 的结果, 框架逻辑不用改)')
