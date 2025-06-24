import onnx

# 加载ONNX模型
model_path = './vllm_demo/model/model32.onnx'  # 替换为你的ONNX模型路径
onnx_model = onnx.load(model_path)

# 检查模型是否加载成功
if onnx_model is not None:
    # 获取模型中的图
    graph = onnx_model.graph
    
    # 统计节点数量
    node_count = len(graph.node)
    
    # 打印每个节点的信息（索引、名称、操作类型）
    nodes_info = [(i, node.name, node.op_type) for i, node in enumerate(graph.node)]
    
    print(f"模型中总共有 {node_count} 个节点.")
    for info in nodes_info:
        print(info)
else:
    print("模型加载失败，请检查路径是否正确。")