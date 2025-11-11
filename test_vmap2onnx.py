import torch
import torch.nn as nn
from functorch import vmap
import onnx
import onnxruntime as ort
import numpy as np

# ================================
# 1. 定义一个极度简单的模型
# ================================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)  # 输入3维，输出1维
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: [3]  ← 单个样本
        return self.relu(self.linear(x))

# 实例化模型
model = SimpleModel()
model.eval()  # 推理模式

# ================================
# 2. 定义 vmap：将模型从处理单个样本扩展到处理 batch
# ================================
# 原始模型只能处理 shape [3] 的输入
# 我们用 vmap 让它自动处理 [B, 3]

def predict_single(x):
    """单个样本前向"""
    return model(x)

# 使用 vmap 创建批量版本
def create_vmap_model():
    def vmap_forward(batch_x):
        # batch_x shape: [B, 3]
        return vmap(predict_single)(batch_x)
    return vmap_forward

vmap_model = create_vmap_model()

# 测试 vmap 模型
x_single = torch.randn(3)
x_batch = torch.randn(5, 3)  # batch size = 5

out_single = model(x_single)  # [1]
out_batch = vmap_model(x_batch)  # [5, 1]

print("单样本输出:", out_single.shape)
print("批量输出:", out_batch.shape)

# ================================
# 3. 导出为 ONNX（关键：导出的是 vmap 模型的等效批量模型）
# ================================
# 注意：functorch.vmap 不是标准 nn.Module，不能直接导出
# 所以我们定义一个等效的、支持 batch 的标准模型

class BatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = model.linear  # 共享参数
        self.relu = model.relu

    def forward(self, x):
        # x shape: [B, 3]
        return self.relu(self.linear(x))  # 自动广播

batch_model = BatchModel()
batch_model.eval()

# 导出 ONNX
onnx_file = "simple_vmap_model.onnx"

dummy_input = torch.randn(1, 3)  # [B, 3]，B 可变
torch.onnx.export(
    batch_model,
    dummy_input,
    onnx_file,
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    do_constant_folding=True,
)

print(f"\n✅ 模型已导出为: {onnx_file}")

# ================================
# 4. 验证 ONNX 模型
# ================================
# 加载 ONNX 模型
session = ort.InferenceSession(onnx_file)

# 测试单个样本
input_np = x_single.unsqueeze(0).numpy()  # [1, 3]
onnx_output = session.run(None, {"input": input_np})[0]

print(f"PyTorch 单样本输出: {out_single.detach().numpy()}")
print(f"ONNX 单样本输出:   {onnx_output.squeeze()}")

# 测试批量
input_batch_np = x_batch.numpy()  # [5, 3]
onnx_batch_output = session.run(None, {"input": input_batch_np})[0]

print(f"vmap 批量输出 shape: {out_batch.shape}")
print(f"ONNX 批量输出 shape: {onnx_batch_output.shape}")

# 检查一致性
np.testing.assert_allclose(
    out_batch.detach().numpy(),
    onnx_batch_output,
    rtol=1e-4,
    atol=1e-5
)
print("\n✅ ONNX 模型输出与 PyTorch 一致！")