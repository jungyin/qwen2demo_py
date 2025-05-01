import torch

# 假设model是你已经定义好的PyTorch模型，并且已经移动到GPU上
model = model.cpu()
dummy_input = torch.randn(1, 3, 224, 224).cpu()  # 根据实际情况调整输入尺寸

# 测量推理时间
start = torch.cpu.Event(enable_timing=True)
end = torch.cpu.Event(enable_timing=True)

start.record()
with torch.no_grad():
    output = model(dummy_input)
end.record()

torch.cpu.synchronize()  # 确保所有计算都已完成

time_elapsed_ms = start.elapsed_time(end)  # 得到的时间是以毫秒为单位