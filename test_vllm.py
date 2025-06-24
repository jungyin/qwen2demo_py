from vllm import LLM

# 使用 Hugging Face Model Hub 中的模型名称
model_name_or_path = "D:/code/transformer_models/models--Qwen--Qwen2.5-3B-Instruct"  # 确保这是正确的路径或模型ID

# 创建 LLM 实例
llm = LLM(model=model_name_or_path)

# 准备输入文本
input_text = "你的输入文本"

# 生成输出
output = llm.generate(input_text)

print(output)