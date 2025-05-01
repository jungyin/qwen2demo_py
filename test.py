from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers
import torch
import time;
import numpy as np

to = torch.from_numpy(np.array([1,2,3,4,5,6,7,8,9,10]))

o_past_key_values = ((to,to),(to,to))

opkv = [[k for k in list(j)] for j in list(o_past_key_values)]


cc = torch.stack([torch.stack(row) for row in opkv])
opkv = torch.stack(opkv,dim=0)



# 加载预训练模型和分词器
model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
# model_name = "Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int8"
# model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# 初始化对话历史
dialog_history = ""
model.to(device)

# 自定义Streamer类


class CustomStreamer(transformers.generation.streamers.BaseStreamer):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.start = True
        self.time123 = time.time()
        self.cachetime = []
    def put(self, value):
        self.cachetime .append(time.time()-self.time123)
        self.time123 = time.time()

        # 将模型生成的ID转换为文本
        if len(value.shape) > 1: value = value[0]
        text = self.tokenizer.decode(value, skip_special_tokens=True)
        
        # 判断是否是首次调用以避免重复输出
        if not self.start:
            print(text, end='', flush=True)
        self.start = False
        

    def end(self):
        # 结束时可以进行一些操作，这里不做任何处理
        meanValue = np.mean(np.array(self.cachetime))
        oneValue = 1.0 / meanValue
        print("每秒tokens:",oneValue)
        pass

custom_streamer = CustomStreamer(tokenizer)
def get_model_response(input_text):
    global dialog_history
    global custom_streamer
    
    prompt= "请帮我写一个傅里叶变化公式,并使用python代码简单复现一下"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.","user":"yinjun"},
        {"role": "user", "content": prompt,"user":"yinjun"}
    ]

    # 将用户输入添加到对话历史
    dialog_history += f"User: {input_text}\n"
    
    # 编码输入（包括对话历史）
    inputs = tokenizer(dialog_history + "System:", return_tensors="pt").to(device)
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # 获取模型生成的回复
    outputs = model.generate(inputs['input_ids'], max_length=1000, pad_token_id=tokenizer.eos_token_id, streamer = custom_streamer)
    
    # 解码模型输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 更新对话历史
    dialog_history += f"{response}\n"
    
    return ""

# 示例对话
# print(get_model_response("你好，能告诉我一些关于科技的趋势吗？"))
# print(get_model_response("那人工智能方面呢？"))
# print(get_model_response("如果我想飞应该怎么做"))
# print(get_model_response("如果1+1，等于多少"))
# print(get_model_response("再加1呢，等于多少"))
print(get_model_response("请帮我写一个傅里叶变化,使用python"))