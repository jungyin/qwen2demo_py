import os
import time
from torchprofile import profile_macs

import numpy as np

import torch 

import torch.nn.functional as F

from utils import add_detailed_export_logging

import transformers.masking_utils as m_utils

# 这个方法不生效，需要手动点过去，吧这个值赋为false
m_utils._is_torch_greater_or_equal_than_2_6= False

os.environ.setdefault("ONNX_PYTORCH_DEBUG", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "D:\\code\\transformer_models\\models--Qwen--Qwen2.5-0.5B-Instruct"
# model_name = "D:\\code\\transformer_models\\models--Qwen--Qwen2.5-3B-Instruct"
# model_name = "D:\\code\\transformer_models\\models--Qwen--Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int8"
# model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
# model_name = "./source/qwen2.5_1.5b_math"
outtype = torch.bfloat16
# fpath = "./onnx/math-1.5b/model.onnx"
fpath = "./onnx/qwen2_3b_bf16/model.onnx"
# fpath = "./onnx/qwen2_0.5b_new/model.onnx"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)

# 使用 torch.fx.symbolic_trace 将模型转换为 FX 图表示
# symbolic_traced = torch.fx.symbolic_trace(model.model)

# 获取所有节点
# nodes = list(symbolic_traced.graph.nodes)

print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = "def get_hr(y, sr=30, min=30, max=180):\
    nfftv =int( 1e5/sr)+1\
    cc = np.min((len(y)-1, 256)) \
    if(nfftv%2 !=0):\
        nfftv+=1\
    p, q = welch(y, sr, nfft=int(nfftv), nperseg=np.min((len(y)-1, 256)))\
    iss = p[1660::]\
    max = max / 60\
    min = min / 60\
    pindex = (p>min)&(p<max)\
    pp = p[pindex]\
    q = q[pindex]\
    qindex = np.argmax(q)\
    rp = p[pindex]\
    return rp[qindex]*60 代码分析,并根据这个代码，提供一个简单的可复用场景,并着重在里面说你做了什么工作，做了什么优化？"

# prompt= "请帮我写一个傅里叶变化公式,并使用python代码简单复现一下"
# prompt = "\n请给我一段，有多行代码的latex公式"
# prompt = "\n一个prompt中，包含那些参数？除了role,content，还有什么"
prompt = "请给我生成一个带有吃苹果的小女孩的图片,然后请我吃苹果"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.","user":"yinjun"},
    {"role": "user", "content": prompt,"user":"yinjun"}
]


def convert_bf16_fp16_to_fp32(model):
    for param in model.parameters():
        if param.dtype == torch.bfloat16 or param.dtype == torch.float16:
            param.data = param.data.to(dtype=outtype)
    for buffer in model.buffers():
        if buffer.dtype == torch.bfloat16 or buffer.dtype == torch.float16:
            buffer.data = buffer.data.to(dtype=outtype)
    return model


def build_cache_random(model):
      
    num_hidden_layers = model.base_model.config.num_hidden_layers
    head_dim = model.base_model.config.hidden_size // model.base_model.config.num_attention_heads
    kvszie = 2 
    # 1,2,?,head_dim
    
    qvshape = [1,2,0,head_dim]
    
    return np.ones([num_hidden_layers,kvszie]+qvshape)

# 预处理 qwen2的输入数据
def mypre(
   input_ids,attention_mask=None, inputs_embeds=None
):

  # create position_ids on the fly for batch generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    

    model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            # "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


# messages = "这是一个测试句子。"


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
text1 = text
input_names = ["input_ids","attention_mask","position_ids"]
input_names.append("past_key_values")
output_names = ["last_hidden_state"]
output_names .append( "past_key_values")


model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

input_ids = model_inputs.data['input_ids']
attention_mask = model_inputs.data['attention_mask']


position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)
past_key_values =torch.from_numpy( build_cache_random(model)).to(position_ids.device).to(outtype)





# input = mypre(model_inputs['input_ids'],model_inputs['attention_mask'])


model.base_model.kk = False

tmodel = model
tmodel.kk = True

#cache_list = numpy.array()

outmodel =  convert_bf16_fp16_to_fp32(tmodel) 
outmodel =  tmodel


# traced_model = torch.jit.script(outmodel)


dummy_input = (input_ids,attention_mask,position_ids,past_key_values)  # 根据实际情况调整输入尺寸

# 计算FLOPs，将所有输入放入tuple中
# macs = profile_macs(outmodel,(input_ids,attention_mask,position_ids,past_key_values) )

# print(f"Model FLOPs: {macs}")

model = outmodel

test = torch.from_numpy(np.array([[47,47,47,0]])).to(torch.long)
max_size = 1024
input_ids = F.pad(input_ids,(0,max_size-input_ids.shape[-1],0,0),value=0)
attention_mask = F.pad(attention_mask,(0,max_size-attention_mask.shape[-1],0,0),value=0)
position_ids = F.pad(position_ids,(0,max_size-position_ids.shape[-1],0,0),value=0)

outpad = (0,0,0,max_size-past_key_values.shape[-2])
past_key_values = F.pad(past_key_values,outpad,value=0)




traced_model = torch.jit.trace(outmodel, (input_ids,attention_mask,position_ids,past_key_values) )
add_detailed_export_logging(traced_model)
# traced_model.save("traced_model.pt") 
# find_unexpected_tensor_types(traced_model)
# outtest = model(input_ids,attention_mask,position_ids,past_key_values)
# torch.onnx.export(outmodel, (input_ids,attention_mask,position_ids,past_key_values,test) ,input_names=input_names ,output_names=output_names , f=fpath ,dynamic_axes={'input_ids':[1],'attention_mask':[1],'position_ids':[1],'last_hidden_state':[1],'past_key_values':[4]},opset_version=18)
torch.onnx.export(traced_model, (input_ids,attention_mask,position_ids,past_key_values) ,input_names=input_names ,output_names=output_names ,verbose= True, f=fpath ,dynamic_axes={'input_ids':[1],'attention_mask':[1],'position_ids':[1],'last_hidden_state':[1],'past_key_values':[4]},opset_version=18)
# torch.onnx.export(outmodel, (input_ids,attention_mask,position_ids) ,input_names=input_names ,output_names=output_names , f="./onnx/model32.onnx" ,dynamic_axes={'input_ids':[1],'attention_mask':[1],'position_ids':[1],'last_hidden_state':[1]},opset_version = 18)


tmodel.kk=False




generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2000
)


generated_ids_build = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# np.save("realout.npy",generated_ids_build)


response = tokenizer.batch_decode(generated_ids_build, skip_special_tokens=True)
print(response[0])

cc = response[0]
cc1 = response[0].split("\n")

k = 10