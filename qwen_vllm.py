
import os
from vllm import LLM, SamplingParams
from mystreamer import CustomStreamer
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "D:\\code\\transformer_models\\models--Qwen--Qwen2.5-Coder-0.5B-Instruct"
# model_name = "D:\\code\\transformer_models\\models--Qwen--Qwen2.5-Coder-0.5B-Instruct-GPTQ-Int8"
# model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
# model_name = "./source/qwen2.5_1.5b_math"
import numpy as np

import torch 

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto",
#     local_files_only=True
# )
# 10s
# 使用 torch.fx.symbolic_trace 将模型转换为 FX 图表示
# symbolic_traced = torch.fx.symbolic_trace(model.model)
# 获取所有节点
# nodes = list(symbolic_traced.graph.nodes)
# sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

tokenizer = AutoTokenizer.from_pretrained(model_name)
custom_streamer = CustomStreamer(tokenizer)

model = LLM(model=model_name)
print(model)

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
    return rp[qindex]*60 代码分析"

prompt= "请帮我写一个傅里叶变化公式,并使用python代码简单复现一下"
# prompt = "\n请给我一段，有多行代码的latex公式"
# prompt = "\n一个prompt中，包含那些参数？除了role,content，还有什么"
# prompt = "请给我生成一个带有吃苹果的小女孩的图片,然后请我吃苹果"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.","user":"yinjun"},
    {"role": "user", "content": prompt,"user":"yinjun"}
]


def convert_bf16_fp16_to_fp32(model):
    for param in model.parameters():
        if param.dtype == torch.bfloat16 or param.dtype == torch.float16:
            param.data = param.data.to(dtype=torch.float16)
    for buffer in model.buffers():
        if buffer.dtype == torch.bfloat16 or buffer.dtype == torch.float16:
            buffer.data = buffer.data.to(dtype=torch.float16)
    return model


def build_cache_random(model):
      
    num_hidden_layers = model.base_model.num_hidden_layers
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


model_inputs = tokenizer([text], return_tensors="pt")

# input_ids = model_inputs.data['input_ids']
# attention_mask = model_inputs.data['attention_mask']

# position_ids = attention_mask.long().cumsum(-1) - 1
# position_ids.masked_fill_(attention_mask == 0, 1)
# past_key_values =torch.from_numpy( build_cache_random(model)).to(position_ids.device).to(torch.float16)
# input = mypre(model_inputs['input_ids'],model_inputs['attention_mask'])


# model.base_model.kk = False
# model.kk = False

tmodel = model
tmodel.kk = True

#cache_list = numpy.array()

# outmodel =  convert_bf16_fp16_to_fp32(tmodel) 
# outmodel =  tmodel


# traced_model = torch.jit.script(outmodel)
# torch.onnx.export(outmodel, (input_ids,attention_mask,position_ids,past_key_values) ,input_names=input_names ,output_names=output_names , f="./onnx/math-1.5b/model.onnx" ,dynamic_axes={'input_ids':[1],'attention_mask':[1],'position_ids':[1],'last_hidden_state':[1],'past_key_values':[4]},opset_version=18)
# torch.onnx.export(outmodel, (input_ids,attention_mask,position_ids) ,input_names=input_names ,output_names=output_names , f="./onnx/model32.onnx" ,dynamic_axes={'input_ids':[1],'attention_mask':[1],'position_ids':[1],'last_hidden_state':[1]},opset_version = 18)

tmodel.kk=False




generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    streamer = custom_streamer
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