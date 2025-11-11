import os
import time
from torchprofile import profile_macs

import numpy as np

import torch 

import torch.nn.functional as F

from utils import add_detailed_export_logging
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import transformers.masking_utils as m_utils

# 这个方法不生效，需要手动点过去，吧这个值赋为false
m_utils._is_torch_greater_or_equal_than_2_6= False

os.environ.setdefault("ONNX_PYTORCH_DEBUG", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "D:/code/transformer_models/models--Qwen--Qwen2.5VL-3B-Instruct"
outtype = torch.bfloat16
# fpath = "./onnx/math-1.5b/model.onnx"
# fpath = "./onnx/qwen2_3b/model.onnx"
fpath = "./onnx/qwen2_0.5b_new/model.onnx"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu",
    local_files_only=True
)

# 使用 torch.fx.symbolic_trace 将模型转换为 FX 图表示
# symbolic_traced = torch.fx.symbolic_trace(model.model)

# 获取所有节点
# nodes = list(symbolic_traced.graph.nodes)

print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            # {
            #     "type": "video",
            #     "video": "D:/code/py/qwen2demo_py/test1.mp4",
            # },
            {"type": "text", "text": "Describe this image and this video,use chinese."},
        ],
    }
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
processor = AutoProcessor.from_pretrained(model_name)

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)

model_inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# messages = "这是一个测试句子。"

input_ids = model_inputs.data['input_ids']
attention_mask = model_inputs.data['attention_mask']

pixel_values = model_inputs.data['pixel_values']
image_grid_thw = model_inputs.data['image_grid_thw']
pixel_values_videos = model_inputs.data['pixel_values_videos']
video_grid_thw = model_inputs.data['video_grid_thw']
second_per_grid_ts = model_inputs.data['second_per_grid_ts']

position_ids = attention_mask.long().cumsum(-1) - 1
position_ids.masked_fill_(attention_mask == 0, 1)
past_key_values =torch.from_numpy( build_cache_random(model)).to(position_ids.device).to(outtype)
# input = mypre(model_inputs['input_ids'],model_inputs['attention_mask'])


input_names = [
            # 输入部分，这个部分和传统llm通用
               "input_ids",
               "attention_mask",
               "position_ids",
               "past_key_values",
               "inputs_embeds",
            # 固定输入部分
               "use_cache",
               "output_attentions",
               "output_hidden_states",
               "return_dict",
            #    放置了图片/视频信息的部分
               "pixel_values",#[patch_number,1176]
               "pixel_values_videos",#[patch_number,1176]
               "image_grid_thw",# [t,h,w]
               "video_grid_thw",# [t,h,w]
            # 这几个待会看看，不知道为什么在推理阶段，似乎就停止了工作
               "rope_deltas", # false
               "cache_position",  # false
               "second_per_grid_ts" #[1]
               ]

# 固化输入部分
constant_inputs={
    "use_cache": torch.tensor(True),
    "output_attentions": torch.tensor(False),
    "output_hidden_states": torch.tensor(False),
    "return_dict": torch.tensor(False),

    "rope_deltas": torch.tensor(False),
    "cache_position": torch.tensor(False),
}

# 动态长度部分
dynamic_axes = {
    'input_ids':[1],
    'attention_mask':[1],
    'position_ids':[1],
    'inputs_embeds':[1],
    'past_key_values':[4],

    'pixel_values':[0],
    'pixel_values_videos':[0],
    'image_grid_thw':[0,1,2],
    'video_grid_thw':[0,1,2]
}


model.base_model.kk = False

tmodel = model
tmodel.kk = True

#cache_list = numpy.array()

# outmodel =  convert_bf16_fp16_to_fp32(tmodel) 
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

outtest = model(input_ids,attention_mask,position_ids,past_key_values,,,,pixel_values,pixel_values_videos,image_grid_thw,video_grid_thw,,,second_per_grid_ts)

torch.onnx.export(outmodel, (input_ids,attention_mask,position_ids,past_key_values) ,input_names=input_names ,output_names=output_names ,verbose= True, f=fpath ,dynamic_axes=dynamic_axes,opset_version=18)


tmodel.kk=False

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
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