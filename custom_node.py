from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import sys
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
import numpy as np
from transformers.cache_utils import Cache, DynamicCache
import torch 
import time
# from vllm.vllm_flash_attn.layers
# sys.path.append("D:/code/py/vllm-main")
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer


tmodel = Qwen2DecoderLayer(Qwen2Config(hidden_size=896,intermediate_size=4864,initializer_range = 0.02,
no_repeat_ngram_size = 0,
num_attention_heads = 14,
num_beam_groups = 1,
num_beams = 1,
num_hidden_layers = 24,
num_key_value_heads = 2,
num_labels = 2),0)

device = "cpu"

df = torch.float32
dl = torch.int64

first_input = {
    'hidden_states':torch.zeros([1,47,896],dtype=df,device=device),
    'attention_mask':torch.zeros([1,1,47,47],dtype=df,device=device),
    'position_ids':torch.zeros([1,47,],dtype=dl,device=device),
    'past_key_value':DynamicCache(),
    'output_attentions':False,
    'use_cache':True,
    }

cache_t = DynamicCache()
k_cache=[]
v_cache=[]

# for i in range(24):
for i in range(1):
    k_cache.append(torch.zeros([1,2,48,64],dtype=df,device=device))
    v_cache.append(torch.zeros([1,2,48,64],dtype=df,device=device))
# k_cache = torch.zeros([24,1,2,337,64],dtype=df)
# v_cache = torch.zeros([24,1,2,337,64],dtype=df)
cache_t.update(k_cache[0],v_cache[0],0)
cache_input = {
    'hidden_states':torch.zeros([1,1,896],dtype=df,device=device),
    'attention_mask':None,
    'position_ids':torch.zeros([1,1,],dtype=dl,device=device),
    'past_key_value':cache_t,
    'output_attentions':False,
    'use_cache':True,
    }


print(torch.xpu.device_count())
print(torch.xpu.is_bf16_supported())
print(torch.cuda.device_count())
print(torch.cuda.is_bf16_supported())
time123 = time.time()
tmodel.to(device)
i = first_input
out_first = tmodel(i['hidden_states'],i['attention_mask'],i['position_ids'],i['past_key_value'],i['output_attentions'],i['use_cache'])
print((time.time()-time123))
i = cache_input
time123 = time.time()
ctime =[]
for k in range(10):
    out_cache = tmodel(i['hidden_states'],i['attention_mask'],i['position_ids'],i['past_key_value'],i['output_attentions'],i['use_cache'])
    ctime .append(time.time()-time123)
    time123 =time.time()
    
print(np.sum(np.array(ctime)))
i = first_input
output_names = ["hiddint"]


# torch.onnx.export(tmodel, (i['hidden_states'],i['attention_mask'],i['position_ids']) ,input_names=["hidden_states","attention_mask","position_ids"] ,output_names=output_names , f="./vllm_demo/model/model32.onnx" ,dynamic_axes={'hidden_states':[1],'attention_mask':[1]},opset_version = 18)
# torch.onnx.export(tmodel,)

print(tmodel)