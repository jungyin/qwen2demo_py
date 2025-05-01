from transformers import AutoModelForCausalLM, AutoTokenizer

import transformers

import time;

import numpy as np
class CustomStreamer(transformers.generation.streamers.BaseStreamer):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.start = True
        self.time123 = time.time()
        self.cachetime = []
    def put(self, value):
        ctime = time.time()-self.time123
        self.cachetime .append(ctime)
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
        print("每秒tokens:",oneValue,"总耗时",np.sum(np.array(self.cachetime)))
        pass
