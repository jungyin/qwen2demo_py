from transformers import AutoTokenizer, VisionEncoderDecoderModel, AutoImageProcessor
from PIL import Image
import requests
import cv2
import numpy as np
import torch

p = "./source/latex"
tokenizer = AutoTokenizer.from_pretrained(p, max_len=296)
feature_extractor = AutoImageProcessor.from_pretrained(p, local_files_only=True)

model = VisionEncoderDecoderModel.from_pretrained(p)




model.base_model.kk = True
model.base_model.decoder.kk = True
model.base_model.encoder.kk = True

input_names = ["pixel_values"]
output_names = ["encoder_out"]
# 开始导出onnx encoder层模型
pixel_values = torch.from_numpy(np.random.rand(1,3,400,500)).to(torch.float32)
# torch.onnx.export(model.base_model.encoder,args=(pixel_values),f="./onnx/ocr_encoder.onnx",input_names=input_names,output_names=output_names)

# 导出decoder层模型
input_names = ["input_ids","attention_mask","encoder_hidden_states"]
output_names = ["prediction_scores"]
input_ids = torch.from_numpy(np.array([[0]])).to(torch.int64)
attention_mask = torch.from_numpy(np.array([[1]])).to(torch.int64)
encoder_hidden_states = torch.from_numpy(np.random.rand(1,208,768)).to(torch.float32)
# torch.onnx.export(model.base_model.decoder,args=(input_ids,attention_mask,encoder_hidden_states),f="./onnx/ocr_decoder.onnx",input_names=input_names,output_names=output_names,dynamic_axes={'input_ids':[1],'attention_mask':[1]})


model.base_model.decoder.kk = False
model.base_model.encoder.kk = False

imgen = Image.open("test.png")
#imgzh = Image.open(requests.get('https://cdn-uploads.huggingface.co/production/uploads/62dbaade36292040577d2d4f/m-oVg8dsQbQZ1fDWbwKtO.png', stream=True).raw)
imgen_np = np.array(imgen).astype(np.float32)
inputs = feature_extractor(imgen, return_tensors="pt")
input_value = inputs.pixel_values

frame = np.array(cv2.imread("test.png"))

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame,[500,400])
frame = frame * feature_extractor.rescale_factor

frame = (frame - feature_extractor.image_mean) / feature_extractor.image_std 

frame = frame.astype(np.float32)

inputtest = frame


inputtest = np.expand_dims(np.transpose(inputtest, (2, 0, 1)),0)

it = inputtest - input_value.numpy()
mit = np.max(it)

mout = model.generate(torch.from_numpy(inputtest).to(input_value.device))
moutstr = tokenizer.decode(mout[0])

print(moutstr.replace('\\[','\\begin{align*}').replace('\\]','\\end{align*}'))
