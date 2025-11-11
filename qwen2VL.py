from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
model_path = "D:/code/transformer_models/models--Qwen--Qwen2.5VL-3B-Instruct"

from utils import Qwen2VLEmbed
import cv2
# default: Load the model on the available device(s)


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# frame1 = cv2.imread("frame0,0.png")
# frame2 = cv2.imread("frame0,1.png")

# cv2.imshow("test",frame1-frame2)
# cv2.waitKey(0)


def reconstruct_from_patches(patches, grid_thw, patch_size=14, temporal_patch_size=2,pt = 0):
    """
    粗略还原 patches 为图像（仅用于调试）
    """
    B = patches.shape[0]
    for b in range(B):
        Tg, Hg, Wg = grid_thw[b]
        C = 3
        Tp, Ph, Pw = temporal_patch_size, patch_size, patch_size
        D = C * Tp * Ph * Pw
        assert patches.shape[2] == D

        # Reshape back
        recon = patches[b].view(Tg, Hg, Wg, Tp, Ph, Pw, C)
        recon = recon.permute(6, 0, 3, 1, 4, 2, 5).contiguous()  # (C, Tg, Tp, Hg, Ph, Wg, Pw)
        recon = recon.view(C, Tg*Tp, Hg*Ph, Wg*Pw)  # (C, T, H, W)

        # 逆归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1).to("cuda")
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1).to("cuda")
        recon = recon * std + mean
        recon = recon.clamp(0,1) * 255
        recon = recon.byte().permute(1,2,3,0).cpu().numpy()  # (T, H, W, C)

        for t in range(recon.shape[0]):
            frame = cv2.cvtColor(recon[t], cv2.COLOR_RGB2BGR)
            cv2.imshow(f"Recon Frame {t},{pt}.png", frame)
            cv2.imwrite(f"frame{t},{pt}.png",frame)
            cv2.waitKey(0)
    cv2.destroyAllWindows()



# default processer
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            # {
            #     "type": "image",
            #     "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            # },
            {
                "type": "image",
                "image": "D:/code/py/qwen2demo_py/test1.png",
            },
            {
                "type": "video",
                "video": "D:/code/py/qwen2demo_py/test1.mp4",
            },
            {"type": "text", "text": "这段视频里，都有什么东西？请尽可能多的描述它,如果无法描述，请说出无法描述的原因。"},
        ],
    }
]

# Preparation for inference
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
p_inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

embd = Qwen2VLEmbed(tokenizer,device="cuda")
inputs = embd.embed_message_vl(text=[text],images=image_inputs,videos=video_inputs)
shapes = inputs['pixel_values_videos'].shape
# kk =  torch.sum(p_inputs["input_ids"] == 151656) 
kk =  torch.sum(inputs["input_ids"] == 151656) 

# reconstruct_from_patches(p_inputs["pixel_values_videos"].unsqueeze(0),p_inputs["video_grid_thw"])
# reconstruct_from_patches(inputs["pixel_values_videos"],inputs["video_grid_thw"],pt=1)


ti1 = inputs["input_ids"] - p_inputs["input_ids"]
ti2 = inputs["pixel_values_videos"] - p_inputs["pixel_values_videos"]
ti3 = inputs["pixel_values"] - p_inputs["pixel_values"]

# inputs["pixel_values_videos"] = p_inputs["pixel_values_videos"]
ti1m = [ti1.max(),ti1.min()]
ti2m = [ti2.max(),ti2.min()]
ti3m = [ti3.max(),ti3.min()]
tk = torch.nonzero(ti2 > 1).cpu().numpy() 
tk1 = torch.where(ti1>0.5) 

# inputs = inputs.to("cuda")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
# Inference: Generation of the output
# inputs = p_inputs
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]



output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
print(output_text)


generated_ids = model.generate(**p_inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]



output_text = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
print(output_text)

