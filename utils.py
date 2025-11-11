import torch
import torch.nn as nn
import numpy as np
import cv2
import torch
from typing import List, Tuple, Union,Optional,Dict
import os
from tokenizers import Tokenizer
import cv2



def add_detailed_export_logging(model):
    """
    为模型添加详细导出日志，显示每一层的完整路径名、输入输出类型等
    """
    layer_counter = 0
    failed_layer = None

    def hook_fn(module, input, output):
        nonlocal layer_counter, failed_layer
        layer_counter += 1
        # 获取该模块在模型中的完整路径名
        # 注意：我们需要在注册 hook 时传入 name
        pass  # 我们稍后在循环中定义

    # 👇 在这里我们使用 named_modules() 的 name
    hooks = []
    for name, module in model.named_modules():
        if list(module.children()):  # 跳过非叶子模块（如 Sequential, TransformerLayer）
            continue
        if module == model:  # 跳过根模块
            continue

        def make_hook(name):
            def hook(module, input, output):
                nonlocal layer_counter, failed_layer
                layer_counter += 1
                if(layer_counter == 244):
                    print("见擦汗")
                layer_id = f"Layer {layer_counter:03d}"
                # 打印基本信息
                print(f"\n[ONNX-Export] {layer_id} | Path: {name}")
                print(f"              Type: {type(module).__name__}")
                
                # 打印输入信息
                print(f"              Input types:")
                if isinstance(input, tuple):
                    for i, inp in enumerate(input):
                        if isinstance(inp, torch.Tensor):
                            print(f"                [{i}] {inp.dtype}, shape: {inp.shape}")
                        else:
                            print(f"                [{i}] {type(inp).__name__}")
                elif isinstance(input, torch.Tensor):
                    print(f"                {input.dtype}, shape: {input.shape}")
                else:
                    print(f"                {type(input).__name__}")

                # 打印输出信息
                print(f"              Output types:")
                if isinstance(output, tuple):
                    for i, out in enumerate(output):
                        if isinstance(out, torch.Tensor):
                            print(f"                [{i}] {out.dtype}, shape: {out.shape}")
                            # 检查是否为 ComplexDouble
                            if out.dtype == torch.complex128:
                                print(f"🔥 ERROR: {layer_id} | {name} | Output[{i}] is torch.complex128!")
                                import traceback
                                traceback.print_stack()
                                raise RuntimeError(f"Unsupported dtype: torch.complex128 in {name}.output[{i}]")
                        else:
                            print(f"                [{i}] {type(out).__name__}")
                elif isinstance(output, torch.Tensor):
                    print(f"                {output.dtype}, shape: {output.shape}")
                    if output.dtype == torch.complex128:
                        print(f"🔥 ERROR: {layer_id} | {name} | Output is torch.complex128!")
                        import traceback
                        traceback.print_stack()
                        raise RuntimeError(f"Unsupported dtype: torch.complex128 in {name}.output")
                else:
                    print(f"                {type(output).__name__}")
            return hook

        hook = module.register_forward_hook(make_hook(name))
        hooks.append(hook)  # 保留引用，防止被回收

    return hooks  # 可用于后续 remove

def check_complex(model):
    def find_complex_nodes(graph):
        complex_nodes = []
        for node in graph.nodes():
            # 获取输出类型
            for value in node.outputs():
                type_str = str(value.type())
                if "ComplexDouble" in type_str or "complex128" in type_str:
                    complex_nodes.append({
                        "op": node.kind(),        # 操作名，如 aten::fft_rfft
                        "name": str(value),       # 输出变量名
                        "type": type_str,
                        "schema": node.schema() if node.schema() else None
                    })
        return complex_nodes

    # 使用
    graph = model.graph
    complex_nodes = find_complex_nodes(graph)

    for node in complex_nodes:
        print("Found ComplexDouble node:")
        print(f"  Op: {node['op']}")
        print(f"  Output: {node['name']}")
        print(f"  Type: {node['type']}")
        if node['schema']:
            print(f"  Schema: {node['schema']}")
        print("---")



def find_unexpected_tensor_types(traced_model):
    """
    遍历 traced_model 的计算图，并打印所有非 bfloat16 类型的张量。
    """
    unexpected_types = set()
    graph = traced_model.graph  # 获取计算图

    for node in graph.nodes():
        for output in node.outputs():
            if output.type().kind() == 'TensorType':
                # 获取张量的数据类型
                scalar_type = output.type().scalarType()
                # 检查数据类型是否是你希望忽略的 bfloat16
                if scalar_type != 'BFloat16':
                    # 打印节点信息，包括操作符和输出类型
                    print(f"找到异常类型: 操作符={node.kind()}, 输出类型={scalar_type}, 位置={node.sourceRange()}")
                    unexpected_types.add(scalar_type)
    
    if len(unexpected_types) == 0:
        print("未找到除 bfloat16 外的异常张量类型。")
    else:
        print(f"\n所有找到的异常类型: {unexpected_types}")

def create_debug_hook(module_name: str,modeldtype = torch.bfloat16) -> Dict[str, any]:
    """创建一个 hook，用于打印张量信息"""
    def debug_hook(module, input, output):
        show = False
        if isinstance(output, torch.Tensor):
            if output.dtype != modeldtype:
                print(f"  Output Tensor Type: {output.dtype}")
                print(f"  Output Tensor Shape: {output.shape}")
                show  = True
        elif isinstance(output, (list, tuple)):
            for i, out_tensor in enumerate(output):
                if isinstance(out_tensor, torch.Tensor) and out_tensor.dtype != modeldtype:
                    print(f"  Output {i} Tensor Type: {out_tensor.dtype}")
                    print(f"  Output {i} Tensor Shape: {out_tensor.shape}")
                    show  = True
        if show:
            print(f"--- Module: {module_name} ---")
            print("-" * 20)
    return debug_hook

def forward_with_type_check(model, *args, **kwargs):
    def hook_fn(module, input, output, name):
        if isinstance(output, torch.Tensor):
            dtype = output.dtype
            tensor_type = output.type()
            if dtype == torch.complex128:
                print(f"⚠️  FOUND ComplexDouble in module: {name}")
                print(f"   Output type: {tensor_type} (dtype: {dtype})")
                print(f"   Shape: {output.shape}")
                import traceback
                traceback.print_stack()  # 打印调用栈，定位代码位置
                raise RuntimeError(f"ComplexDouble found in {name}")  # 中断执行
        elif isinstance(output, (list, tuple)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and out.dtype == torch.complex128:
                    print(f"⚠️  FOUND ComplexDouble in module: {name}, output[{i}]")
                    print(f"   Output type: {out.type()}, shape: {out.shape}")
                    import traceback
                    traceback.print_stack()
                    raise RuntimeError(f"ComplexDouble in {name}, output[{i}]")

    # 注册前向钩子（hook）到每一层
    hooks = []
    for name, module in model.named_modules():
        hook = module.register_forward_hook(lambda mod, inp, out, name=name: hook_fn(mod, inp, out, name))
        hooks.append(hook)

    try:
        with torch.no_grad():
            model(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        # 移除所有钩子
        for hook in hooks:
            hook.remove()


def check_tensor_tree(obj, name="input"):
    if isinstance(obj, torch.Tensor):
        if obj.dtype == torch.complex128:
            print(f"❌ {name} is ComplexDouble! shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"✅ {name}: {obj.dtype}, shape={obj.shape}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            check_tensor_tree(item, f"{name}[{i}]")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_tensor_tree(v, f"{name}['{k}']")
    else:
        pass




class UnifiedVideoProcessor:
    def __init__(
        self,
        patch_size: int = 14, #每个像素的色彩值块有多大
        merge_size: int = 2, #合并块大小
        temporal_patch_size: int = 2,  # 每个时间 patch 包含多少帧,理解为fs也行
        image_mean: List[float] = [0.485, 0.456, 0.406],
        # image_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        image_std: List[float] = [0.229, 0.224, 0.225],
        # image_std: List[float] = [0.26862954, 0.26130258, 0.27577711],

        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        device = "cpu"
    ):
        
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.image_mean = torch.tensor(image_mean).to(device=device).view(1,1, -1,  1, 1)  # (1,C,1,1,1)
        self.image_std = torch.tensor(image_std).to(device=device).view(1,1, -1,  1, 1)
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.device = device

    def __call__(self, inputs: Union[List[np.array], List[List[np.array]],List[str]]):
        ninput = []
        if isinstance(inputs[0],str):
            for path in inputs:
                image = cv2.imread(path)
                ninput.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            inputs = ninput
        return self.preprocess(inputs)
    def __setdevice__(self,device):
        self.device=device
        self.image_mean=self.image_mean.to(device)
        self.image_std=self.image_std.to(device)

    def preprocess(
        self,
        inputs: Union[List[np.array], List[List[np.array]]]
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        inputs: 
          - 图像: [Image1, Image2, ...] → 每张图视为 T=1
          - 视频: [[Frame1, Frame2, ...], [Video2_Frames], ...]
        """
        # Step 1: 转为 tensor 并堆成 batch
        frames_batch = []
        for item in inputs:
            if isinstance(item, list) or True:
                # 视频：多帧
                frames = [self._pil_to_tensor(frame) for frame in item]
            else:
                # 图像：单帧，T=1
                frames = [self._pil_to_tensor(item)]
            frames_tensor = torch.stack(frames, dim=0)  # (T,C, H, W)
            frames_batch.append(frames_tensor)
        # Stack batch: (B, C, T, H, W)
        pixel_values = torch.stack(frames_batch, dim=0)  # (B, C, T, H, W)

        # Step 2: Rescale to [0,1]
        pixel_values = pixel_values * self.rescale_factor

        # Step 3: Normalize
        if self.do_normalize:
            pixel_values = (pixel_values - self.image_mean) / self.image_std

        # Step 4: Check divisibility
        B, T, C,  H, W = pixel_values.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"H, W must be divisible by patch_size {self.patch_size}")
        if T % self.temporal_patch_size != 0:
            raise ValueError(f"T must be divisible by temporal_patch_size {self.temporal_patch_size}")

        # Compute grid sizes
        grid_h, grid_w = H // self.patch_size, W // self.patch_size
        grid_t = T // self.temporal_patch_size

        if grid_h % self.merge_size != 0 or grid_w % self.merge_size != 0:
            raise ValueError(f"grid_h/grid_w must be divisible by merge_size {self.merge_size}")
        
        patches = pixel_values.view(
            B,
            grid_t,
            self.temporal_patch_size,
            C,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        # Rearrange so merged patches are grouped
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
        # Shape: (B, grid_t, Hg, Wg, Tp, Mh, Mw, C, Ph, Pw)

        # Step 7: Flatten merged patches into tokens
        patches = patches.reshape(
            B,
            grid_t * (grid_h ) * (grid_w ),
            C * self.temporal_patch_size * self.patch_size * self.patch_size,

        )

        # Output grid_thw for each batch item
        grid_thw = [[grid_t, grid_h , grid_w ] for _ in range(B)]
        return patches, torch.tensor(grid_thw).to(patches.device)

    def _pil_to_tensor(self, image: np.array) -> torch.Tensor:
        # Convert to RGB
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')
        return torch.tensor(image).to(dtype=torch.float32).to(self.device)  # (C, H, W)
    

class UnifiedImageProcessor:
    def __init__(
        self,
        patch_size: int = 14, #每个像素的色彩值块有多大
        merge_size: int = 2, #合并块大小
        temporal_patch_size: int = 2,  # 每个时间 patch 包含多少帧,理解为fs也行
        # image_mean: List[float] = [0.485, 0.456, 0.406],
        image_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        # image_std: List[float] = [0.229, 0.224, 0.225],
        image_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        device = "cpu"
    ):
        
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.temporal_patch_size = temporal_patch_size
        self.image_mean = torch.tensor(image_mean).to(device=device).view(1,1, -1, 1, 1)  # (1,C,1,1,1)
        self.image_std = torch.tensor(image_std).to(device=device).view(1,1, -1, 1, 1)
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.device = device

    def __setdevice__(self,device):
        self.device=device
        self.image_mean=self.image_mean.to(device)
        self.image_std=self.image_std.to(device)

    def __call__(self, inputs: Union[List[np.array], List[List[np.array]],List[str]]):
        ninput = []
        if isinstance(inputs[0],str):
            for path in inputs:
                image = cv2.imread(path)
                ninput.append(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            inputs = ninput
        return self.preprocess(inputs)

    def preprocess(
        self,
        inputs: Union[List[np.array], List[List[np.array]]]
    ) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        inputs: 
          - 图像: [Image1, Image2, ...] → 每张图视为 T=1
          - 视频: [[Frame1, Frame2, ...], [Video2_Frames], ...]
        """
        # Step 1: 转为 tensor 并堆成 batch
        frames_batch = []
        for item in inputs:
            if isinstance(item, list):
                # 视频：多帧
                frames = [self._pil_to_tensor(frame) for frame in item]
            else:
                # 图像：单帧，T=1
                frames = [self._pil_to_tensor(item)]
            frames_tensor = torch.stack(frames, dim=0)  # (T，C, H, W)
            frames_batch.append(frames_tensor)

        # Stack batch: (B, C, T, H, W)
        pixel_values = torch.stack(frames_batch, dim=0)  # (B, T, C, H, W)

        # Step 2: Rescale to [0,1]
        pixel_values = pixel_values * self.rescale_factor

        # Step 3: Normalize
        if self.do_normalize:
            pixel_values = (pixel_values - self.image_mean) / self.image_std

        # Step 4: Check divisibility
        B, T, C,  H, W = pixel_values.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"H, W must be divisible by patch_size {self.patch_size}")
        if T % self.temporal_patch_size != 0:
            repeats = pixel_values[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
            pixel_values = torch.cat([pixel_values, repeats], dim=1)
            T = pixel_values.shape[1]
        # Compute grid sizes
     
        grid_h, grid_w = H // self.patch_size, W // self.patch_size
        grid_t = T // self.temporal_patch_size

        if grid_h % self.merge_size != 0 or grid_w % self.merge_size != 0:
            raise ValueError(f"grid_h/grid_w must be divisible by merge_size {self.merge_size}")

        # Step 5: Unfold spatial dimensions → (B, C, T, grid_h, P, grid_w, P)
        # patches = pixel_values.unfold(3, self.patch_size, self.patch_size).unfold(4, self.patch_size, self.patch_size)
        # Shape: (B, C, T, grid_h, grid_w, P, P)

        # Reshape to (B, T, C, grid_h, grid_w, P, P)
        # 这俩个步骤未找到，我试着先注释了吧
        # patches = patches.permute(0, 2, 1, 3, 4, 5, 6).contiguous()

        # Step 6: Patch Merging via view + permute
        pixel_values = pixel_values.view(
            B,
            grid_t,
            self.temporal_patch_size,
            C,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        # Rearrange so merged patches are grouped
        pixel_values = pixel_values.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(
            B,grid_t * (grid_h) * (grid_w), C * self.temporal_patch_size   * self.patch_size * self.patch_size
        )
        # Shape: (B, grid_t, Hg, Wg, Tp, Mh, Mw, C, Ph, Pw)

        # Step 7: Flatten merged patches into tokens
      
        # Output grid_thw for each batch item
        grid_thw = [[grid_t, grid_h , grid_w ] for _ in range(B)]

        return pixel_values, grid_thw 

    def _pil_to_tensor(self, image: np.array) -> torch.Tensor:
        # Convert to RGB
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')
        img_np = np.array(image).astype(np.float32)  # (H, W, C)
        return torch.from_numpy(img_np).to(self.device).permute(2, 0, 1)  # (C, H, W)

class Qwen2VLEmbed:
    def __init__(self,tokenizer:Tokenizer,video_fs = 2,device = "cpu",image_processor:UnifiedImageProcessor = None,video_processor:UnifiedVideoProcessor = None):
        self.tokenizer = tokenizer
        if image_processor == None:
            image_processor = UnifiedImageProcessor(device=device)
        if video_processor == None:
            video_processor = UnifiedVideoProcessor(device=device)

        self.image_processor = image_processor
        self.video_processor = video_processor

        self.device = device
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )


    def embed_message_vl(
        self,
        text,
        text_padding:bool = True,
        images: Union[List[np.array], List[List[np.array]],List[str]] = [],
        videos: Union[List[np.array], List[List[np.array]],List[str]] = [],
        return_mm_token_type_ids :bool= False, 
    ) :
        image_inputs = videos_inputs =video_grid_thw = image_grid_thw = None
        second_per_grid_ts = 0
        if images is not None:
            image_inputs, image_grid_thw = self.image_processor(inputs=images)
            image_grid_thw = torch.tensor(image_grid_thw).to(self.device)
            image_inputs =image_inputs.to(self.device)
        if videos is not None:
            fps = self.video_processor.temporal_patch_size

            if isinstance(videos[0],str):
                video_list = []
                for path in videos:
                    video_list.append(self.extract_video_frames_at_fps(path,self.image_processor.temporal_patch_size))
                videos = np.array([video_list])

            videos_inputs,video_grid_thw = self.video_processor(inputs=videos)
        

            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            # videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)



        text_inputs = self.tokenizer(text, padding = text_padding)
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])
        text_inputs["input_ids"] = np.array(text_inputs["input_ids"],dtype=np.int64)
        text_inputs["attention_mask"] = np.array(text_inputs["attention_mask"],dtype= np.int64)
        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return {
            "input_ids":torch.from_numpy(text_inputs["input_ids"]).to(device=self.device),
            "attention_mask":torch.from_numpy(text_inputs["attention_mask"]).to(device=self.device),
            "pixel_values":image_inputs,
            "image_grid_thw":image_grid_thw,
            "pixel_values_videos":videos_inputs,
            "video_grid_thw":video_grid_thw,
            "second_per_grid_ts":second_per_grid_ts,
            }
        # return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _check_special_mm_tokens(self, text: list[str], text_inputs, modalities: list[str]):
        """
        Checks that number of special tokens in text and processed text is same. The count can be different
        if tokenized text was truncated, leading to issues in model code.
        """
        for modality in modalities:
            token_str = getattr(self, f"{modality}_token")
            token_id = getattr(self, f"{modality}_token_id")
            ids_count = [list(ids).count(token_id) for ids in text_inputs["input_ids"]]
            text_count = [sample.count(token_str) for sample in text]

            if ids_count != text_count:
                raise ValueError(
                    f"Mismatch in `{modality}` token count between text and `input_ids`. Got ids={ids_count} and text={text_count}. "
                    "Likely due to `truncation='max_length'`. Please disable truncation or increase `max_length`."
                )


    def extract_video_frames_at_fps(
        self,
        video_path: str,
        target_fps: float = 2.0
    ) -> np.ndarray:
        """
        从视频文件中以指定 FPS 提取帧，输出为 (T, H, W, C) 的 RGB numpy 数组
        
        Args:
            video_path: 视频文件路径（.mp4, .avi 等）
            target_fps: 目标提取帧率，如 2.0 表示每秒取 2 帧
        
        Returns:
            frames: np.ndarray of shape (T, H, W, C), dtype=uint8, RGB
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")
        
        # 获取视频原始 FPS 和总帧数
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / original_fps
        print(f"视频信息: {original_fps:.2f} FPS, {duration:.2f} 秒")
        
        frames = []
        frame_interval = 1.0 / target_fps  # 每帧间隔多少秒
        current_time = 0.0
        
        while current_time < duration:
            # 设置视频读取位置到指定时间（秒）
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # OpenCV 默认是 BGR → 转为 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.transpose(frame_rgb,[1,2,0])
            frames.append(frame_rgb)
            
            current_time += frame_interval
        
        cap.release()
        
        if not frames:
            raise ValueError("未能提取到任何帧，请检查视频文件")
        
        return np.stack(frames, axis=0)  # (T, H, W, C)