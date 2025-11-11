# my_cuda_project/setup.py

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys

# 设置 C++ 标准为 C++20 (根据之前的讨论，这很关键)
# 对于 MSVC (Windows): /std:c++20
# 对于 GCC/Clang (Linux/macOS): -std=c++20
cxx_flags = []
if sys.platform == "win32":
    # Windows MSVC 编译器参数
    cxx_flags.append("/std:c++20")
    # /Zc:__cplusplus 确保 MSVC 的 __cplusplus 宏正确报告 C++ 标准
    # 这对使用 C++17/20 特性至关重要
    cxx_flags.append("/Zc:__cplusplus") 
else:
    # Linux/macOS GCC/Clang 编译器参数
    cxx_flags.append("-std=c++20")

# NVCC 编译器参数 (CUDA 编译器)
# 同样确保 NVCC 也使用 C++20 标准
nvcc_flags = ["-std=c++20"] 

# 可以添加其他 NVCC flags，例如针对特定 CUDA 架构
# nvcc_flags.append("-gencode=arch=compute_86,code=sm_86") # 例如 RTX 30系列

setup(
    name='test_pyc', # 包的名称
    version='0.1.0',
    packages=find_packages(), # 自动查找包
    ext_modules=[
        CUDAExtension(
            name='my_module._C', # 最终生成的扩展模块名称 (Python 中导入时会是 my_module._C)
            sources=['test_cu/cuda_kernel.cu'], # 你的 CUDA C++ 源文件
            # 传递 C++ 和 CUDA 编译器的额外参数
            extra_compile_args={
                'cxx': cxx_flags,  # C++ 编译器 (例如 cl.exe) 的参数
                'nvcc': nvcc_flags, # CUDA 编译器 (nvcc.exe) 的参数
            },
            # PyTorch C++ 扩展默认会包含 PyTorch 的头文件路径
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension # 使用 PyTorch 的 BuildExtension
    }
)