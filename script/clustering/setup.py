from pathlib import Path
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as cpp_ext

# Set CUDA_HOME to the actual CUDA installation
os.environ['CUDA_HOME'] = '/usr/local/cuda-11.8'

# Monkey-patch to bypass CUDA version check
original_check = cpp_ext._check_cuda_version
def _bypass_cuda_check(*args, **kwargs):
    pass
cpp_ext._check_cuda_version = _bypass_cuda_check

ROOT = Path(__file__).parent.resolve()
C_SRC = ROOT / "csrc"

W2_DIR = C_SRC / "w2"

SOURCES = [
    C_SRC / "binding.cpp",
    W2_DIR / "nearest.cu",
    W2_DIR / "merge.cu",
    W2_DIR / "fps.cu",
    W2_DIR / "kmeans.cu",
    W2_DIR / "aggregate.cu",
    W2_DIR / "clustering.cu",
]

INCLUDE_DIRS = [
    str(C_SRC),
    str(W2_DIR),
    '/usr/local/cuda-11.8/include',
    '/opt/conda/envs/gsnerf/targets/x86_64-linux/include',
    '/opt/conda/envs/gsnerf/include',
    '/opt/conda/envs/gsnerf/lib/python3.9/site-packages/nvidia/cusolver/include',
    '/opt/conda/envs/gsnerf/lib/python3.9/site-packages/nvidia/cublas/include',
]

setup(
    name="clustering",
    packages=['clustering'],
    package_dir={'': '.'},
    setup_requires=['torch'], # Added torch as setup_requires
    ext_modules=[
        CUDAExtension(
            name="clustering._clustering",
            sources=[str(path) for path in SOURCES],
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    # 결정적 연산을 위한 옵션들
                    '--fmad=false',  # FMA 비활성화 (결정성 보장)
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '-D__NV_LEGACY_LAUNCH',
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
