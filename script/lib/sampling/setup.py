from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as cpp_ext

# Monkey-patch to bypass CUDA version check
original_check = cpp_ext._check_cuda_version
def _bypass_cuda_check(*args, **kwargs):
    pass
cpp_ext._check_cuda_version = _bypass_cuda_check

ROOT = Path(__file__).parent.resolve()
CUDA_DIR = ROOT / 'cuda'

SOURCES = [
    ROOT / 'binding.cu',
    CUDA_DIR / 'common.cu',
    CUDA_DIR / 'sample_uniform.cu',
    CUDA_DIR / 'sample_importance_uniform.cu',
    CUDA_DIR / 'sample_adaptive.cu',
    CUDA_DIR / 'sample_occgrid.cu',
    CUDA_DIR / 'sample_instant_ngp.cu',  # New: instant-ngp method
]

INCLUDE_DIRS = [
    str(ROOT),
    str(CUDA_DIR),
    '/opt/conda/envs/gsnerf/targets/x86_64-linux/include',
    '/opt/conda/envs/gsnerf/include',
    '/opt/conda/envs/gsnerf/lib/python3.9/site-packages/nvidia/cusolver/include',
    '/opt/conda/envs/gsnerf/lib/python3.9/site-packages/nvidia/cublas/include',
]

setup(
    name='sampling',
    packages=['sampling'],
    package_dir={'': '.'},
    setup_requires=['torch'], # Added torch as setup_requires
    ext_modules=[
        CUDAExtension(
            name='sampling._sampling',
            sources=[str(path) for path in SOURCES],
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-lineinfo',
                    '-Xcompiler', '-fPIC',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '--use_fast_math',
                ],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
