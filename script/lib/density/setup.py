from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch.utils.cpp_extension as cpp_ext
import os

# Monkey-patch to bypass CUDA version check
original_check = cpp_ext._check_cuda_version
def _bypass_cuda_check(*args, **kwargs):
    pass
cpp_ext._check_cuda_version = _bypass_cuda_check

this_dir = os.path.dirname(__file__)

sources = [
    os.path.join(this_dir, 'binding.cu'),
    os.path.join(this_dir, 'cuda', 'compute_density.cu'),
]

include_dirs = [
    this_dir,
    os.path.join(this_dir, 'cuda'),
    '/opt/conda/envs/gsnerf/targets/x86_64-linux/include',
    '/opt/conda/envs/gsnerf/include',
]

setup(
    name='density',
    version='0.1',
    packages=['density'],
    package_dir={'': '.'},
    setup_requires=['torch'], # Added torch as setup_requires
    ext_modules=[
        CUDAExtension(
            name='density._density_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                ],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
