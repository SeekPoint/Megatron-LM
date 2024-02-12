# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import pathlib
import subprocess

from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with diferent order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicity in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""
from pydebug import gd, infoTensor
# gd.debuginfo(prj="mt")

def load(args):
    gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)

    bare_metal_minor = 6

    gd.debuginfo(prj="mt",
                 info=f'args={args}, '
                      f'bare_metal_major={bare_metal_major}, '
                      f'bare_metal_minor={bare_metal_minor}')

    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')
        # cc_flag.append('arch=compute_75,code=sm_75')
        if int(bare_metal_minor) >= 7:
            cc_flag.append('-gencode')
            cc_flag.append('arch=compute_90,code=sm_90')
    '''
[1/2]
/usr/local/cuda/bin/nvcc  -DTORCH_EXTENSION_NAME=scaled_upper_triang_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /usr/local/lib/python3.9/site-packages/torch/include -isystem /usr/local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/lib/python3.9/site-packages/torch/include/TH -isystem /usr/local/lib/python3.9/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/local/include/python3.9 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 --compiler-options '-fPIC' -O3 -gencode arch=compute_70,code=sm_70 --use_fast_math -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 arch=compute_75,code=sm_75 -std=c++14 -c /share/yk_repo/Megatron-LM/tag_23.06/megatron/fused_kernels/scaled_upper_triang_masked_softmax_cuda.cu -o scaled_upper_triang_masked_softmax_cuda.cuda.o
FAILED: scaled_upper_triang_masked_softmax_cuda.cuda.o
/usr/local/cuda/bin/nvcc  -DTORCH_EXTENSION_NAME=scaled_upper_triang_masked_softmax_cuda -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /usr/local/lib/python3.9/site-packages/torch/include -isystem /usr/local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/lib/python3.9/site-packages/torch/include/TH -isystem /usr/local/lib/python3.9/site-packages/torch/include/THC -isystem /usr/local/cuda/include -isystem /usr/local/include/python3.9 -D_GLIBCXX_USE_CXX11_ABI=0 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr -gencode=arch=compute_75,code=compute_75 -gencode=arch=compute_75,code=sm_75 --compiler-options '-fPIC' -O3 -gencode arch=compute_70,code=sm_70 --use_fast_math -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda -gencode arch=compute_80,code=sm_80 arch=compute_75,code=sm_75 -std=c++14 -c /share/yk_repo/Megatron-LM/tag_23.06/megatron/fused_kernels/scaled_upper_triang_masked_softmax_cuda.cu -o scaled_upper_triang_masked_softmax_cuda.cuda.o
nvcc fatal   : A single input file is required for a non-link phase when an outputfile is specified
ninja: build stopped: subcommand failed.    

root@9daa04e3405e:/share/yk_repo/Megatron-LM/tag_23.06/apex# pip install --global-option="--cpp_ext" --global-option="--cuda_ext" .
    '''

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'
    _create_build_dir(buildpath)

    gd.debuginfo(prj="mt", info=f'srcpath={srcpath}, buildpath={buildpath}')

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
        gd.debuginfo(prj="mt")
        return cpp_extension.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '-gencode', 'arch=compute_70,code=sm_70',
                               '--use_fast_math'] + extra_cuda_flags + cc_flag,
            verbose=(args.rank == 0)
        )

    # ==============
    # Fused softmax.
    # ==============

    if args.masked_softmax_fusion:
        gd.debuginfo(prj="mt")
        extra_cuda_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                            '-U__CUDA_NO_HALF_CONVERSIONS__',
                            '--expt-relaxed-constexpr',
                            '--expt-extended-lambda']

        # Upper triangular softmax.
        sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
                 srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu']

        scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_upper_triang_masked_softmax_cuda",
            sources, extra_cuda_flags)

        gd.debuginfo(prj="mt",
                     info=f'scaled_upper_triang_masked_softmax_cuda={scaled_upper_triang_masked_softmax_cuda}')

        # Masked softmax.
        sources=[srcpath / 'scaled_masked_softmax.cpp',
                 srcpath / 'scaled_masked_softmax_cuda.cu']

        scaled_masked_softmax_cuda = _cpp_extention_load_helper(
            "scaled_masked_softmax_cuda", sources, extra_cuda_flags)

        gd.debuginfo(prj="mt",
                     info=f'scaled_masked_softmax_cuda={scaled_masked_softmax_cuda}')

        # Softmax
        sources=[srcpath / 'scaled_softmax.cpp',
                 srcpath / 'scaled_softmax_cuda.cu']

        scaled_softmax_cuda = _cpp_extention_load_helper(
            "scaled_softmax_cuda", sources, extra_cuda_flags)

        gd.debuginfo(prj="mt",
                     info=f'scaled_softmax_cuda={scaled_softmax_cuda}')

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

def _get_cuda_bare_metal_version(cuda_dir):

    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    gd.debuginfo(prj="mt", info=f'raw_output={raw_output}')
    gd.debuginfo(prj="mt", info=f'bare_metal_major={bare_metal_major}, bare_metal_minor={bare_metal_minor}')

    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            gd.debuginfo(prj="mt", info=f"Creation of the build directory {buildpath} failed")
