# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from .utils import split_tensor_along_last_dim
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")
# 在 megatron/mpu/mappings.py 之中有对 tensor_model_group 的使用：
#reduce函数利用 _TENSOR_MODEL_PARALLEL_GROUP 进行在组内进行集合通信。

# 对应的后向传播就使用了 all-reduce，反向传播时候，输入是多个GPU上的梯度整体，通过 all-reduce 合并。
def _reduce(input_):
    gd.debuginfo(prj='mt')
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_

# 4.4.4.2 _split_along_last_dim
# _split_along_last_dim 完成了张量切分操作。
def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    gd.debuginfo(prj='mt')

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    gd.debuginfo(prj='mt')
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = get_tensor_model_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset:dim_offset+local_dim_size].contiguous()

    return output

'''
4.4.4 基础函数
4.4.4.1 _gather_along_last_dim
_gather_along_last_dim 是沿着最后一个维度进行拼接。
'''
def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""
    gd.debuginfo(prj='mt')
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""
    gd.debuginfo(prj='mt')
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(),
                                       group=get_tensor_model_parallel_group())

    return output

def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    gd.debuginfo(prj='mt')
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    
    dim_size[0] = dim_size[0] // world_size
   
    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), 
                                           group=get_tensor_model_parallel_group())
    return output

#从 return 的 _CopyToModelParallelRegion函数可以看到，其 forward 就是简单的把输入转移到输出。
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    gd.debuginfo(prj='mt')
    @staticmethod
    def symbolic(graph, input_):
        gd.debuginfo(prj='mt')
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        gd.debuginfo(prj='mt')
        return input_  # 简单的把输入转移到输出，就是对应了前向复制identity

    @staticmethod
    def backward(ctx, grad_output):
        gd.debuginfo(prj='mt')
        return _reduce(grad_output) # 反向传播时候，输入是多个GPU上的梯度整体，通过all-reduce合并


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    gd.debuginfo(prj='mt')
    @staticmethod
    def symbolic(graph, input_):
        gd.debuginfo(prj='mt')
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        gd.debuginfo(prj='mt')
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        gd.debuginfo(prj='mt')
        return grad_output

# 具体 _ScatterToModelParallelRegion 完成了实际业务，具体 _split,_gather 操作在前面都介绍过。
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""
    gd.debuginfo(prj='mt')
    @staticmethod
    def symbolic(graph, input_):
        gd.debuginfo(prj='mt')
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        gd.debuginfo(prj='mt')
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        gd.debuginfo(prj='mt')
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""
    gd.debuginfo(prj='mt')
    @staticmethod
    def symbolic(graph, input_):
        gd.debuginfo(prj='mt')
        return _gather_along_last_dim(input_)
    
    @staticmethod
    def forward(ctx, input_):
        gd.debuginfo(prj='mt')
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        gd.debuginfo(prj='mt')
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""
    gd.debuginfo(prj='mt')
    @staticmethod
    def symbolic(graph, input_):
        gd.debuginfo(prj='mt')
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        gd.debuginfo(prj='mt')
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        gd.debuginfo(prj='mt')
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""
    gd.debuginfo(prj='mt')
    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        gd.debuginfo(prj='mt')
        return _gather_along_first_dim(input_)
    
    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        gd.debuginfo(prj='mt')
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce 
        # scattered and whereas if the computation is duplicated, 
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            gd.debuginfo(prj='mt')
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            gd.debuginfo(prj='mt')
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        gd.debuginfo(prj='mt')
        return _reduce_scatter_along_first_dim(input_)
    
    @staticmethod
    def forward(ctx, input_):
        gd.debuginfo(prj='mt')
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        gd.debuginfo(prj='mt')
        return _gather_along_first_dim(grad_output)


# -----------------
# Helper functions.
# -----------------
'''
4.4.2.1 同步操作
这里主要分析copy_to_tensor_model_parallel_region，其做了前向copy操作，同时构建了后向 all-reduce。
'''
def copy_to_tensor_model_parallel_region(input_):
    gd.debuginfo(prj='mt')
    return _CopyToModelParallelRegion.apply(input_)

'''
5.4.3 g 操作

reduce_from_tensor_model_parallel_region 对应了 g 操作，作用是:

    前向操作是 all-reduce之后得到最终输出.
    反向操作则直接拷贝操作。
50.png
代码为：
'''
def reduce_from_tensor_model_parallel_region(input_):
    gd.debuginfo(prj='mt')
    return _ReduceFromModelParallelRegion.apply(input_)

'''
5.4.2 f 操作

scatter_to_tensor_model_parallel_region 对应了f操作，其作用是：

    前向切分split输入，同时搭建后向的 all-gather 操作。
    后向操作进行 all-gather 操作。
49.png
代码为：
'''
def scatter_to_tensor_model_parallel_region(input_):
    gd.debuginfo(prj='mt')
    return _ScatterToModelParallelRegion.apply(input_)

#4.4.3 g 操作
def gather_from_tensor_model_parallel_region(input_):
    gd.debuginfo(prj='mt')
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    gd.debuginfo(prj='mt')
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    gd.debuginfo(prj='mt')
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad)


def reduce_scatter_to_sequence_parallel_region(input_):
    gd.debuginfo(prj='mt')
    return _ReduceScatterToSequenceParallelRegion.apply(input_)

