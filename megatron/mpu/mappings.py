# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .initialize import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from .utils import split_tensor_along_last_dim

#对应的后向传播就使用了All-reduce，反向传播时候，输入是多个GPU上的梯度整体，通过all-reduce合并。
def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size() # 获取本tensor进程组的world size
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank() # 获取自己的rank
    output = input_list[rank].contiguous() # 获取切分后，自己对应的rank

    return output

'''
3.3.4 基础函数
我们接下来看看上面用到的一些基础函数。

3.3.4.1 gather
_gather 是沿着最后一个维度进行拼接。
'''
def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank() # 获得本worker在tensor并行之中的rank

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    # 在本 tensor 进程组之间进行 all-gather操作
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

#我们还是需要看看 _CopyToModelParallelRegion。
# 可以看到，其 forward 就是简单的把输入转移到输出，就是对应了前向复制identity。
class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_  # 简单的把输入转移到输出，就是对应了前向复制identity

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)   # 反向传播时候，输入是多个GPU上的梯度整体，通过all-reduce合并


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_) # 前面有介绍

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output #就是indentity 操作，直接把输入拷贝到两个GPU之上

#具体 _ScatterToModelParallelRegion 完成了实际业务，具体 _split, _gather 操作在前面都介绍过。
class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------
'''
3.3.2 f 操作
F 操作是对输入进行初步处理，具体是：

    前向传播时候直接拷贝。
    后向传播做all-reduce。
41.png

3.3.2.1 同步操作
这里我们主要分析 copy_to_tensor_model_parallel_region，其做了前向copy操作，同时构建了后向 all-reduce。
'''
def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)

'''
4.4.2 g 操作
reduce_from_tensor_model_parallel_region 对应了 g 操作，作用是:
    前向操作是 all-reduce之后得到最终输出.
    反向操作则直接拷贝操作。
48.png
代码为：
'''
def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)

'''
4.4.1 f 操作
scatter_to_tensor_model_parallel_region 对应了f操作，其作用是：

    前向切分split输入，同时搭建后向的 all-gather 操作。
    后向操作进行 all-gather 操作。
47.png

代码为：
'''
def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)

'''
3.3.3 g 操作
以下对应了图之中的 g 操作。G操作是最终生成输出Y，逻辑是：

    前向传播时候做 all-gather；
    后向传播需要执行 split，把梯度scatter到不同GPU之上。
42.png
'''
def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)
