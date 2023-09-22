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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .initialize import get_tensor_model_parallel_group
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from megatron import get_args


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}


def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)

# 3.2.2 初始化权重
# 以下代码实现了初始化权重。
def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None

'''
0x05 Embedding
我们接下来看看 embedding。
为了让内存做到均衡配置，对embedding也会按照vocab维度来做shard操作，最终把分区放到多个GPU之上。
这样每个卡上都有嵌入表的一部分。
'''
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        # 得到分区的起始，终止位置
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        # 得到分区内嵌入数目
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            # 对权重进行分区
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            # 对权重进行分区
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    '''
    因为每一个GPU只是获得了总体嵌入的一部分，所以对于每个worker来说，可能有一个输入找不到嵌入，
    因此需要对embedding最终输出做一个 all-reduce操作，这样可以得到完整embedding。
    '''
    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            # input_mask 意思是单词不在本worker的 embedding 分区范围内，所以设置为0
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class LinearWithGradAccumulationAndAsyncAllreduce(torch.autograd.Function):
    """
    Linear layer execution with asynchronous all-reduce and gradient accumulation
    fusion in backprop.
    """
    @staticmethod
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
                async_grad_allreduce):
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_input = grad_output.matmul(weight)

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                       grad_output.shape[2])
        input = input.view(input.shape[0] * input.shape[1], input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True)
            # Delay the start of weight gradient computation shortly (3us) to have
            # all-reduce scheduled first and have GPU resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
        if ctx.gradient_accumulation_fusion:
            import fused_dense_cuda
            fused_dense_cuda.wgrad_gemm_accum_fp32(input, grad_output, weight.main_grad)
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None

'''
0x03 ColumnParallelLinear
ColumnParallelLinear 就是按列进行切分，也就是纵刀流。注意，这里说的是对权重进行列切分。就是：
    Y=XA=X[A1,A2]=[XA1,XA2]具体切分如下：

37.png

3.1 定义
因为 Python 语言特性，这里有用的只是注释，从注释中可以看出来，对于 $ Y = XA + b ，A被以如下方式进行并行化：
A = [A_1, ..., A_p] $
 '''
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """
    # 3.2
    # 初始化
    # 初始化代码之中主要是用切分的信息来初始化权重。
    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size() # 获得本tensor并行组的world size
        self.output_size_per_partition = divide(output_size, world_size) # 获得本子模型应输出size
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            # 用切分的size初始化权重
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu( # 初始化权重
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            # 用切分的size初始化权重
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,  # 初始化权重
                                          partition_dim=0, stride=stride)

        if bias:
            if args.use_cpu_initialization:
                # 用切分的size初始化权重
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                # 用切分的size初始化权重
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.async_tensor_model_parallel_allreduce = (
                args.async_tensor_model_parallel_allreduce and
                world_size > 1)
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

    '''
    3.4 代码实现
我们接下来结合代码来分析。

3.3.1 ColumnParallelLinear
ColumnParallelLinear 的 forward 代码之中，主要是实施了 f 和 g 的forward操作，同时把 f 和 g 的backward 操作搭建起来，
具体如下：

    如果配置了异步操作，则使用 ColumnParallelLinearWithAsyncAllreduce 完成 f 运算符的功能，
    这一个函数包括了identity 操作，矩阵乘法，搭建后向传播操作。
    
    如果是同步操作，则：
    
        使用 copy_to_tensor_model_parallel_region 完成前向传播 identity 操作，建立反向传播all-reduce，就是图中f的backward。
        identity 操作 就是把输入 X 完整的拷贝到多个GPU之上，类似 X 通过 f 的前向操作，变成了 [X, X, ..., X]。
        
        使用 linear 对 [X, X, ..., X] 和 权重 A 完成矩阵乘法操作。
        
    如果gather_output为True，则在前向传播时候把 Yi做all-gather，
    因为反向传播时需要把完整梯度scatter到对应GPU之上，所以要搭建对于的split操作。
    MLP实现之中，此处设置为 False，这样每个GPU输出的是自己partition 的 4h/p，直接传送给下一个线性层。
        
    '''
    def forward(self, input_):
        # 如果选择忽略bias，就会设置为None，后续就不用处理了
        bias = self.bias if not self.skip_bias_add else None

        # 下面主要是图中的 f 操作  yknote代码有不同
        if self.async_tensor_model_parallel_allreduce:
            # 建立反向传播时候的异步all-reduce
            input_parallel = input_
        else:
            # Set up backprop all-reduce.
            # 建立反向传播all-reduce，就是图中f的backward
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        ''' 
        yknote代码有不同
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, bias) # 矩阵乘法操作
        '''
        # Matrix multiply. # 矩阵乘法操作
        output_parallel = LinearWithGradAccumulationAndAsyncAllreduce.apply(
            input_parallel, self.weight, bias, self.gradient_accumulation_fusion,
            self.async_tensor_model_parallel_allreduce)

        # 下面就是图中的 g 操作
        if self.gather_output: # 是否需要聚合操作
            # All-gather across the partitions.
            # 聚合输出，就是图中g的forward
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        # 如果不忽略bias，还得传出去
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

'''
0x04 RowParallelLinear
RowParallelLinear 这里是按照行进行切分，就是横刀流，注意这里是对权重A实施行切分。
比如公式为 Y = XA，X是输入，A是权重，Y是输出，行切分就是针对A的第一个维度进行切分，这里 X1最后一个维度等于 A1第一个维度。
    XA=[X1,X2] [A1A2]T = X1A1+ X2A2 = Y1+Y2 =Y  
    yknote [A1A2]T 表示列，T是补上加的
具体如下： 43.png

4.1 定义
定义之中只有注释有用，可以看出来如何切分。
'''
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
    """
    '''
    4.2 初始化
    和列切分类似，初始化之中主要是获取每个权重分区的大小，然后据此切分权重。
    '''
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)  # 获取每个权重分区的大小
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=args.params_dtype))
            # 切分权重
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            # 切分权重
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

    '''
    4.4 代码实现
    我们接下来看看代码如何实现。
    
    4.4.1 RowParallelLinear
    RowParallelLinear 的 forward 代码之中，主要是实施了 f 和 g 的forward操作，
    同时把 f 和 g 的backward 操作搭建起来，
    具体如下：
    '''
    def forward(self, input_):
        # 这里，输入的张量已经被分割到每个GPU，输出张量是all-reduce之后的整体
        # Set up backprop all-reduce.
        if self.input_is_parallel:  # 是否已经是split的输入
            # Transformer's MLP 到达这里，因为已经split，所以直接就接了输入，不会scatter
            input_parallel = input_
        else:
            # 独立 row parallel 线性层到这里，会进行前向切分和后向拼接
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        # 比如 X_i 和 A_i 进行乘法操作
        output_parallel = LinearWithGradAccumulationAndAsyncAllreduce.apply(
            input_parallel, self.weight, None,
            self.gradient_accumulation_fusion, None)

        # All-reduce across all the partitions.
        # 进行前向all-reduce操作，这样每个GPU之上都是完整的最新结果，同时搭建了后向的identity操作。
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

