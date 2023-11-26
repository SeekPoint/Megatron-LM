# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import math
import os
from typing import Optional
import warnings
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from torch.cuda.amp import custom_fwd, custom_bwd

from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_global_memory_buffer,
)
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)

from .random import get_cuda_rng_tracker
from .utils import (
    divide,
    split_tensor_along_last_dim,
    VocabUtility,
)

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}

def param_is_not_tensor_parallel_duplicate(param):
    gd.debuginfo(prj='mt')
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    gd.debuginfo(prj='mt')
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        gd.debuginfo(prj='mt')
        if not hasattr(tensor, attribute):
            gd.debuginfo(prj='mt')
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        gd.debuginfo(prj='mt')
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        gd.debuginfo(prj='mt')
        if hasattr(source_tensor, attribute):
            gd.debuginfo(prj='mt')
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        gd.debuginfo(prj='mt')
        maybe_copy(attribute)

'''
4.2.2 初始化权重
下面代码实现了模型权重的初始化。
'''
def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""
    gd.debuginfo(prj='mt')
    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        gd.debuginfo(prj='mt')
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False,
                                  *, params_dtype=torch.float32):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""
    gd.debuginfo(prj='mt')
    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

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

# 6 Embedding
# 为了让内存做到均衡配置，对Embedding表也会按照 vocab维度来做shard操作，最终把分区放到多个 GPU 之上。
# 这样每个卡上都有Embedding表的一部分。
class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        params_dtype
        use_cpu_initialization
        perform_initialization
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                 init_method=init.xavier_normal_,
                 params_dtype: torch.dtype=torch.float32,
                 use_cpu_initialization: bool=False,
                 perform_initialization: bool=True):
        gd.debuginfo(prj='mt', info=f"C:{self.__class__.__name__}")
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
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # Allocate weights and initialize.
        if use_cpu_initialization:
            gd.debuginfo(prj='mt', info=f"C:{self.__class__.__name__}")
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight, self.num_embeddings, self.embedding_dim,
                    self.num_embeddings_per_partition, 0, init_method,
                    params_dtype=params_dtype)
        else:
            gd.debuginfo(prj='mt', info=f"C:{self.__class__.__name__}")
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=1)
    '''
    因为每一个 GPU 只是获得了总体 embedding 的一部分，
    所以对于每个 worker 来说，可能有一个输入找不到 embedding，
    因此需要对embedding最终输出做一个all-reduce操作，这样可以得到完整embedding。
    '''
    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            gd.debuginfo(prj='mt', info=f"C:{self.__class__.__name__}")
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            gd.debuginfo(prj='mt', info=f"C:{self.__class__.__name__}")
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

# 这定义了一个名为LinearWithGradAccumulationAndAsyncCommunication的类，
# 该类继承自torch.autograd.Function。
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    # 使用两个装饰器标记forward方法。其中@staticmethod表示这是一个静态方法，
    # 而@custom_fwd是一个自定义装饰器，用于特定的前向传播操作。
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
    ):
        gd.debuginfo(prj='mt')
        # 使用上下文对象ctx保存输入和权重，以便在后向传播中使用。
        ctx.save_for_backward(input, weight)
        # 在上下文对象ctx中存储其他变量和标志。
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        # 如果启用了序列并行，则进行以下操作：
        if sequence_parallel:
            gd.debuginfo(prj='mt')
            # 获取模型并行的world_size（通常是参与并行处理的GPU数量）。
            world_size = get_tensor_model_parallel_world_size()
            # 更改输入的第一个维度以考虑模型并行的全部大小。
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            # 收集所有GPU上的输入。
            get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group())
            # 更新total_input为收集的数据。
            total_input = all_gather_buffer
        else:
            gd.debuginfo(prj='mt')
            # 如果不使用序列并行，则total_input仅仅是传入的输入。
            total_input = input

        # 对total_input和weight的转置进行矩阵乘法以计算输出。
        output = torch.matmul(total_input, weight.t())
        # 如果提供了偏置，则将其添加到输出中
        if bias is not None:
            gd.debuginfo(prj='mt')
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        gd.debuginfo(prj='mt')

        # 从上下文对象中恢复前向传播保存的张量
        input, weight = ctx.saved_tensors

        # 从上下文对象中恢复偏置使用的信息。
        use_bias = ctx.use_bias

        # 如果启用了序列并行，要如何获取完整的输入数据。
        # 它通过分布式的_all_gather_base函数来异步地聚集所有输入。
        if ctx.sequence_parallel:
            gd.debuginfo(prj='mt')
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_tensor_model_parallel_group(), async_op=True)

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        # 如果没有启用序列并行，那么完整的输入就是原始输入。
        else:
            gd.debuginfo(prj='mt')
            total_input = input

        # 通过矩阵乘法计算关于输入的梯度。
        grad_input = grad_output.matmul(weight)

        # 如果启用了序列并行，则等待所有聚集操作完成。
        if ctx.sequence_parallel:
            gd.debuginfo(prj='mt')
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        # 这些是注释，提到在NeMo的前向传递中，执行gather和slicing操作可能会导致grad_output张量
        # 不是连续的。PyTorch只检查张量是否是连续的，并且只在不连续时克隆它。
        grad_output = grad_output.contiguous() # 确保grad_output是连续的

        # Convert the tensor shapes to 2D for execution compatibility
        # 将grad_output张量的形状转化为2D，以确保兼容性。
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1],
                                       grad_output.shape[2])

        # 同样地，将total_input张量也转化为2D。
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1],
    			       total_input.shape[2])

        # 如果启用了异步的梯度all-reduce，执行该操作。这是一个分布式操作，用于聚合所有工作节点上的梯度。
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        # 如果启用了序列并行，则不应该在此处启用异步all-reduce（由assert语句确保）。
        # 接着，创建一个新的sub_grad_input张量，并执行一个reduce_scatter操作。
        # 这是一个分布式操作，它会将输入的梯度从所有工作节点上聚合到一个工作节点上。
        if ctx.sequence_parallel:
            gd.debuginfo(prj='mt')
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
                                                            group=get_tensor_model_parallel_group(),
                                                            async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        # 根据是否启用了梯度累积融合，使用特定的CUDA操作或标准的矩阵乘法来计算权重的梯度。
        # 这个条件检查是否启用了梯度累积融合。梯度累积通常在小批量训练中用于累积梯度以在较大的有效批量上更新模型。
        if ctx.gradient_accumulation_fusion:
            gd.debuginfo(prj='mt')
            if weight.main_grad.dtype == torch.float32:
                gd.debuginfo(prj='mt')
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output, weight.main_grad)
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                gd.debuginfo(prj='mt')
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output, weight.main_grad)
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            # 在梯度累积融合的情况下，设置grad_weight为None，
            # 这意味着梯度已经在前面的CUDA函数中直接更新了（weight.main_grad），所以在这里没有返回值。
            grad_weight = None
        else:
            gd.debuginfo(prj='mt')
            grad_weight = grad_output.t().matmul(total_input)
        # 如果使用偏置，则计算关于偏置的梯度。
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # 如果启用了序列并行，等待上述操作完成，并返回计算得到的梯度。
        if ctx.sequence_parallel:
            gd.debuginfo(prj='mt')
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        # 如果启用了异步all-reduce，等待all-reduce操作完成。
        if ctx.async_grad_allreduce:
            gd.debuginfo(prj='mt')
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None

#可以看到gradient_accumulation_fusion这个优化作用于Linear层中对weight求梯度的时候，调用了apex库提供的2个fuse cuda kernel原地更新了weight的梯度。

# 这部分定义了一个函数，名为linear_with_grad_accumulation_and_async_allreduce，
# 它接收七个参数：输入张量、权重张量、一个可选的偏置张量和3个布尔标志。
def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion."

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel_enabled is True, this must be
        False, as no all reduce is performed.

    sequence_parallel_enabled (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """
    """带有反向传播的异步通信和梯度累积融合的线性层实现.

    此函数提供了一个选项，可以将反向传播计算的结果累积到一个现有的梯度缓冲区中，
    从而避免在梯度计算后进行额外的加法核操作。

    此外，输入梯度的张量并行all reduce可以与权重梯度的计算异步进行。

    在使用序列并行的情况下，输入梯度的reduce scatter与权重梯度的计算异步进行。

    使用此模块需要环境变量CUDA_DEVICE_MAX_CONNECTIONS=1。代码中有一些集合操作，
    应该在计算核之前调度，以使通信与计算重叠，这对于加速是必要的，但对于正确性则不是必要的，
    因此调度器不会强制这种排序。将CUDA_DEVICE_MAX_CONNECTIONS设置为1会强制按照它们被调用的顺序调度内核。

    Arguments:

    input (torch.Tensor required): 输入，类似torch.nn.functional.linear

    weight (torch.Tensor required): 权重，类似torch.nn.functional.linear

    bias (torch.Tensor optional): 偏置，类似torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): 执行梯度累积融合，
    需要自定义的CUDA扩展模块fused_weight_gradient_mlp_cuda。
    要使用gradient_accumulation_fusion，你必须使用--cpp_ext和--cuda_ext安装APEX。
    例如："pip install --global-option="--cpp_ext" --global-option="--cuda_ext" ." ===这是打印出来的建议，实际还可以尝试
    pip install -v --no-cache-dir ./
    或者
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    注意，此扩展要求CUDA版本大于或等于11。否则，你必须关闭梯度累积融合。

    async_grad_allreduce (bool required): 异步地与权重梯度的计算进行输入梯度的allreduce。
    如果sequence_parallel_enabled为True，这必须为False，因为不执行allreduce。

    sequence_parallel_enabled (bool required): 表示使用了序列并行，
    因此在前向传播中，输入是add gather后的，在反向传播中，输入梯度是reduce scatter后的。
    """
    gd.debuginfo(prj='mt')
    # 这部分创建了一个名为args的列表，它基本上是函数输入参数的集合。
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel_enabled,
    ]

    # 这部分检查是否已经发出警告。函数使用一个类级别变量warned来记住是否已经向用户显示了警告。
    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        gd.debuginfo(prj='mt')
        # 这部分检查环境变量CUDA_DEVICE_MAX_CONNECTIONS是否设置为"1"。
        # 如果没有，并且满足某些条件（sequence_parallel_enabled或async_grad_allreduce），
        # 它会发出警告。然后将warned标志设置为True，以便不会重复发出此警告。
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            gd.debuginfo(prj='mt')
            if sequence_parallel_enabled:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    # 最后，函数调用另一个名为LinearWithGradAccumulationAndAsyncCommunication的类并返回其结果。
    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)

# 在函数外部，初始化属性warned为False。这用于检查是否已经向用户发出警告。
linear_with_grad_accumulation_and_async_allreduce.warned = False

'''
4.1 定义
对于 Y = XA + b ，
A被以如下的方式进行并行化： A = 【A1，，，，，Ap】
通过类ColumnParallelLinear的注释可以看出来 ，
'''
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
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
        async_tensor_model_parallel_allreduce:
        params_dtype:
        use_cpu_initialization:
        gradient_accumulation_fusion:
        sequence_parallel_enabled:
    """
    '''
    4.2 初始化
    __init__ 函数主要是用切分的信息来初始化权重。
    '''
    def __init__(self, input_size, output_size, *,
                 bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 async_tensor_model_parallel_allreduce=True,
                 params_dtype=torch.float32,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 gradient_accumulation_fusion=False,
                 sequence_parallel_enabled: bool = False,
                 ):
        gd.debuginfo(prj='mt')
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()  # 获得该tensor并行组的world size
        self.output_size_per_partition = divide(output_size, world_size) # 获得该子模型应输出size
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            gd.debuginfo(prj='mt')
            # 用切分的size初始化权重
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=params_dtype))
            if perform_initialization:
                gd.debuginfo(prj='mt')
                self.master_weight = _initialize_affine_weight_cpu(  # 初始化权重
                    self.weight, self.output_size, self.input_size,
                    self.output_size_per_partition, 0, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            gd.debuginfo(prj='mt')
            # 用切分的size初始化权重
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                gd.debuginfo(prj='mt')
                _initialize_affine_weight_gpu(self.weight, init_method, # 初始化权重
                                              partition_dim=0, stride=stride)

        if bias:
            gd.debuginfo(prj='mt')
            if use_cpu_initialization:
                gd.debuginfo(prj='mt')
                # 用切分的size初始化权重
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=params_dtype))
            else:
                gd.debuginfo(prj='mt')
                # 用切分的size初始化权重
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype))

            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            gd.debuginfo(prj='mt')
            self.register_parameter('bias', None)

        self.async_tensor_model_parallel_allreduce = (
                async_tensor_model_parallel_allreduce and
                world_size > 1)

        if sequence_parallel_enabled:
            gd.debuginfo(prj='mt')
            if world_size <= 1:
                warnings.warn(
                    f"`sequence_parallel_enabled` is set to `True`, but tensor model parallel size is {world_size}. "
                    f"Disabling sequence parallel."
                )
                sequence_parallel_enabled = False
        self.sequence_parallel_enabled = sequence_parallel_enabled

        if gradient_accumulation_fusion:
            gd.debuginfo(prj='mt')
            if not _grad_accum_fusion_available:
                raise RuntimeError(
                    "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                    "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                    "module is not found. To use gradient_accumulation_fusion you must "
                    "install APEX with --cpp_ext and --cuda_ext. For example: "
                    "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                    "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                    "gradient accumulation fusion."
                )
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` "
                "cannot be enabled at the same time."
            )

    '''
    4.4.1 ColumnParallelLinear
    
    ColumnParallelLinear 的 forward 代码之中，
    主要是实施了 f 和 g 的forward操作，同时把 f和 g 的 backward 操作搭建起来：
    
        如果gather_output为 True，则在前向传播时候把 Yi 做all-gather，
        因为反向传播时需要把完整梯度scatter到对应GPU之上，所以要搭建对于的split操作。
        MLP实现之中，此处设置为 False，这样每个GPU 输出的是自己对应 partition 的 4h/p，直接传送给下一个线性层。
    '''
    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        gd.debuginfo(prj='mt')
        # 如果选择忽略 bias，就会设置为 None，后续就不用处理了
        bias = self.bias if not self.skip_bias_add else None

        # 下面主要是图中的 f 操作
        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            gd.debuginfo(prj='mt')
            # 建立反向传播时候的异步all-reduce
            input_parallel = input_
        else:
            gd.debuginfo(prj='mt')
            # Set up backprop all-reduce.
            # 建立反向传播all-reduce，就是图中f的backward
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        # Maxtrix multiply with asynchronouse all-reduce execution.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
        )

        # 下面就是图中的 g 操作
        if self.gather_output:  # 是否需要聚合操作
            # All-gather across the partitions.
            # 聚合输出，就是图中 g 的 forward
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel)
            gd.debuginfo(prj='mt')
        else:
            gd.debuginfo(prj='mt')
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None # 如果不忽略bias，还得传出去
        return output, output_bias

'''
5.1 定义
通过注释，可以看出
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

    Keyword Arguments:
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
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        gradient_accumulation_fusion:
        sequence_parallel_enabled:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 params_dtype=torch.float32,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 gradient_accumulation_fusion=False,
                 sequence_parallel_enabled: bool = False,
                 ):
        gd.debuginfo(prj='mt')

        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if use_cpu_initialization:
            gd.debuginfo(prj='mt')
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=params_dtype))
            if perform_initialization:
                gd.debuginfo(prj='mt')
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.input_size_per_partition, 1, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test,
                    params_dtype=params_dtype)
        else:
            gd.debuginfo(prj='mt')
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                gd.debuginfo(prj='mt')
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=1, stride=stride)
        if bias:
            gd.debuginfo(prj='mt')
            if use_cpu_initialization:
                gd.debuginfo(prj='mt')
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=params_dtype))
            else:
                gd.debuginfo(prj='mt')
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=params_dtype))
            setattr(self.bias, 'sequence_parallel', sequence_parallel_enabled)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    '''
    5.4 代码实现
    5.4.1 RowParallelLinear
    
    RowParallelLinear 的 forward 代码之中，主要是实施了 f 和 g 的forward操作，同时把 f 和 g 的 backward 操作搭建起来，
    具体如下：
    '''
    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            gd.debuginfo(prj='mt')
            input_parallel = input_
        else:
            gd.debuginfo(prj='mt')
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
        )

        # All-reduce across all the partitions.
        if self.sequence_parallel_enabled:
            gd.debuginfo(prj='mt')
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            gd.debuginfo(prj='mt')
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            gd.debuginfo(prj='mt')
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            gd.debuginfo(prj='mt')
            output = output_
            output_bias = self.bias

        return output, output_bias
