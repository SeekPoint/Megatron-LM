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

from abc import ABC
from abc import abstractmethod

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_args
from megatron import mpu
from .module import MegatronModule


'''
5.2.3 内存
MemoryBuffer 是内存抽象。
'''
class MemoryBuffer:

    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.zeros(self.numel,   # 初始化内存
                                dtype=self.dtype,
                                device=torch.cuda.current_device(),
                                requires_grad=False)


    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()   # 定位到该张量在内存buffer之中的位置
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]  # 拿到内存
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor



class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        self.module = module


    @abstractmethod
    def allreduce_gradients(self):
        pass


    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


'''
5.2.1 定义
定义只有注释可以看看，使用连续的（contiguous）内存来存储和累积梯度，每一种类型的张量属于一个统一的内存，可以统一做 allreduce。


LocalDDP实现在DistributedDataParallel类. 
从arguments.py的默认参数来看，use_contiguous_buffers这个flag默认是True. 那么会开一份连续的buffer用于通信，记作main_grad. 
并且会注册一个backward hook，用于accumulate gradients. 相关代码如下:


其中，MemoryBuffer就是torch.zeros创建的一个in-memory buffer on GPU.

所以LocalDDP是没有计算和通信的overlap的. 其实pipeline parallelism完全可以用torchDDP，设置下gradient accumulation steps为m就行了. 因为pipeline parallelism把一个minibatch拆成m个microbatches，和gradient accumulation没有区别.

另外我不清楚为什么要搞一个continuous buffer. 注释里面有一句:

has the potential to reduce memory fragmentation.
还是不理解...

'''
class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to storre and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """
    '''
    5.2.2 初始化
初始化方法的目的是把同类型梯度连续存储。
    '''
    def __init__(self, module,
                 accumulate_allreduce_grads_in_fp32,
                 use_contiguous_buffers):

        super(DistributedDataParallel, self).__init__(module)

        self.accumulate_allreduce_grads_in_fp32 \
            = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        if self.use_contiguous_buffers:  # 这里只考虑连续内存
            self._grad_buffers = {}  # 定义buffer

            # Simple function to define buffer type.
            def _get_buffer_type(param):  # 返回buffer类型
                return torch.float if \
                    self.accumulate_allreduce_grads_in_fp32 else param.dtype

            # First calculate total number of elements per type.
            type_num_elements = {}
            for param in self.module.parameters():   # 遍历模型参数
                if param.requires_grad:   # 如果需要计算梯度
                    dtype = _get_buffer_type(param)  # 获取参数类型
                    type_num_elements[dtype] = type_num_elements.get(dtype, 0) \
                                               + param.data.nelement()  # 该类型参数数目做相应增加

            # 目前 type_num_elements 是各种类型参数的个数
            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items(): # 遍历各种类型
                self._grad_buffers[dtype] = MemoryBuffer(num_elements, dtype)  # 分配内存

            # 这里是假定反向传播是参数的反方向，存储每个参数梯度的起始位置
            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in self.module.parameters(): # 遍历模型参数
                if param.requires_grad: # 如果需要计算梯度
                    dtype = _get_buffer_type(param) # 获取参数类型
                    type_num_elements[dtype] -= param.data.nelement()  # 减少size
                    # 确定该参数在MemoryBuffer的位置
                    param.main_grad = self._grad_buffers[dtype].get( # 获取该参数对应的内存
                        param.data.shape, type_num_elements[dtype])

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():  # 遍历模型参数
                if param.requires_grad:  # 如果需要计算梯度
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]  # 得到参数对应的梯度函数
                    grad_acc.register_hook(self._make_param_hook(param))  # 注册了hook
                    self.grad_accs.append(grad_acc)  # 统一管理梯度函数，其实就是book keeping作用

    '''
    5.2.4 支撑函数
下面是两个支撑函数，分别是用于拷贝梯度和#将buffer清零。
我们假定模型有6个参数，3个 fp32，3 个 fp16，所以被组合成两个连续内存 MemoryBuffer。
图！！！！！！
    '''
    #拷贝梯度
    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""
        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            if param.grad is not None:
                # The gradient function of linear layers is fused with GEMMs
                #  # 把梯度拷贝到连续内存之中
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                param.grad = None
        return param_hook

    #将buffer清零
    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        begining of each iteration."""
        assert self._grad_buffers is not None, 'buffers are not initialized.'
        for _, buffer_ in self._grad_buffers.items():
            buffer_.zero()


    def broadcast_params(self):
        for param in self.module.parameters():
            torch.distributed.broadcast(param.data,
                                        src=mpu.get_data_parallel_src_rank(),
                                        group=mpu.get_data_parallel_group())

    '''
    5.2.5 梯度规约
allreduce_gradients 是 DDP 对外提供的 API，在后面 train step 之中会调用到。

运行时候，分别对两种类型的连续内存做 AllReduce。
图！！！！


在 allreduce_gradients之中，会对本数据并行组进行all-reduce。
    '''
    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        # 连续内存
        if self._grad_buffers is not None:
            for _, buffer_ in self._grad_buffers.items():  # 遍历各种类型的buffer
                buffer_.data /= mpu.get_data_parallel_world_size()  # 数据并行 world size
                torch.distributed.all_reduce( # 统一归并
                    buffer_.data, group=mpu.get_data_parallel_group())  # 数据并行组
        else:
            # Otherwise, bucketize and all-reduce
            buckets = {} # 否则还是用桶来归并
            # Pack the buckets.
            for param in self.module.parameters(): # 遍历梯度
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param) # 同类型的梯度放到对应类型的桶之中
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket] # 把桶里的梯度拿出来
                coalesced = _flatten_dense_tensors(grads) # 打平梯度
                coalesced /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(  # 归并
                    coalesced, group=mpu.get_data_parallel_group())
                for buf, synced in zip(grads, _unflatten_dense_tensors(
                        coalesced, grads)):
                    buf.copy_(synced)
