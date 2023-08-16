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

from functools import reduce
import operator
import torch

from megatron import get_args
from megatron import mpu

#具体使用是在 megatron/p2p_communication.py，_communicate 之中会用流水线组信息来进行通信。这里省略了大部分代码。
'''
3.4 通信模块
3.4.1 基础通信方法
pipeline parallelism需要inter-stage的P2P通信，其主要实现是_communnicate函数，_communicate 函数主要是封装了 PyTorch 的基础通信函数，给流水线并行提供了stage之间的双向P2P通信。在此基础之上，又封装了一些API方法。这个函数的注释写得不错，解释得非常清楚。这里需要注意的是：每个层怎么知道自己在流水线之中上下游 rank 是什么？这是通过例如这样的调用mpu.get_pipeline_model_parallel_next_rank() 来知道的。

_communicate 具体代码如下：



pipeline parallelism需要inter-stage的P2P通信. 主要实现是_communnicate函数. 其他几个API都是调用该函数.

这个函数的注释写得不错，解释得非常清楚. 实现的功能是在stages中双向send/recv.



_communicate函数支持双向的send/recv，

另外对于pipeline的first stage和last stage需要特判一下. 比如recv_forward对于first stage就不需要做，recv_backward对于last stage也不需要做.

'''
def _communicate(tensor_send_next, tensor_send_prev, recv_prev, recv_next,
                 tensor_shape,
                 use_ring_exchange=False,
                 dtype_=None):
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Takes the following arguments:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        tensor_shape: shape of tensor to receive (this method assumes that all
                      tensors sent and received in a single function call are
                      the same shape).
        use_ring_exchange: boolean for whether torch.distributed.ring_exchange()
                           API should be used.
        dtype_: optional, this is used when the tensor that needs to be
                communicated is different from args.params_dtype.
    Returns:
        (tensor_recv_prev, tensor_recv_next)
    """
    args = get_args()

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    # Some legacy inference code doesn't set the tensor shape, do so now
    # for the normal values for gpt/bert. This could be removed if inference
    # code is changed to provide tensor_shape.
    if tensor_shape is None:
        tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)

    override_scatter_gather_tensors_in_pipeline = False
    if args.scatter_gather_tensors_in_pipeline and \
            not args.sequence_parallel:
        tensor_chunk_shape = reduce(operator.mul, tensor_shape, 1)
        if tensor_chunk_shape % mpu.get_tensor_model_parallel_world_size() == 0:
            tensor_chunk_shape = tensor_chunk_shape // \
                mpu.get_tensor_model_parallel_world_size()
        else:
            tensor_chunk_shape = tensor_shape
            override_scatter_gather_tensors_in_pipeline = True
    else:
        tensor_chunk_shape = tensor_shape
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float

    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

    # 如果需要接受张量，则先分配空张量，接受的张量会存在此处
    #1. 如果需要recv，会临时创建一个empty tensor作为buffer，并且返回.
    if recv_prev:
        tensor_recv_prev = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)
    if recv_next:
        tensor_recv_next = torch.empty(tensor_chunk_shape,
                                       requires_grad=requires_grad,
                                       device=torch.cuda.current_device(),
                                       dtype=dtype)

    # Split tensor into smaller chunks if using scatter-gather optimization.
    '''
    3. 有一个trick - scatter-gather optimization. NVIDIA在SC '21上的论文 里面提到过这个优化. 
    在使用了tensor model parallelism的时候，最后一层的输出是重复的 (因为做了all-reduce)，
    所以可以split成t份然后再发，下游收到后再做一个gather得到完整的数据.
    '''
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline and \
            not args.sequence_parallel:
        if tensor_send_next is not None:
            tensor_send_next = mpu.split_tensor_into_1d_equal_chunks(tensor_send_next)

        if tensor_send_prev is not None:
            tensor_send_prev = mpu.split_tensor_into_1d_equal_chunks(tensor_send_prev)

    # Send tensors in both the forward and backward directions as appropriate.
    # 这里使用get_pipeline_model_parallel_group 进行通信
    if use_ring_exchange:
        # 如果需要，则使用ring exchange，这个是新版本PyTorch才有
        torch.distributed.ring_exchange(tensor_send_prev=tensor_send_prev,
                                        tensor_recv_prev=tensor_recv_prev,
                                        tensor_send_next=tensor_send_next,
                                        tensor_recv_next=tensor_recv_next,
                                        group=mpu.get_pipeline_model_parallel_group())
    else:
        # 先根据目标rank生成对应的torch.distributed.P2POp，放入列表
        #2. 使用了torch.distributed.batch_isend_irecv，批量做异步send/recv (实际上就是个for循环...) 然后调用wait()进行同步.
        ops = []
        if tensor_send_prev is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_prev,
                mpu.get_pipeline_model_parallel_prev_rank())   # 得到流水线前一个rank
            ops.append(send_prev_op)
        if tensor_recv_prev is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_prev,
                mpu.get_pipeline_model_parallel_prev_rank())
            ops.append(recv_prev_op)
        if tensor_send_next is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor_send_next,
                mpu.get_pipeline_model_parallel_next_rank())  # 得到流水线下一个rank
            ops.append(send_next_op)
        if tensor_recv_next is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor_recv_next,
                mpu.get_pipeline_model_parallel_next_rank())
            ops.append(recv_next_op)
        # 然后做批量异步send/recv
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()   # 用wait来同步
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    # If using scatter-gather optimization, gather smaller chunks.
    # 特殊优化，21年论文中提到，大概因为做了all-reduce，因此可以先split发送，下游gather成统一数据
    # 有兴趣读者可以深入研究论文和代码，
    if not override_scatter_gather_tensors_in_pipeline and \
            args.scatter_gather_tensors_in_pipeline and \
            not args.sequence_parallel:
        if recv_prev:
            tensor_recv_prev = mpu.gather_split_1d_tensor(
                tensor_recv_prev).view(tensor_shape).requires_grad_()
            tensor_recv_prev = mpu.make_viewless_tensor(tensor_recv_prev,
                                                        requires_grad = True,
                                                        keep_graph = False)

        if recv_next:
            tensor_recv_next = mpu.gather_split_1d_tensor(
                tensor_recv_next).view(tensor_shape).requires_grad_()
            tensor_recv_next = mpu.make_viewless_tensor(tensor_recv_next,
                                                        requires_grad = True,
                                                        keep_graph = False)

    return tensor_recv_prev, tensor_recv_next


def recv_forward(tensor_shape=None, dtype_=None, timers=None):
    """Receive tensor from previous rank in pipeline (forward receive)."""

    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('forward-recv').start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype_)
        if timers is not None:
            timers('forward-recv').stop()
    return input_tensor


def recv_backward(tensor_shape=None, timers=None):
    """Receive tensor from next rank in pipeline (backward receive)."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('backward-recv').start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('backward-recv').stop()
    return output_tensor_grad


def send_forward(output_tensor, tensor_shape=None, dtype_=None, timers=None):
    """Send tensor to next rank in pipeline (forward send)."""

    if not mpu.is_pipeline_last_stage():
        if timers is not None:
            timers('forward-send').start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype_)
        if timers is not None:
            timers('forward-send').stop()


def send_backward(input_tensor_grad, tensor_shape=None, timers=None):
    """Send tensor to previous rank in pipeline (backward send)."""
    if not mpu.is_pipeline_first_stage():
        if timers is not None:
            timers('backward-send').start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('backward-send').stop()


def send_forward_recv_backward(output_tensor, tensor_shape=None, timers=None):
    """Batched send and recv with next rank in pipeline."""
    if mpu.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if timers is not None:
            timers('forward-send-backward-recv').start()
        _, output_tensor_grad = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(input_tensor_grad, tensor_shape=None, timers=None):
    """Batched send and recv with previous rank in pipeline."""
    if mpu.is_pipeline_first_stage():
        input_tensor = None
    else:
        if timers is not None:
            timers('backward-send-forward-recv').start()
        input_tensor, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape)
        if timers is not None:
            timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(output_tensor, recv_prev, tensor_shape=None, timers=None):
    """Batched recv from previous rank and send to next rank in pipeline."""
    if timers is not None:
        timers('forward-send-forward-recv').start()
    input_tensor, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape)
    if timers is not None:
        timers('forward-send-forward-recv').stop()
    return input_tensor


def send_backward_recv_backward(input_tensor_grad, recv_next, tensor_shape=None, timers=None):
    """Batched recv from next rank and send to previous rank in pipeline."""
    if timers is not None:
        timers('backward-send-backward-recv').start()
    _, output_tensor_grad = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape)
    if timers is not None:
        timers('backward-send-backward-recv').stop()
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
        output_tensor, input_tensor_grad, recv_prev,
        recv_next, tensor_shape=None, timers=None):
    """Batched send and recv with previous and next ranks in pipeline."""
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').start()
    input_tensor, output_tensor_grad = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape)
    if timers is not None:
        timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad
