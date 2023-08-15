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

from contextlib import contextmanager
import torch
from torch.autograd.variable import Variable
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_num_microbatches
from megatron import get_timers
from megatron import mpu
from megatron import p2p_communication
from megatron.utils import unwrap_model
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.model import ModelType

'''
6.3 获取schedule
get_forward_backward_func 获取 pipeline 的schedule，这里分为 flush 和 interleaving 两种，我们后续会分析这两种schedule。

训练逻辑大体拓展为：

图！！！！

至此，Megatron 基本架构分析完毕，下一篇我们介绍模型并行设置。



0x01 背景
在流水线训练之中，如何给流水线各个阶段安排执行执行序列是一个关键，所以这里我们看看如何做schedule。

对于 Megatron 来说，在训练时候，get_forward_backward_func 获取pipeline 的schedule，这里分为 flush 和 interleaving 两种， 因为时间所限，我们只分析 flush 的schedule，有兴趣的读者可以自行研究 interleaving。

概括来说，Megatron 是基于 PipeDream-2BW 之上实现了定期刷新。

PipeDream-2BW 在流水线之中维护了两个版本的模型权重，“2BW” 是 双缓冲权重（double-buffered weights）”，PipeDream-2BW 会为每个微批次生成一个新的模型版本K（K>d），但是因为有些剩余后向传递仍然依赖于旧版本模型，所以新的模型版本无法立即取代旧版本，但是由于只保存了两个版本，所以极大降低了内存占用。
PipeDream-flush 则在 PipeDream-2BW 之上添加了一个全局同步的流水线更新刷新操作，思路类似 GPipe。这种方法通过牺牲吞吐量的能力部分下降的代价来减少了内存占用（即只维护一个版本的模型权重）。


0x03 PipeDream-Flush 实现
我们前面提到，当没有设置 virtual_pipeline_model_parallel_size 时候，就是使用 Flush 方法得到流水线schedule，具体函数是 forward_backward_pipelining_without_interleaving。

为何要选择 1F1B？论文作者提到，因为它将in-flight microbatches 数量缩减到流水线深度 d，而不是GPipe的微批次数目 m，所以 1F1B 是memory-efficient。为了降低bubble time，一般来说，m >> d。

'''
def get_forward_backward_func():
    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if args.virtual_pipeline_model_parallel_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
            assert get_num_microbatches() % \
                args.pipeline_model_parallel_size == 0, \
                'number of microbatches (%d) is not divisible by pipeline-' \
                'model-parallel-size (%d) when using interleaved schedule' % (
                    get_num_microbatches(),
                    args.pipeline_model_parallel_size,
                )
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def deallocate_output_tensor(out):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if out is None:
        return
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device = out.device,
        dtype = out.dtype,
    )
        
def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    assert output.numel() == 1, \
        "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), \
        "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), \
        "grad_output == '%s'." % type(grad_output).__name__

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(
            output,
            memory_format = torch.preserve_format,
        )

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors = (output,),
        grad_tensors = (grad_output,),
        keep_graph = False,
        create_graph = False,
        inputs = tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )
        

def forward_step(forward_step_func,
                 data_iterator,
                 model,
                 input_tensor,
                 forward_data_store,
                 collect_non_loss_data=False):
    """Forward step for passed-in model.

    If first stage, input tensor is obtained from data_iterator, otherwise
    passed-in input_tensor is used.

    Returns output tensor."""
    args = get_args()
    timers = get_timers()

    timers('forward-compute').start()
    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    unwrapped_model.set_input_tensor(input_tensor)
    output_tensor, loss_func = forward_step_func(data_iterator, model)
    if mpu.is_pipeline_last_stage():
        if not collect_non_loss_data:
            output_tensor = loss_func(output_tensor)
            loss, loss_reduced = output_tensor
            output_tensor = loss / get_num_microbatches()
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    timers('forward-compute').stop()

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    if mpu.is_pipeline_stage_after_split() and \
            args.model_type == ModelType.encoder_and_decoder:
        return [output_tensor, input_tensor[-1]]
    if unwrap_output_tensor:
        return output_tensor
    return [output_tensor]


def backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.
    args = get_args()

    timers = get_timers()
    timers('backward-compute').start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None:
        output_tensor = optimizer.scale_loss(output_tensor[0])
    custom_backward(output_tensor[0], output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
            mpu.is_pipeline_stage_after_split() and \
            args.model_type == ModelType.encoder_and_decoder:
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    timers('backward-compute').stop()

    return input_tensor_grad


@contextmanager
def dummy_handler():
    try:
        yield
    finally:
        pass


def forward_backward_no_pipelining(forward_step_func,
                                   data_iterator, model,
                                   optimizer,
                                   timers,
                                   forward_only,
                                   collect_non_loss_data=False):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses."""
    assert len(model) == 1
    model = model[0]

    context_handler = dummy_handler
    if isinstance(model, torchDDP):
        context_handler = model.no_sync

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    with context_handler():
        for i in range(get_num_microbatches() - 1):
            output_tensor = forward_step(forward_step_func, data_iterator,
                                         model, input_tensor, forward_data_store,
                                         collect_non_loss_data)
            if not forward_only:
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(forward_step_func, data_iterator,
                                 model, input_tensor, forward_data_store,
                                 collect_non_loss_data)
    if not forward_only:
        backward_step(optimizer, input_tensor, output_tensor, output_tensor_grad)

    return forward_data_store


def forward_backward_pipelining_with_interleaving(forward_step_func,
                                                  data_iterator, model,
                                                  optimizer,
                                                  timers,
                                                  forward_only, 
                                                  collect_non_loss_data=False):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    forward_data_store = []
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]

    pipeline_parallel_size = mpu.get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = mpu.get_pipeline_model_parallel_rank()

    args = get_args()
    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    tensor_shape = (seq_length, args.micro_batch_size, args.hidden_size)
    
    # Compute number of warmup and remaining microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches() * num_model_chunks
    all_warmup_microbatches = False
    if forward_only:
        num_warmup_microbatches = num_microbatches
    else:
        # Run all forward passes and then all backward passes if number of
        # microbatches is just the number of pipeline stages.
        # Otherwise, perform (num_model_chunks-1)*pipeline_parallel_size on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        if get_num_microbatches() == pipeline_parallel_size:
            num_warmup_microbatches = num_microbatches
            all_warmup_microbatches = True
        else:
            num_warmup_microbatches = \
                (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (
                num_model_chunks - 1) * pipeline_parallel_size
            num_warmup_microbatches = min(num_warmup_microbatches,
                                          num_microbatches)
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    def get_model_chunk_id(microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (pipeline_parallel_size * num_model_chunks)
        model_chunk_id = microbatch_id_in_group // pipeline_parallel_size
        if not forward:
            model_chunk_id = (num_model_chunks - model_chunk_id - 1)
        return model_chunk_id

    def forward_step_helper(microbatch_id):
        """Helper method to run forward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        forward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        # forward step
        if mpu.is_pipeline_first_stage():
            if len(input_tensors[model_chunk_id]) == \
                    len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id][-1]
        output_tensor = forward_step(forward_step_func,
                                     data_iterator[model_chunk_id],
                                     model[model_chunk_id],
                                     input_tensor, 
                                     forward_data_store,
                                     collect_non_loss_data)
        output_tensors[model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if forward_only:
            input_tensors[model_chunk_id].pop()
            output_tensors[model_chunk_id].pop()

        return output_tensor

    def backward_step_helper(microbatch_id):
        """Helper method to run backward step with model split into chunks
        (run set_virtual_pipeline_model_parallel_rank() before calling
        backward_step())."""
        model_chunk_id = get_model_chunk_id(microbatch_id, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(model_chunk_id)

        if mpu.is_pipeline_last_stage():
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)
        input_tensor_grad = \
            backward_step(optimizer,
                          input_tensor,
                          output_tensor,
                          output_tensor_grad)

        return input_tensor_grad

    # Run warmup forward passes.
    mpu.set_virtual_pipeline_model_parallel_rank(0)
    input_tensors[0].append(
        p2p_communication.recv_forward(tensor_shape, timers=timers))
    for k in range(num_warmup_microbatches):
        output_tensor = forward_step_helper(k)

        # Determine if tensor should be received from previous stage.
        next_forward_model_chunk_id = get_model_chunk_id(k+1, forward=True)
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            if next_forward_model_chunk_id == 0:
                recv_prev = False
        if k == (num_microbatches - 1):
            recv_prev = False

        # Don't send tensor downstream if on last stage.
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if k == (num_warmup_microbatches - 1) and not forward_only and \
                not all_warmup_microbatches:
            input_tensor_grad = None
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                recv_next = False
            input_tensor, output_tensor_grad = \
                p2p_communication.send_forward_backward_recv_forward_backward(
                        output_tensor, input_tensor_grad,
                        recv_prev=recv_prev, recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        timers=timers)
            output_tensor_grads[num_model_chunks-1].append(output_tensor_grad)
        else:
            input_tensor = \
                p2p_communication.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    timers=timers)
        input_tensors[next_forward_model_chunk_id].append(input_tensor)
        deallocate_output_tensor(output_tensor)

    # Run 1F1B in steady state.
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches
        output_tensor = forward_step_helper(forward_k)

        # Backward pass.
        backward_k = k
        input_tensor_grad = backward_step_helper(backward_k)

        # Send output_tensor and input_tensor_grad, receive input_tensor
        # and output_tensor_grad.

        # Determine if current stage has anything to send in either direction,
        # otherwise set tensor to None.
        forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        mpu.set_virtual_pipeline_model_parallel_rank(forward_model_chunk_id)
        if mpu.is_pipeline_last_stage():
            output_tensor = None

        backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
        mpu.set_virtual_pipeline_model_parallel_rank(backward_model_chunk_id)
        if mpu.is_pipeline_first_stage():
            input_tensor_grad = None

        # Determine if peers are sending, and where in data structure to put
        # received tensors.
        recv_prev = True
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            # First stage is ahead of last stage by (pipeline_parallel_size - 1).
            next_forward_model_chunk_id = get_model_chunk_id(
                forward_k - (pipeline_parallel_size - 1), forward=True)
            if next_forward_model_chunk_id == (num_model_chunks - 1):
                recv_prev = False
            next_forward_model_chunk_id += 1
        else:
            next_forward_model_chunk_id = get_model_chunk_id(forward_k + 1,
                                                             forward=True)

        recv_next = True
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
            next_backward_model_chunk_id = get_model_chunk_id(
                backward_k - (pipeline_parallel_size - 1), forward=False)
            if next_backward_model_chunk_id == 0:
                recv_next = False
            next_backward_model_chunk_id -= 1
        else:
            next_backward_model_chunk_id = get_model_chunk_id(backward_k + 1,
                                                              forward=False)

        # If last iteration, don't receive; we already received one extra
        # before the start of the for loop.
        if k == (num_microbatches_remaining - 1):
            recv_prev = False

        # Communicate tensors.
        input_tensor, output_tensor_grad = \
            p2p_communication.send_forward_backward_recv_forward_backward(
                    output_tensor, input_tensor_grad,
                    recv_prev=recv_prev, recv_next=recv_next,
                    tensor_shape=tensor_shape, timers=timers)
        deallocate_output_tensor(output_tensor)

        # Put input_tensor and output_tensor_grad in data structures in the
        # right location.
        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(
                output_tensor_grad)

    # Run cooldown backward passes (flush out pipeline).
    if not forward_only:
        if all_warmup_microbatches:
            output_tensor_grads[num_model_chunks-1].append(
                p2p_communication.recv_backward(tensor_shape, timers=timers))
        for k in range(num_microbatches_remaining, num_microbatches):
            input_tensor_grad = backward_step_helper(k)
            next_backward_model_chunk_id = get_model_chunk_id(k+1, forward=False)
            recv_next = True
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                if next_backward_model_chunk_id == (num_model_chunks - 1):
                    recv_next = False
            if k == (num_microbatches - 1):
                recv_next = False
            output_tensor_grads[next_backward_model_chunk_id].append(
                p2p_communication.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next,
                    tensor_shape=tensor_shape,
                    timers=timers))

    return forward_data_store


def get_tensor_shapes(rank, model_type):
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    args = get_args()
    tensor_shapes = []

    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length

    if model_type == ModelType.encoder_and_decoder:
        if args.sequence_parallel:
            decoder_seq_length = args.decoder_seq_length // mpu.get_tensor_model_parallel_world_size()
        else:
            decoder_seq_length = args.decoder_seq_length

        if mpu.is_pipeline_stage_before_split(rank):
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
        else:
            tensor_shapes.append((decoder_seq_length, args.micro_batch_size, args.hidden_size))
            tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
    else:
        tensor_shapes.append((seq_length, args.micro_batch_size, args.hidden_size))
    return tensor_shapes

# 其中，第一个stage因为没有上游，所以recv_forward将会返回None，其他情况下将返回一个上游激活。
def recv_forward(tensor_shapes, timers):
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape,
                                                                timers=timers))
    return input_tensors


def recv_backward(tensor_shapes, timers):
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape,
                                                                       timers=timers))
    return output_tensor_grads


def send_forward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, tensor_shape, timers=timers)


def send_backward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, tensor_shape, timers=timers)

'''
3.5.2 串行
其中，send_forward_recv_backward 这个从名字就能看到逻辑，这个函数先发送给下游，再从下游接受。

可以发现，对于单个 worker，都是阻塞进行，因为 send 和 recv 都是阻塞，这样通信和计算必须串行，不能重叠。因为前面热身阶段已经把前向传递一直从 worker 0 传送到 worker d，所以 worker d 可以直接拿到 input，就进行处理，然后直接进行反向计算，然后返回给上游。所以串行也无所谓。我们从论文之中的图例也可以看出来：



图：PipeDream-Flush在稳定状态下交替进行向前和向后传播，通过将激活隐藏限制为仅执行中（in-flight）的微批次来保持较低的内存占用。从图上可以看到:

Worker 1的执行序列是：1 FW(warmup), 2 FW, 1 BW，3 FW，2 BW，4 FW，3 BW，4 BW(cooldown)
Worker 2的执行序列是：1 FW，1BW， 2 FW， 2 BW， 3 FW， 3 BW， 4 FW， 4 BW，worker 2直接就进入了稳定状态。


'''
def send_forward_recv_backward(output_tensors, tensor_shapes, timers):
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        # 发送自己的激活，然后得到下游传上来的梯度
        output_tensor_grad = p2p_communication.send_forward_recv_backward(
                output_tensor, tensor_shape, timers=timers)
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads #返回梯度

'''
3.4.2 API
在 _communicate 的基础之上，封装了众多API函数，主要就是依据参数的不同来做不同处理，比如：
'''
def send_backward_recv_forward(input_tensor_grads, tensor_shapes, timers):
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(
                input_tensor_grad, tensor_shape, timers=timers)
        input_tensors.append(input_tensor)
    return input_tensors

'''
3.1 总体思路
3.1.1 缺省计划
GPipe提出了一个执行计划，其中首先执行一个批次中所有微批次的正向传播，然后执行所有微批次的反向传播（如图3所示）。我们可以量化GPipe流水线气泡的大小(𝑡𝑝𝑏 )。我们将批次中的微批次数量表示为𝑚，流水线阶段的数量（用于流水线并行的设备数量）为𝑝，每次迭代的理想时间为𝑡𝑖𝑑 （假设完美或理想的缩放），以及执行单个微批次前进和后退通道的时间𝑡𝑓 和𝑡𝑏。

在此计划中，流水线气泡包含：

在批次开始时的 𝑝 − 1 个前向传播。
在批次结束时候的 𝑝 − 1 个向后传播。
在流水线中花费的总时间𝑡𝑝𝑏 = (𝑝−1)·(𝑡𝑓 +𝑡𝑏)，于是此任务的处理时间为 𝑡𝑖𝑑 =𝑚·(𝑡𝑓 +𝑡𝑏)。因此，在流水线气泡中花费的计算时间的理想占比（fraction）为：

Bubble time fraction(pipeline bubble size)=tpbtid=p−1m


图3 : GPipe流水线计划，所有微批次（以数字表示）均为前向传播（蓝色），然后为后向传播（绿色）。灰色区域表示流水线气泡。为简单起见，我们假设前向传播的时间是后向传播的两倍。流水线计划的效率不取决于此时间因素。本例中的每个批次由8个微批次组成，每个蓝色或绿色框中的数字是给相应微批次的唯一标识符（比如，第一批由1− 8个微批次组成，第二批由微批次9− 16组成等）。优化器在流水线刷新时进行步进（step）并更新权重参数，以确保严格的优化器语义。

为了使气泡时间占比（fraction）很小，我们需要𝑚 ≫ 𝑝。但是对于这么大的𝑚, 这种方法具有很高的内存占用，因为它需要将中间激活（或在使用激活重新编译时仅为每个流水线阶段输入激活）保存在内存中，以供所有 𝑚 个微批次在训练迭代的整个生命周期中都使用到。

3.1.2 PipeDream计划


PipeDream-Flush 把一个迭代分成三个阶段:

预热前向传播阶段（warmup forward passes）：在这里，除了最后一个stage，每个worker 会做前向计算，进行不同数目的前向传播，并且向其下游发送激活，一直到最后一个stage被激发。该计划将执行中的（in-flight）微批次数量（未完成反向传播且需要保持激活的微批次数量）限制在流水线深度之内，而不是一个批次中的微批次数量。

稳定 1F1B 阶段（Run 1F1B in steady state）：进入稳定状态之后，每个 worker 都进行1F1B 操作。

冷却反向传播阶段（Cooldown backward passes）：此阶段会把执行中的（in-flight）的微批次执行完毕，只是执行反向计算和向反向计算下游发送梯度。

这个新计划在气泡中花费的时间与GPipe是相同的，但是未完成的向前传播的数量最多和流水线阶段的数量一样。因此，该计划要求将激活减少到 𝑝 或更少的微批次（GPipe计划则是 m 个微批次）。因此，当𝑚 ≫ 𝑝 的时候, PipeDream-Flush 的内存效率比GPipe高得多。

我们首先给出具体代码如下，后续会逐步分析。



3.2 启动阶段
这是在每个 worker 之上都会做的，每个worker 的rank 不同，具体逻辑如下：

首先需要确定本worker在热身阶段需要执行的微批次数目，是min((world-size - rank - 1), num_microbatches)，因为rank是依次递增，所以热身所需的微批次会逐次递减，直到为0，这样就会直接进入稳定阶段进行计算，比如 world size 为5，rank区间为0～4，微批次数目为4，则从前往后几个stage的热身批次为 5 - 0 - 1 = 4， 5 - 1 - 1 = 3， 5 - 2 - 1 = 2， 5 - 3 - 1 = 1， 5 - 4 - 1 = 0（就直接进入稳定状态）。
其次计算稳定阶段所需要计算的微批次。
当需要进行反向传播时候，需要建立两个FIFO队列，input_tensors 保存来自上游的激活，output_tensors 保存来自下游的激活。

'''
def forward_backward_pipelining_without_interleaving(forward_step_func,
                                                     data_iterator,
                                                     model,
                                                     optimizer,
                                                     timers,
                                                     forward_only,
                                                     collect_non_loss_data=False):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
    args = get_args()
    timers = get_timers()

    assert len(model) == 1
    model = model[0]

    # Compute number of warmup microbatches.
    num_microbatches = get_num_microbatches()  # 得到微批次数目

    '''
    # 需要确定本worker在热身阶段需要执行的微批次数目，是min((world-size - rank - 1), num_microbatches)
    # 因为rank是依次递增，所以热身所需的微批次会逐次递减，直到为0，这样就会直接进入稳定阶段进行计算
    # 比如 world size 为5，rank区间为0～4，微批次数目为4，则从前往后几个stage的热身批次为 5 - 0 - 1， 5 - 1 - 1， 5 - 2 - 1， 5 - 3 - 1， 5 - 4 - 1。
    '''
    num_warmup_microbatches = \
        (mpu.get_pipeline_model_parallel_world_size() -
         mpu.get_pipeline_model_parallel_rank() - 1)
    num_warmup_microbatches = min(
        num_warmup_microbatches,
        num_microbatches)

    # 计算稳定阶段所需要计算的微批次
    num_microbatches_remaining = \
        num_microbatches - num_warmup_microbatches

    unwrapped_model = unwrap_model(
        model, (torchDDP, LocalDDP, Float16Module))
    model_type = unwrapped_model.model_type
    rank = mpu.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(rank-1, model_type)
    send_tensor_shapes = get_tensor_shapes(rank, model_type)

    # Input, output tensors only need to be saved when doing backward passes
    # 当需要进行反向传播时候，需要建立两个队列，input_tensors 保存来自上游的激活，output_tensors 保存来自下游的激活
    input_tensors = None
    output_tensors = None
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []


    '''
    3.3 热身阶段
    热身阶段会根据本worker在热身阶段需要执行的微批次数目，依次进行处理：
    
        从上游获取输入激活。
        本地进行前向计算，上游输入的激活就是本stage的输入。
        向下游发送本地激活。
        如果需要反向传播，则每个 worker 在 input_tensor 之中保存上游激活，在output_tensor 之中保存发送给下游的激活。
        早期阶段会运行尽可能多的向前传播，这样后期阶段可以立即从1F1B开始。
    '''
    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # 从上游获取输入激活
        input_tensor = recv_forward(recv_tensor_shapes, timers=timers)

        # 本地进行前向计算，上游输入的激活就是本stage的输入
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, forward_data_store,
                                     collect_non_loss_data)

        # 向下游发送本地激活
        send_forward(output_tensor, send_tensor_shapes, timers=timers)

        if not forward_only:
            input_tensors.append(input_tensor)  # 保存上游激活
            output_tensors.append(output_tensor)  # 保存本地计算的激活，就是发送给下游的激活
            deallocate_output_tensor(output_tensor[0])

    '''
    3.5 稳定阶段
    稳定阶段的总体逻辑如下：前向计算 -> 发送激活给前向计算下游 & 从下游接受梯度 -> 后向计算 -> 给上游发送本worker计算的梯度 & 从上游接受激活。
    
    3.5.1 逻辑
    稳定阶段具体逻辑如下：
    
    forward_step ：拿到一个微批次（上游激活），进行本地前向计算。
    send_forward：
        如果只是前向传播，则调用send_forward把本地结算结果发送给下游。
        否则调用 send_forward_recv_backward : 本地计算结果发给下游，再从下游接受其梯度。
        每个 worker 在 input_tensor 之中保存上游激活，在output_tensor 之中保存发送给下游的激活。
    backward_step : 本地后向计算。
        从队列中弹出第一个未处理的（就是最早未处理的）上游激活。
        从队列弹出对应的本地激活。
        进行反向计算，利用(上游激活，本地激活，下游梯度)来对最早的未处理的微批次进行反向计算，得到本地梯度。
    send_backward：
        如果是最后一个微批次，只需要把本地梯度 input_tensor_grad 传递给前向计算的上游。
        否则调用 send_backward_recv_forward 把本地梯度 input_tensor_grad 传递给前向计算的上游，还需要从上游再获取一个激活值。
    跳回1继续处理下一个微批次（上游激活）。
    '''
    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        # 需要在稳定状态下运行，所以得拿到前面层的激活值
        input_tensor = recv_forward(recv_tensor_shapes, timers=timers)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = (i == (num_microbatches_remaining - 1))
        # 前向计算
        output_tensor = forward_step(forward_step_func, data_iterator, model,
                                     input_tensor, forward_data_store,
                                     collect_non_loss_data)
        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, timers=timers)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, timers=timers)

        else:
            # 发送中间激活给下游，并且从下游获取其反向梯度
            output_tensor_grad = \
                send_forward_recv_backward(output_tensor,
                                           send_tensor_shapes,
                                           timers=timers)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)  # 保存上游激活到队列
            output_tensors.append(output_tensor)  # 保存本地计算的激活，就是发送给下游的激活到队列
            deallocate_output_tensor(output_tensor[0])

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)  # 从队列中弹出第一个未处理的（就是最早未处理的）上游激活
            output_tensor = output_tensors.pop(0) # 从队列弹出对应的本地激活

            # 反向计算，利用(上游激活，本地激活，下游梯度)来对最早的未处理的微批次进行反向计算，得到本地梯度
            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)  # 下游传来的梯度在这里

            if last_iteration:
                input_tensor = None
                # 如果是最后一个微批次，把本地梯度 input_tensor_grad 传递给前向计算的上游
                send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)
            else:
                # 如果不是最后一个微批次，把本地梯度 input_tensor_grad 传递给前向计算的上游，还需要从上游再获取一个激活值
                input_tensor = \
                    send_backward_recv_forward(
                        input_tensor_grad, recv_tensor_shapes, timers=timers)
    '''
    3.6 冷却阶段
        冷却阶段和热身阶段对称，也执行num_warmup_microbatches个步骤，但是只做反向传播。
        这个阶段因为是清理未完毕的反向传播，所以只是从队列中pop。具体就是弹出上游激活和传递给下游的激活，然后进行梯度计算。
    '''
    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, timers=timers)

            input_tensor_grad = \
                backward_step(optimizer, input_tensor, output_tensor,
                              output_tensor_grad)

            send_backward(input_tensor_grad, recv_tensor_shapes, timers=timers)

    return forward_data_store
