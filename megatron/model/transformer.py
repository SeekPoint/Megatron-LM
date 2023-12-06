# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Transformer."""
from contextlib import nullcontext
import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from megatron import get_timers, get_args, get_retro_args, core, get_num_microbatches
from .module import MegatronModule
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.model import LayerNorm
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu
from pydebug import gd, infoTensor

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        gd.debuginfo(prj="mt", info=f'C: {self.__class__.__name__}')
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        gd.debuginfo(prj="mt", info=f'hidden_state={infoTensor(hidden_state)}')

        if self.drop_prob == 0. or not self.training:
            gd.debuginfo(prj="mt")
            return hidden_state

        keep_prob = 1 - self.drop_prob

        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        gd.debuginfo(prj="mt", info=f'shape={infoTensor(shape)}')

        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        gd.debuginfo(prj="mt", info=f'random_tensor={infoTensor(random_tensor)}')

        random_tensor.floor_()  # binarize
        gd.debuginfo(prj="mt", info=f'random_tensor={infoTensor(random_tensor)}')

        output = hidden_state.div(keep_prob) * random_tensor
        gd.debuginfo(prj="mt", info=f'output={infoTensor(output)}')

        return output

def _args_to_kwargs():

    args = get_args()

    common_kwargs = {
        "params_dtype": args.params_dtype,
        "use_cpu_initialization": args.use_cpu_initialization,
        "perform_initialization": args.perform_initialization,
        "gradient_accumulation_fusion": args.gradient_accumulation_fusion,
        "sequence_parallel_enabled": args.sequence_parallel,
    }

    gd.debuginfo(prj="mt", info=f'common_kwargs={common_kwargs}')

    return common_kwargs

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        gd.debuginfo(prj="mt", info=f'C: {self.__class__.__name__}')

        super(ParallelMLP, self).__init__()
        args = get_args()

        self.add_bias = args.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        # 列切分
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size * 2 if args.swiglu else args.ffn_hidden_size,
            bias=self.add_bias,
            gather_output=False, # 这里是false，采用第二种方案
            init_method=init_method,
            skip_bias_add=True,
            async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
            **_args_to_kwargs())

        self.bias_gelu_fusion = False
        self.activation_func = None
        self.swiglu = args.swiglu

        if args.openai_gelu:
            gd.debuginfo(prj="mt")
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            gd.debuginfo(prj="mt")
            self.activation_func = erf_gelu
        elif args.swiglu:
            gd.debuginfo(prj="mt")
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu
        elif args.squared_relu:
            gd.debuginfo(prj="mt")
            def squared_relu(x):
                return torch.pow(F.relu(x), 2)
            self.activation_func = squared_relu
        else:
            gd.debuginfo(prj="mt")
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear( # 行切分··
            args.ffn_hidden_size,
            args.hidden_size,
            bias=self.add_bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())

    '''
    3.2.2 前向操作
    这里分别调用了 ColumnParallelLinear完成了维度从 H 到 4H 的转换，RowParallelLinear 完成了维度从 4H 到 H 的转换。
    
    ColumnParallelLinear 可以独立使用，也可以作为 ParallelMLP 的前半部分。
    它的功能是将输入从 H 维度扩展到 4H 维度，可能涉及到列并行的操作，以便在多个处理单元上并行计算。

    RowParallelLinear 也可以独立使用，也可以作为 ParallelMLP 的后半部分。
    它的功能是将输入从 4H 维度转换回到 H 维度，可能涉及到行并行的操作，以便在多个处理单元上并行计算。
    '''
    def forward(self, hidden_states):
        gd.debuginfo(prj="mt", info=f'hidden_states={infoTensor(hidden_states)}')

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            assert self.add_bias is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
            gd.debuginfo(prj="mt", info=f'1-intermediate_parallel={infoTensor(intermediate_parallel)}')
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
                gd.debuginfo(prj="mt", info=f'2-intermediate_parallel={infoTensor(intermediate_parallel)}')

            intermediate_parallel = self.activation_func(intermediate_parallel)
            gd.debuginfo(prj="mt", info=f'3-intermediate_parallel={infoTensor(intermediate_parallel)}')

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)

        gd.debuginfo(prj="mt", info=f'output={infoTensor(output)}')
        gd.debuginfo(prj="mt", info=f'output_bias={infoTensor(output_bias)}')

        return output, output_bias

class SwitchMLP(MegatronModule):
    """
    Routes input to one of N MLP "experts"
    """
    def __init__(self, init_method, output_layer_init_method):
        gd.debuginfo(prj="mt", info=f'C: {self.__class__.__name__}')
        super(SwitchMLP, self).__init__()
        args = get_args()
        self.router = torch.nn.Linear(args.hidden_size, args.num_experts)
        self.experts = torch.nn.ModuleList()
        for i in range(args.num_experts):
            self.experts.append(ParallelMLP(init_method, output_layer_init_method))

    def forward(self, hidden_states):
        gd.debuginfo(prj="mt", info=f'hidden_states={infoTensor(hidden_states)}')

        # hidden_states: [s, b, h]
        s = hidden_states.size(0)
        b = hidden_states.size(1)
        h = hidden_states.size(2)
        gd.debuginfo(prj="mt", info=f's={s}, b={b}, h={h}')

        route = self.router(hidden_states)
        gd.debuginfo(prj="mt", info=f'1-route={infoTensor(route)}')

        route = torch.nn.functional.softmax(route, dim=2)
        gd.debuginfo(prj="mt", info=f'2-route={infoTensor(route)}')

        max_prob, max_ind = torch.max(route, dim=2)
        gd.debuginfo(prj="mt", info=f'A : max_prob={max_prob}, max_ind={max_ind}')

        max_prob = torch.unsqueeze(max_prob, 2) # [s b 1]
        gd.debuginfo(prj="mt", info=f'2-max_prob={infoTensor(max_prob)}')

        # TODO (rprenger) TODO this could be made easier to read
        # Converting [s, b, h] to [s*b, h].
        # Each vector could be routed differently
        hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [s*b h]
        gd.debuginfo(prj="mt", info=f'hidden_states={infoTensor(hidden_states)}')

        max_prob = max_prob.view(-1, max_prob.size(2)) # [s*b 1]
        max_ind = max_ind.view(-1) # [s*b]
        gd.debuginfo(prj="mt", info=f'B : max_prob={max_prob}, max_ind={max_ind}')

        output_total = torch.empty_like(hidden_states)
        output_bias_total = torch.empty_like(hidden_states)
        gd.debuginfo(prj="mt", info=f'output_total={infoTensor(output_total)}')
        gd.debuginfo(prj="mt", info=f'output_bias_total={infoTensor(output_bias_total)}')
        #TODO (rprenger) This does each expert in serial, but it could be parallelized

        for expert_num, expert in enumerate(self.experts):
            gd.debuginfo(prj="mt", info=f'expert_num={expert_num}, expert={expert}')

            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices,:]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices,:] = output
            output_bias_total[local_indices,:] = output_bias

        output_total = output_total*max_prob
        gd.debuginfo(prj="mt", info=f'1-output_total={infoTensor(output_total)}')

        output_bias_total = output_bias_total*max_prob
        gd.debuginfo(prj="mt", info=f'1-output_bias_total={infoTensor(output_bias_total)}')

        output_total = output_total.view(s, b, h)
        gd.debuginfo(prj="mt", info=f'2-output_total={infoTensor(output_total)}')

        output_bias_total = output_bias_total.view(s, b, h)
        gd.debuginfo(prj="mt", info=f'2-output_bias_total={infoTensor(output_bias_total)}')

        return output_total, output_bias_total


class CoreAttention(MegatronModule):

    def __init__(self, layer_number,
                 attn_mask_type=AttnMaskType.padding):
        gd.debuginfo(prj="mt", info=f'C: {self.__class__.__name__}')
        super(CoreAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = args.sequence_parallel

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                           world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            gd.debuginfo(prj="mt")
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

    def forward(self,
                query_layer,
                key_layer,
                value_layer,
                attention_mask):

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0030')

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        gd.debuginfo(prj="mt", info=f'output_size={output_size}')

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        gd.debuginfo(prj="mt", info=f'query_layer={infoTensor(query_layer)}')

        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)
        gd.debuginfo(prj="mt", info=f'key_layer={infoTensor(key_layer)}')

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")
        gd.debuginfo(prj="mt", info=f'matmul_input_buffer={infoTensor(matmul_input_buffer)}')

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))
        gd.debuginfo(prj="mt", info=f'matmul_result={infoTensor(matmul_result)}')

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        gd.debuginfo(prj="mt", info=f'attention_scores={infoTensor(attention_scores)}')

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)
        gd.debuginfo(prj="mt", info=f'1-attention_probs={infoTensor(attention_probs)}')

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
                gd.debuginfo(prj="mt", info=f'2-attention_probs={infoTensor(attention_probs)}')
        else:
            attention_probs = self.attention_dropout(attention_probs)
            gd.debuginfo(prj="mt", info=f'3-attention_probs={infoTensor(attention_probs)}')

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))
        gd.debuginfo(prj="mt", info=f'output_size={output_size}')

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        gd.debuginfo(prj="mt", info=f'value_layer={infoTensor(value_layer)}')

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        gd.debuginfo(prj="mt", info=f'attention_probs={infoTensor(attention_probs)}')

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        gd.debuginfo(prj="mt", info=f'new_context_layer_shape={new_context_layer_shape}')

        context_layer = context_layer.view(*new_context_layer_shape)
        gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0030')
        return context_layer


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        gd.debuginfo(prj="mt", info=f'C: {self.__class__.__name__}')
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0031')

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        gd.debuginfo(prj="mt", info=f'batch_size={batch_size}, seqlen_q={seqlen_q}, seqlen_k={seqlen_k}')

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        gd.debuginfo(prj="mt", info=f'q={q}, k={k}, v={v}')

        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
        gd.debuginfo(prj="mt", info=f'cu_seqlens_q={infoTensor(cu_seqlens_q)}')

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            gd.debuginfo(prj="mt", info=f'is_causal={is_causal}, cu_seqlens_k={cu_seqlens_k}')
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            self.dropout_p = 0

            gd.debuginfo(prj="mt", info=f'is_causal={is_causal}, cu_seqlens_k={cu_seqlens_k}')

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            self.dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        gd.debuginfo(prj="mt", info=f'a-output={output}')

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)

        gd.debuginfo(prj="mt", info=f'b-output={output}')

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0031')

        return output


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        gd.debuginfo(prj="mt", info=f'C: {self.__class__.__name__}')
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype
        self.sequence_parallel = args.sequence_parallel

        self.use_flash_attn = args.use_flash_attn \
            and attention_type == AttnType.self_attn \
            and self.attn_mask_type == AttnMaskType.causal

        if self.use_flash_attn:
            gd.debuginfo(prj="mt")
            if flash_attn_unpadded_func is None:
                raise ImportError('FlashAttention is not installed, please install with '
                                  'pip install flash-attn')
            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            gd.debuginfo(prj="mt")
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                bias=args.add_bias_linear,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())
        else:
            gd.debuginfo(prj="mt")
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                bias=args.add_bias_linear,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())


            self.key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                bias=args.add_bias_linear,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())

        self.core_attention = CoreAttention(self.layer_number,
                                            self.attn_mask_type)
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=args.attention_dropout
            )

        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            args.hidden_size,
            bias=args.add_bias_linear,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask,
                                        rotary_pos_emb=None):
        gd.debuginfo(prj="mt")
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            gd.debuginfo(prj="mt")
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
            return output_

        q_pos_emb, k_pos_emb = (None, None) if rotary_pos_emb is None \
            else rotary_pos_emb

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask,
            q_pos_emb, k_pos_emb)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        gd.debuginfo(prj="mt")
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0032')

        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        is_first_step = False

        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (inference_key_memory, inference_value_memory)
                gd.debuginfo(prj="mt", info=f'inf_max_seq_len={inf_max_seq_len}, '
                                            f'inf_max_batch_size={inf_max_batch_size}')
                gd.debuginfo(prj="mt",
                             info=f'inference_key_memory={infoTensor(inference_key_memory)}, '
                                  f'inference_value_memory={infoTensor(inference_value_memory)}')
                is_first_step = True
            else:
                inference_key_memory, inference_value_memory = inference_params.key_value_memory_dict[self.layer_number]
                gd.debuginfo(prj="mt",
                             info=f'inference_key_memory={infoTensor(inference_key_memory)}, '
                                  f'inference_value_memory={infoTensor(inference_value_memory)}')

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            gd.debuginfo(prj="mt", info=f'1-mixed_x_layer={infoTensor(mixed_x_layer)}')

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            gd.debuginfo(prj="mt", info=f'new_tensor_shape={new_tensor_shape}')

            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
            gd.debuginfo(prj="mt", info=f'2-mixed_x_layer={infoTensor(mixed_x_layer)}')

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
            gd.debuginfo(prj="mt",
                         info=f'query_layer={infoTensor(query_layer)}, '
                              f'key_layer={infoTensor(key_layer)}, '
                              f'value_layer={infoTensor(value_layer)}')
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)
            gd.debuginfo(prj="mt", info=f'1-mixed_kv_layer={mixed_kv_layer}')

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            gd.debuginfo(prj="mt", info=f'1-new_tensor_shape={new_tensor_shape}')

            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
            gd.debuginfo(prj="mt", info=f'2-mixed_kv_layer={mixed_kv_layer}')

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)

            gd.debuginfo(prj="mt",
                         info=f'query_layer={infoTensor(query_layer)}, '
                              f'key_layer={infoTensor(key_layer)}, '
                              f'value_layer={infoTensor(value_layer)}')

            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            gd.debuginfo(prj="mt", info=f'2-new_tensor_shape={new_tensor_shape}')

            query_layer = query_layer.view(*new_tensor_shape)
            gd.debuginfo(prj="mt", info=f'2-query_layer={infoTensor(query_layer)}')

        # ==================================
        # Adjust key and value for inference
        # ==================================

        # duplicate the pos_emb for self attention
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
                gd.debuginfo(prj="mt", info=f'rotary_pos_emb={rotary_pos_emb}')
            else:
                rotary_pos_emb = ((rotary_pos_emb,) * 2)
                gd.debuginfo(prj="mt", info=f'rotary_pos_emb={rotary_pos_emb}')

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            gd.debuginfo(prj="mt", info=f'batch_start={batch_start}, batch_end={batch_end}')
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)

            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer

            key_layer = inference_key_memory[
                :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...]

            gd.debuginfo(prj="mt",
                         info=f'key_layer={infoTensor(key_layer)}, '
                              f'value_layer={infoTensor(value_layer)}')


            # adjust the key rotary positional embedding
            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb
                gd.debuginfo(prj="mt", info=f'q_pos_emb={q_pos_emb}, k_pos_emb={k_pos_emb}')
                # need to cross check this condition during inference
                # if not set_inference_key_value_memory:
                if not is_first_step:
                    # In inference, we compute one token at a time.
                    # Select the correct positional embedding
                    # (only the last token in the sequence)
                    q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
                    gd.debuginfo(prj="mt", info=f'q_pos_emb={q_pos_emb}')
                else:
                    # In the first forward pass of inference,
                    # we use the entire provided prefix.
                    # q_pos_emb here has the rope embeddings of the entire
                    # prefix + to-be-generated output so
                    # we slice to just the prefix.
                    q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
                    gd.debuginfo(prj="mt", info=f'q_pos_emb={q_pos_emb}')

                k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
                rotary_pos_emb = (q_pos_emb, k_pos_emb)
                gd.debuginfo(prj="mt", info=f'q_pos_emb={q_pos_emb}')
                gd.debuginfo(prj="mt", info=f'rotary_pos_emb={rotary_pos_emb}')


        # ==================================
        # core attention computation
        # ==================================

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            gd.debuginfo(prj="mt", info=f'q_pos_emb={q_pos_emb}')
            gd.debuginfo(prj="mt", info=f'k_pos_emb={k_pos_emb}')

            query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
            gd.debuginfo(prj="mt", info=f'query_layer={infoTensor(query_layer)}')
            gd.debuginfo(prj="mt", info=f'key_layer={infoTensor(key_layer)}')

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        if not self.use_flash_attn:
            if self.checkpoint_core_attention:
                context_layer = self._checkpointed_attention_forward(
                    query_layer, key_layer, value_layer, attention_mask)
                gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')
            else:
                context_layer = self.core_attention(
                    query_layer, key_layer, value_layer, attention_mask)
                gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')
        else:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]

            gd.debuginfo(prj="mt", info=f'q={q}, k={k}, v={v}')

            if not self.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(q, k, v)
                    gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')
            else:
                context_layer = self.core_attention_flash(q, k, v)
                gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')

            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
            gd.debuginfo(prj="mt", info=f'4-context_layer={infoTensor(context_layer)}')

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)
        gd.debuginfo(prj="mt", info=f'output={infoTensor(output)}')
        gd.debuginfo(prj="mt", info=f'bias={infoTensor(bias)}')

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0032')

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # gd.debuginfo(prj="mt") 被jit调用，不能用

    # type: (Tensor, Optional[Tensor], Tensor, float, bool) -> Tensor
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    gd.debuginfo(prj="mt")
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    # gd.debuginfo(prj="mt") jit 不能用

    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: Optional[torch.Tensor],
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    # gd.debuginfo(prj="mt") jit 不能用
    return bias_dropout_add(x, bias, residual, prob, False)

'''
2.1 初始化
ParallelTransformerLayer地址：/Megatron-LM/megatron/model/transformer.py

其初始化方法流程如下：

    生成一个 LayerNorm 处理输入数据。
    生成并行 Attention。
    生成处理 Attention 输出的 LayerNorm。
    如果是 decoder，则生成一个 ParallelAttention。
    生成一个并行 MLP。
    
如下图：    30.png
'''
class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
                 # retriever=None):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0036')
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        # 通过 LayerNorm 处理输入数据
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)

        # Self attention.
        # 并行 Attention
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        # 生成处理attention输出的LayerNorm
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p)

        # Cross attention.
        # 如果是 decoder
        if self.layer_type in (LayerType.decoder,
                               LayerType.retro_decoder,
                               LayerType.retro_decoder_with_retriever,
                               LayerType.retro_encoder):
            gd.debuginfo(prj="mt")

            # 则生成一个ParallelAttention
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel,
                apply_layernorm_1p=args.apply_layernorm_1p)

        # MLP
        # 生成一个并行MLP
        if args.num_experts is not None:
            gd.debuginfo(prj="mt")
            self.mlp = SwitchMLP(init_method, output_layer_init_method)
        else:
            gd.debuginfo(prj="mt")
            self.mlp = ParallelMLP(init_method, output_layer_init_method)

        # Set bias+dropout+add fusion grad_enable execution handler.
        # 算子融合，加速用的
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

        if args.retro_add_retriever:
            gd.debuginfo(prj="mt")
            retro_args = get_retro_args()
            self.retro_num_neighbors = args.retro_num_neighbors
            self.retro_chunk_length = retro_args.retro_gpt_chunk_length
            self.retro_retrieved_length = retro_args.retro_gpt_retrieved_length

        # Retriever (bi-directional transformer with cross attention)
        if layer_type == LayerType.retro_decoder_with_retriever:
            gd.debuginfo(prj="mt")
            self.retriever = ParallelTransformer(
                init_method,
                output_layer_init_method,
                model_type=ModelType.retro_encoder,
                self_attn_mask_type=AttnMaskType.padding,
                pre_process=True,
                post_process=False,
            )
            self._retriever_key = 'retriever'
        else:
            gd.debuginfo(prj="mt")
            self.retriever = None

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0036')

    def default_decoder_cross_attention(self,
                                        encoder_output,
                                        enc_dec_attn_mask,
                                        layernorm_input,
                                        layernorm_output,
                                        bias_dropout_add_func):
        '''Cross attention for a standard encoder-decoder model.'''

        # Attention.
        attention_output, attention_bias = self.inter_attention(layernorm_output, enc_dec_attn_mask, encoder_output=encoder_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            gd.debuginfo(prj="mt")
            residual = layernorm_output
        else:
            gd.debuginfo(prj="mt")
            residual = layernorm_input

        if attention_bias is not None:
            gd.debuginfo(prj="mt")
            attention_bias = attention_bias.expand_as(residual)

        # Bias-dropout-add.
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias,
                residual,
                self.hidden_dropout)

        # Layer norm.
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        return layernorm_input, layernorm_output

    def retro_encoder_cross_attention(self,
                                      retriever_output,
                                      layernorm_input,
                                      layernorm_output,
                                      bias_dropout_add_func):
        """Cross attention for Retro encoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """
        gd.debuginfo(prj="mt")

        ns, bs, d = layernorm_output.shape # [r, bs * l * k, d]

        # Divide sequence dimension into chunks.
        chunked_outputs = layernorm_output.reshape(self.retro_retrieved_length,
                                                   -1,
                                                   self.retro_num_neighbors,
                                                   d)
        chunked_outputs_before_layer_norm = \
            layernorm_input.reshape(self.retro_retrieved_length, -1,
                                    self.retro_num_neighbors, d) # [r, bs*l, k, d]

        # Per-chunk attention.
        layernorm_inputs = []
        layernorm_outputs = []
        for k in range(self.retro_num_neighbors):

            # Attention.
            chunked_output = chunked_outputs[:,:,k].contiguous()
            attention_output, attention_bias = \
                self.inter_attention(
                    chunked_output, # Q (neighbor embedding)
                    None,
                    encoder_output=retriever_output) # K, V (hidden act)

            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                gd.debuginfo(prj="mt")
                residual = chunked_output
            else:
                gd.debuginfo(prj="mt")
                residual = chunked_outputs_before_layer_norm[:,:,k]

            # Re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    None if attention_bias is None else attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
                layernorm_inputs.append(layernorm_input)

            # Layer norm.
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)
            layernorm_outputs.append(layernorm_output)

        # Concatenate layer norms.
        # layernorm_input : [r, k * bs * l, d]
        # layernorm_output : [r, k * bs * l, d]
        layernorm_input = torch.stack(layernorm_inputs, dim=1).reshape(ns, bs, d)
        layernorm_output = torch.stack(layernorm_outputs, dim=1).reshape(ns, bs, d)

        return layernorm_input, layernorm_output

    def retro_decoder_cross_attention(self,
                                      retriever_input,
                                      retriever_output,
                                      retriever_attn_mask,
                                      layernorm_input,
                                      layernorm_output,
                                      inference_params,
                                      bias_dropout_add_func):
        """Cross attention for Retro decoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            m  : Number of tokens per chunk.
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """
        gd.debuginfo(prj="mt")

        ns, bs, d = layernorm_output.shape
        l = int(np.ceil(ns / self.retro_chunk_length))

        # Retrieve neighbors.
        if self.layer_type == LayerType.retro_decoder_with_retriever:
            gd.debuginfo(prj="mt")
            first_ns = ns % self.retro_chunk_length
            if first_ns > 0:
                raise Exception("test this case.")
                gd.debuginfo(prj="mt")

                first_chunk, rest_chunk = layernorm_output[:first_ns], layernorm_output[first_ns:]
                first_chunk = torch.nn.functional.pad(
                    first_chunk,
                    (0, 0, 0, 0, 0, self.retro_chunk_length - first_ns),
                    'constant',
                    0)
                chunked_output = \
                    torch.cat((first_chunk, rest_chunk), dim=0) # [l * m, bs, d]
            else:
                chunked_output = layernorm_output # [l * m, bs, d]
                gd.debuginfo(prj="mt")

            chunked_output = chunked_output \
                .reshape(l, self.retro_chunk_length, bs, d) \
                .permute(1, 2, 0, 3) \
                .reshape(self.retro_chunk_length, bs * l, d) \
                .contiguous()

            # Get Encoder Output
            retriever_output = self.retriever(
                hidden_states=retriever_input,
                attention_mask=retriever_attn_mask,
                retriever_output=chunked_output,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params) # [r, k * bs * l , d]
            retriever_output = retriever_output.reshape(
                self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d) # [r * k, bs * l, d]

        # Chunks.
        pad = (ns - 1) % self.retro_chunk_length
        attending_chunks = layernorm_output[pad:]
        padded_chunks = torch.nn.functional.pad(
            attending_chunks,
            (0, 0, 0, 0, 0, self.retro_chunk_length - 1),
            'constant', 0)
        padded_chunked_output = padded_chunks \
            .reshape(l, self.retro_chunk_length, bs, d) \
            .permute(1, 2, 0, 3)
        padded_chunked_output = padded_chunked_output.reshape(
            self.retro_chunk_length, bs * l, d).contiguous()

        # Encoder output.
        attention_output, attention_bias = \
            self.inter_attention(padded_chunked_output,
                                 None,
                                 encoder_output=retriever_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            gd.debuginfo(prj="mt")
            residual = layernorm_output
        else:
            gd.debuginfo(prj="mt")
            residual = layernorm_input

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                None if attention_bias is None else attention_bias.expand_as(attention_output),
                torch.zeros_like(attention_output),
                self.hidden_dropout)
            layernorm_input = layernorm_input \
                .reshape(self.retro_chunk_length, bs, l, d) \
                .permute(2, 0, 1, 3) # [l, m, bs, d]
            layernorm_input = layernorm_input.reshape(self.retro_chunk_length * l, bs, d)
            layernorm_input = torch.nn.functional.pad(
                layernorm_input,
                (0, 0, 0, 0, pad, 0),
                'constant', 0)[:ns] # [ns, b, d]
            layernorm_input = layernorm_input + residual

        # Layer norm post the decoder attention
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        return retriever_output, layernorm_input, layernorm_output

    '''
    4.2.3.2 前向传播
    要就是调用内部 ParallelTransformerLayer 的 forward 方法。
    和 BERT 模型不同的是，GPTModel 这里第一层和最后一层没有进行特殊处理。
    '''
    '''
    2.2 前向传播
        ParallelTransformerLayer 的前向传播方法如下，即调用各种成员函数进行前向操作。
    '''
    def forward(self,
                hidden_states,
                attention_mask,
                encoder_output=None,
                enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0001')
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        gd.debuginfo(prj="mt", info=f'layernorm_output={infoTensor(layernorm_output)}')

        # Self attention.
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb)

        gd.debuginfo(prj="mt", info=f'attention_output={infoTensor(attention_output)}')
        gd.debuginfo(prj="mt", info=f'attention_bias={infoTensor(attention_bias)}')

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
            gd.debuginfo(prj="mt", info=f'layernorm_output={infoTensor(layernorm_output)}')
        else:
            residual = hidden_states
            gd.debuginfo(prj="mt", info=f'hidden_states={infoTensor(hidden_states)}')

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    gd.debuginfo(prj="mt")
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    gd.debuginfo(prj="mt")
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                gd.debuginfo(prj="mt")
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            if attention_bias is not None:
                attention_bias = attention_bias.expand_as(residual)
                gd.debuginfo(prj="mt", info=f'attention_bias={infoTensor(attention_bias)}')

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias,
                    residual,
                    self.hidden_dropout)

                gd.debuginfo(prj="mt", info=f'layernorm_input={infoTensor(layernorm_input)}')
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            gd.debuginfo(prj="mt", info=f'out={infoTensor(out)}')

            layernorm_input = residual + self.drop_path(out)
            gd.debuginfo(prj="mt", info=f'layernorm_input={infoTensor(layernorm_input)}')

        # Layer norm post the self attention.
        # 处理attention输出
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        gd.debuginfo(prj="mt", info=f'layernorm_output={infoTensor(layernorm_output)}')

        # Cross attention.
        if self.layer_type == LayerType.encoder:
            gd.debuginfo(prj="mt")
            pass
        elif self.layer_type == LayerType.decoder:
            layernorm_input, layernorm_output = \
                self.default_decoder_cross_attention(
                    encoder_output,
                    enc_dec_attn_mask,
                    layernorm_input,
                    layernorm_output,
                    bias_dropout_add_func)
            gd.debuginfo(prj="mt", info=f'layernorm_input={infoTensor(layernorm_input)}')
            gd.debuginfo(prj="mt", info=f'layernorm_output={infoTensor(layernorm_output)}')
        elif self.layer_type == LayerType.retro_encoder:
            layernorm_input, layernorm_output = \
                self.retro_encoder_cross_attention(
                    retriever_output,
                    layernorm_input,
                    layernorm_output,
                    bias_dropout_add_func)
            gd.debuginfo(prj="mt", info=f'layernorm_input={infoTensor(layernorm_input)}')
            gd.debuginfo(prj="mt", info=f'layernorm_output={infoTensor(layernorm_output)}')
        elif self.layer_type in (LayerType.retro_decoder,
                                 LayerType.retro_decoder_with_retriever):
            retriever_output, layernorm_input, layernorm_output = \
                self.retro_decoder_cross_attention(
                    retriever_input,
                    retriever_output,
                    retriever_attn_mask,
                    layernorm_input,
                    layernorm_output,
                    inference_params,
                    bias_dropout_add_func)
            gd.debuginfo(prj="mt", info=f'retriever_output={infoTensor(retriever_output)}')
            gd.debuginfo(prj="mt", info=f'layernorm_output={infoTensor(layernorm_output)}')
            gd.debuginfo(prj="mt", info=f'layernorm_input={infoTensor(layernorm_input)}')
        else:
            raise Exception("Unsupported layer type, '%s'." %
                            self.layer_type.name)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        gd.debuginfo(prj="mt", info=f'mlp_output={infoTensor(mlp_output)}')
        gd.debuginfo(prj="mt", info=f'mlp_bias={infoTensor(mlp_bias)}')

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
            gd.debuginfo(prj="mt", info=f'layernorm_output={infoTensor(layernorm_output)}')
        else:
            residual = layernorm_input
            gd.debuginfo(prj="mt", info=f'layernorm_input={infoTensor(layernorm_input)}')

        if self.drop_path is None:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
                gd.debuginfo(prj="mt", info=f'mlp_bias={infoTensor(mlp_bias)}')

            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias,
                    residual,
                    self.hidden_dropout)
                gd.debuginfo(prj="mt", info=f'output={infoTensor(output)}')

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = core.utils.make_viewless_tensor(inp = output,
                                                     requires_grad = output.requires_grad,
                                                     keep_graph = True)
            gd.debuginfo(prj="mt", info=f'output={infoTensor(output)}')

        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
                gd.debuginfo(prj="mt", info=f'mlp_output={infoTensor(mlp_output)}')

            out = torch.nn.functional.dropout(mlp_output,
                                              p=self.hidden_dropout,
                                              training=self.training)
            gd.debuginfo(prj="mt", info=f'out={infoTensor(out)}')

            output = residual + self.drop_path(out)
            gd.debuginfo(prj="mt", info=f'output={infoTensor(output)}')

        if self.layer_type == LayerType.retro_decoder_with_retriever:
            gd.debuginfo(prj="mt", info=f'output={infoTensor(output)}')
            gd.debuginfo(prj="mt", info=f'retriever_output={infoTensor(retriever_output)}')
            return output, retriever_output
        else:
            gd.debuginfo(prj="mt", info=f'output={infoTensor(output)}')
            return output

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0001')


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        gd.debuginfo(prj="mt")
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        gd.debuginfo(prj="mt")
        return hidden_states.clone()


def _get_num_layers(args, model_type, is_decoder=False):
    gd.debuginfo(prj="mt")

    """Compute the number of transformer layers resident on the current rank."""
    is_encoder_and_decoder_model = (model_type == ModelType.encoder_and_decoder)

    if model_type == ModelType.retro_encoder:
        gd.debuginfo(prj="mt")
        num_layers = args.retro_encoder_layers
    elif mpu.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            gd.debuginfo(prj="mt")
            assert args.pipeline_model_parallel_split_rank is not None

            # When a standalone embedding stage is used, a rank is taken from
            # the encoder's ranks, to be used for the encoder's embedding
            # layer. This way, the rank referenced by the 'split rank' remains
            # the same whether or not a standalone embedding stage is used.
            num_ranks_in_encoder = (
                args.pipeline_model_parallel_split_rank - 1
                if args.standalone_embedding_stage else
                args.pipeline_model_parallel_split_rank
            )
            num_ranks_in_decoder = args.transformer_pipeline_model_parallel_size - num_ranks_in_encoder
            assert args.encoder_num_layers % num_ranks_in_encoder == 0, \
                    'encoder_num_layers (%d) must be divisible by number of ranks given to encoder (%d)' % (args.encoder_num_layers, num_ranks_in_encoder)
            assert args.decoder_num_layers % num_ranks_in_decoder == 0, \
                    'decoder_num_layers (%d) must be divisible by number of ranks given to decoder (%d)' % (args.decoder_num_layers, num_ranks_in_decoder)
            if mpu.is_pipeline_stage_before_split():
                gd.debuginfo(prj="mt")
                num_layers = (
                    0
                    if args.standalone_embedding_stage
                    and mpu.get_pipeline_model_parallel_rank() == 0 else
                    args.encoder_num_layers // num_ranks_in_encoder
                )
            else:
                gd.debuginfo(prj="mt")
                num_layers = args.decoder_num_layers // num_ranks_in_decoder
        else:
            gd.debuginfo(prj="mt")
            assert args.num_layers == args.encoder_num_layers
            assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'num_layers must be divisible by transformer_pipeline_model_parallel_size'

            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if args.standalone_embedding_stage
                and mpu.get_pipeline_model_parallel_rank() == 0 else
                args.num_layers // args.transformer_pipeline_model_parallel_size
            )
    else:
        if not is_decoder:
            gd.debuginfo(prj="mt")
            num_layers = args.encoder_num_layers
        else:
            gd.debuginfo(prj="mt")
            num_layers = args.decoder_num_layers
    return num_layers


def _get_layer_type(model_type, default_layer_type, retro_layer_numbers,
                    layer_number):
    args = get_args()
    if args.retro_add_retriever and layer_number in retro_layer_numbers:
        if model_type == ModelType.retro_decoder:
            gd.debuginfo(prj="mt")
            return LayerType.retro_decoder_with_retriever \
                if layer_number == retro_layer_numbers[0] \
                   else LayerType.retro_decoder
        elif model_type == ModelType.retro_encoder:
            gd.debuginfo(prj="mt")
            return LayerType.retro_encoder
        else:
            raise Exception("Unsupported model type, '%s'." % model_type)
    else:
        gd.debuginfo(prj="mt")
        return default_layer_type

'''
4.2.3 ParallelTransformer
这里会调用 ParallelTransformerLayer 生成具体的 Transformer 层，我们会在后文中进行分析。
即，ParallelTransformer 包括多个 Transformer，其中每层 Transformer 是一个 ParallelTransformerLayer。
args.transformer-impl 表示采用哪种 Transformer 的实现方式：
transformer_engine：
https://github.com/NVIDIA/TransformerEngine 的实现版本，包含一些NVIDIA家量化加速的技术
2. local：本地实现的简单版本
这里我们主要研究 local 模式


逻辑上的嵌套表示图如下：
24.png
'''
'''
9. 如何把模型分到GPU
实际上，不是通过将模型部分复制到特定的rank或 GPU 上来实现的，而是每个rank或GPU主动获取自己对应的层。

通过调用mpu.initialize_model_parallel来设置模型并行、数据并行等各种进程组，
因此，每个 rank 对应的进程都有自己的全局变量，实际上进程会自动映射到 GPU 上。
例如，rank 2对应的进程在启动后才知道自己是rank 2，然后从初始化的全局变量中了解自己的数据并行组是[g0, g2]，
张量模型并行组是[g2, g3]，流水线模型并行组是[g2, g6, g10, g14]。

在ParallelTransformer的初始化中，偏移量（offset）根据 rank决定应该生成哪些层，
然后通过
self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])
来生成相应的层。

在get_model方法中，根据流水线的rank和is_pipeline_first_stage，确定是否是第一层或最后一层，并做出相应的处理。

最终，将模型参数复制到自己相应的 GPU 上。
这种方式让每个进程主动获取其应该处理的部分，实现了模型的分块和放置。

具体可以参考 ParallelTransformer 的初始化代码部分：
'''
class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 model_type, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0034')
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.layer_type = layer_type
        self.model_type = model_type
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = args.transformer_impl
        self.retro_add_retriever = args.retro_add_retriever

        # Store activation checkpoiting flag.
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = \
            args.distribute_saved_activations and not args.sequence_parallel

        self.sequence_parallel = args.sequence_parallel

        # Transformer Engine Init.
        self.transformer_engine_rope_available = False
        if self.transformer_impl == 'transformer_engine':
            gd.debuginfo(prj="mt")
            global transformer_engine
            import transformer_engine
            from importlib.metadata import version
            from pkg_resources import packaging

            te_version = packaging.version.Version(version("transformer-engine"))
            if te_version >= packaging.version.Version("0.10.0"):
                self.transformer_engine_rope_available = True

            del version, packaging

        self.use_fp8 = args.fp8_e4m3 or args.fp8_hybrid
        self.fp8_recipe = None
        self.fp8_group = None
        if self.use_fp8:
            self.fp8_group = mpu.get_data_parallel_group()
            if args.fp8_e4m3:
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif args.fp8_hybrid:
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=args.fp8_margin,
                interval=args.fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=args.fp8_amax_history_len,
                amax_compute_algo=args.fp8_amax_compute_algo,
                override_linear_precision=(False, False, not args.fp8_wgrad),
            )

        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        # Number of layers.
        # 获得本Transformer的具体层数
        self.num_layers = _get_num_layers(args, model_type,
                                          layer_type==LayerType.decoder)

        gd.debuginfo(prj="mt", info=f'self.num_layers={self.num_layers}')
        #
        # if self.num_layers is None:
        #     self.num_layers = 0

        self.drop_path_rates = [
            rate.item() for rate in
            torch.linspace(0, self.drop_path_rate, args.num_layers)]

        self.retro_layer_numbers = None
        if model_type == ModelType.retro_decoder:
            retro_layer_start = 6 if args.num_layers <= 15 else 9
            self.retro_layer_numbers = \
                np.arange(retro_layer_start, args.num_layers + 1, 3).tolist()
        if model_type == ModelType.retro_encoder:
            self.retro_layer_numbers = [1]

        # Transformer layers.
        if args.retro_add_retriever:
            assert self.recompute_granularity != 'full', \
                "Full recompute not supported for Retro."
            assert args.transformer_impl == 'local', \
                "Transformer engine does not support Retro layers."

        def build_layer(layer_number):
            if args.transformer_impl == 'local':
                gd.debuginfo(prj="mt")
                current_layer_type = _get_layer_type(
                    model_type, layer_type, self.retro_layer_numbers,
                    layer_number)
                return ParallelTransformerLayer(   # 返回一层 Transformmer
                    init_method,
                    output_layer_init_method,
                    layer_number,
                    layer_type=current_layer_type,
                    self_attn_mask_type=self_attn_mask_type,
                    drop_path_rate=self.drop_path_rates[layer_number - 1])
            else:
                gd.debuginfo(prj="mt")
                return transformer_engine.pytorch.TransformerLayer(
                    args.hidden_size,
                    args.ffn_hidden_size,
                    args.num_attention_heads,
                    layernorm_epsilon=args.layernorm_epsilon,
                    hidden_dropout=args.hidden_dropout,
                    attention_dropout=args.attention_dropout,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    layer_number=layer_number,
                    kv_channels=args.kv_channels,
                    self_attn_mask_type=self_attn_mask_type.name,
                    tp_group=mpu.get_tensor_model_parallel_group(),
                    get_rng_state_tracker=tensor_parallel.get_cuda_rng_tracker,
                    fuse_wgrad_accumulation=args.gradient_accumulation_fusion,
                    apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
                    attention_softmax_in_fp32=args.attention_softmax_in_fp32,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    sequence_parallel=args.sequence_parallel,
                    params_dtype=args.params_dtype,
                    apply_residual_connection_post_layernorm=args.apply_residual_connection_post_layernorm,
                    output_layernorm=False,
                    layer_type="encoder",
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    set_parallel_mode=True,
                    fuse_qkv_params=True)

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size

            gd.debuginfo(prj="mt", info=f'self.num_layers={self.num_layers}')

            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and mpu.get_pipeline_model_parallel_world_size() > 1:
                gd.debuginfo(prj="mt")
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                    gd.debuginfo(prj="mt", info=f'offset={offset}')
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
                    gd.debuginfo(prj="mt", info=f'offset={offset}')
            else:
                gd.debuginfo(prj="mt", info=f'self.num_layers={self.num_layers}, '
                                            f'mpu.get_pipeline_model_parallel_rank()={mpu.get_pipeline_model_parallel_rank()}')
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers
                gd.debuginfo(prj="mt", info=f'offset={offset}')

        if self.num_layers == 0:

            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
            gd.debuginfo(prj="mt", info=f'self.layers={self.layers}')
        else:
            self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)])
            gd.debuginfo(prj="mt", info=f'self.layers={self.layers}')

            # Update dropout rate for Retro encoder.
            if model_type == ModelType.retro_encoder:
                gd.debuginfo(prj="mt")
                for layer in self.layers:
                    if layer.self_attention.use_flash_attn:
                        gd.debuginfo(prj="mt")
                        layer.self_attention.core_attention_flash.dropout_p = \
                            torch.nn.Dropout(args.retro_encoder_attention_dropout)
                    else:
                        gd.debuginfo(prj="mt")
                        layer.self_attention.core_attention.attention_dropout.p =\
                            args.retro_encoder_attention_dropout
                    layer.hidden_dropout = args.retro_encoder_hidden_dropout

        if self.post_process and self.post_layer_norm:
            gd.debuginfo(prj="mt")
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel,
                apply_layernorm_1p=args.apply_layernorm_1p)
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0034')

    def _get_layer(self, layer_number):
        gd.debuginfo(prj="mt")
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask,
                              rotary_pos_emb, is_first_microbatch):
        gd.debuginfo(prj="mt")

        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_
            return custom_forward

        te_forward_kwargs = {}
        if self.transformer_impl == 'transformer_engine':
            gd.debuginfo(prj="mt")
            te_forward_kwargs['is_first_microbatch'] = is_first_microbatch
            if self.transformer_engine_rope_available:
                gd.debuginfo(prj="mt")
                te_forward_kwargs['rotary_pos_emb'] = rotary_pos_emb

        if self.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                if self.transformer_impl == 'transformer_engine':
                    gd.debuginfo(prj="mt")
                    hidden_states = transformer_engine.pytorch.distributed.checkpoint(
                        custom(l, l + self.recompute_num_layers),
                        self.distribute_saved_activations,
                        tensor_parallel.get_cuda_rng_tracker,
                        mpu.get_tensor_model_parallel_group(),
                        hidden_states, attention_mask, encoder_output,
                        enc_dec_attn_mask, **te_forward_kwargs)
                else:
                    gd.debuginfo(prj="mt")
                    hidden_states = tensor_parallel.checkpoint(
                        custom(l, l + self.recompute_num_layers),
                        self.distribute_saved_activations,
                        hidden_states, attention_mask,
                        encoder_output, enc_dec_attn_mask,
                        None, None, None, None, rotary_pos_emb)

                l += self.recompute_num_layers

        elif self.recompute_method == 'block':
            gd.debuginfo(prj="mt")
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    if self.transformer_impl == 'transformer_engine':
                        gd.debuginfo(prj="mt")
                        hidden_states = transformer_engine.pytorch.distributed.checkpoint(
                            custom(l, l + 1),
                            self.distribute_saved_activations,
                            tensor_parallel.get_cuda_rng_tracker,
                            mpu.get_tensor_model_parallel_group(),
                            hidden_states, attention_mask, encoder_output,
                            enc_dec_attn_mask, **te_forward_kwargs)
                    else:
                        gd.debuginfo(prj="mt")
                        hidden_states = tensor_parallel.checkpoint(
                            custom(l, l + 1),
                            self.distribute_saved_activations,
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb)
                else:
                    if self.transformer_impl == 'transformer_engine':
                        gd.debuginfo(prj="mt")
                        hidden_states = custom(l, l + 1)(
                            hidden_states, attention_mask, encoder_output,
                            enc_dec_attn_mask, **te_forward_kwargs)
                    else:
                        gd.debuginfo(prj="mt")
                        hidden_states = custom(l, l + 1)(
                            hidden_states, attention_mask,
                            encoder_output, enc_dec_attn_mask,
                            None, None, None, None, rotary_pos_emb)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0035')

        # hidden_states: [s, b, h]

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        # RNG context.
        if self.sequence_parallel:
            gd.debuginfo(prj="mt")
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            gd.debuginfo(prj="mt")
            rng_context = nullcontext()

        # Forward layers.
        with rng_context:
            # The fp8_autocast context manager is a no-op when enabled=True
            # The if...else serves to short circuit name resolution for fp8_autocast
            with transformer_engine.pytorch.fp8_autocast(
                enabled=self.use_fp8,
                fp8_recipe=self.fp8_recipe,
                fp8_group=self.fp8_group
            ) if self.use_fp8 else nullcontext():
                # Determine if the current iteration is first microbatch
                if self.num_microbatches_in_previous_step != get_num_microbatches():
                    self.microbatch_count = 0 # Reset count on new batch size rampup interval
                self.num_microbatches_in_previous_step = get_num_microbatches()
                is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

                # Forward pass.
                if self.recompute_granularity == 'full':
                    hidden_states = self._checkpointed_forward(hidden_states,
                                                               attention_mask,
                                                               encoder_output,
                                                               enc_dec_attn_mask,
                                                               rotary_pos_emb,
                                                               is_first_microbatch)
                else:
                    gd.debuginfo(prj="mt")
                    forward_kwargs = {
                        'encoder_output': encoder_output,
                        'enc_dec_attn_mask': enc_dec_attn_mask,
                        'inference_params': inference_params,
                    }

                    if self.transformer_impl == 'transformer_engine':
                        gd.debuginfo(prj="mt")
                        forward_kwargs['is_first_microbatch'] = is_first_microbatch
                        forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention
                        if self.transformer_engine_rope_available:
                            forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                    else:
                        gd.debuginfo(prj="mt")
                        forward_kwargs['rotary_pos_emb'] = rotary_pos_emb
                        forward_kwargs['retriever_input'] = retriever_input
                        forward_kwargs['retriever_output'] = retriever_output
                        forward_kwargs['retriever_attn_mask'] = retriever_attn_mask

                    for index in range(self.num_layers):
                        layer = self._get_layer(index)
                        gd.debuginfo(prj="mt", info=f'layer={layer}')
                        hidden_states = layer(
                            hidden_states,
                            attention_mask,
                            **forward_kwargs)

                        # First Retro decoder layer returns both hidden_states
                        # and retriever_output. Make retriever_output available
                        # to subsequence Retro layers.
                        if isinstance(hidden_states, tuple):
                            assert len(hidden_states) == 2
                            hidden_states, retriever_output = hidden_states
                            forward_kwargs["retriever_output"] = retriever_output

                # Skip counter update for eval and activation checkpointing
                if torch.is_grad_enabled() and self.training:
                    gd.debuginfo(prj="mt")
                    self.microbatch_count += 1

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            gd.debuginfo(prj="mt")
            hidden_states = self.final_layernorm(hidden_states)

        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__0035')

        return hidden_states
