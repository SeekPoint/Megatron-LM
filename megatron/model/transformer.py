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

"""Transformer."""
import math
from contextlib import nullcontext
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model import LayerNorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu

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
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # Project to 4h.
        self.dense_h_to_4h = mpu.ColumnParallelLinear( # 列切分
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False, # 这里是false，采用第二种方案
            init_method=init_method,
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion  # gelu
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear( # 行切分
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)

    '''
    2.2.2 前向操作
    这里分别调用了 ColumnParallelLinear 完成了 H 到 4H 的转换，RowParallelLinear 完成了 4H 到 H 的转换
    '''
    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states) # 纵向切分

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel) # 横向切分
        return output, output_bias

class SwitchMLP(MegatronModule):
    """
    Routes input to one of N MLP "experts"
    """
    def __init__(self, init_method, output_layer_init_method):
        super(SwitchMLP, self).__init__()
        args = get_args()
        self.router = torch.nn.Linear(args.hidden_size, args.num_experts)
        self.experts = torch.nn.ModuleList()
        for i in range(args.num_experts):
            self.experts.append(ParallelMLP(init_method, output_layer_init_method))

    def forward(self, hidden_states):
        # hidden_states: [b, s, h]
        b = hidden_states.size(0)
        s = hidden_states.size(1)
        h = hidden_states.size(2)
        route = self.router(hidden_states)
        route = torch.nn.functional.softmax(route, dim=2)
        max_prob, max_ind = torch.max(route, dim=2)
        max_prob = torch.unsqueeze(max_prob, 2) # [b s 1]

        # TODO (rprenger) TODO this could be made easier to read
        # Converting [b, s, h] to [b*s, h].
        # Each vector could be routed differently
        hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [b*s h]
        max_prob = max_prob.view(-1, max_prob.size(2)) # [b*s 1]
        max_ind = max_ind.view(-1) # [b*s]

        output_total = torch.empty_like(hidden_states)
        output_bias_total = torch.empty_like(hidden_states)
        #TODO (rprenger) This does each expert in serial, but it could be parallelized
        
        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices,:]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices,:] = output
            output_bias_total[local_indices,:] = output_bias

        output_total = output_total*max_prob
        output_bias_total = output_bias_total*max_prob
        output_total = output_total.view(b, s, h)
        output_bias_total = output_bias_total.view(b, s, h)

        return output_total, output_bias_total

class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(projection_size,
                                                    world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method)
        else:
            assert attention_type == AttnType.cross_attn
            self.query = mpu.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method)

            self.key_value = mpu.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
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

        # Output.
        self.dense = mpu.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True)


    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())
        

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None):
        # hidden_states: [sq, b, h]


        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]


        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = mpu.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)


        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
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


        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)


        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

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

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: torch.Tensor,
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)

'''
0x01 并行Transformer层
在论文篇之中，我们了解到，因为模型越来越大，其尺寸远远超过了处理器的内存限制，
因此产生了诸如激活检查点（activation checkpointing）这样的内存管理技术。
而模型并行则通过对模型进行各种分片来克服单个处理器内存限制，这样模型权重和其关联的优化器状态就可以分散到多个设备之上。

ParallelTransformerLayer 就是对 Transformer 层的并行实现，所以我们接着分析。

1.1 初始化
ParallelTransformerLayer 初始化方法之中，建立了如下：

    生成一个LayerNorm处理输入数据。
    生成并行Attention。
    生成处理attention输出的LayerNorm。
    如果是decoder，则生成一个ParallelAttention。
    生成一个并行MLP。
对应就是：    
31.jpg    
'''
class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(  # 生成一个LayerNorm处理输入数据
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm)

        # Self attention.
        self.self_attention = ParallelAttention( # 生成并行Attention
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(  # 生成处理attention输出的LayerNorm
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm)

        if self.layer_type == LayerType.decoder: # 如果本层是decoder
            self.inter_attention = ParallelAttention( # 则生成一个ParallelAttention
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm)

        # MLP
        if args.num_experts is not None:
            self.mlp = SwitchMLP(init_method, output_layer_init_method)
        else:
            self.mlp = ParallelMLP(init_method, output_layer_init_method)  # 生成一个并行MLP

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

    '''
    1.2 前向传播
    其前向传播方法如下，就是调用各种成员函数进行前向操作。
    '''
    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [b, s, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states) # 对输入进行处理
        # Self attention. # attention操作
        attention_output, attention_bias = \
            self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params)

        # Residual connection. 残差连接
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output   #norm之后结果作为X
        else:
            residual = hidden_states  # 原始输入X

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:  # dropout操作
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            #yknote代码有不同
            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func( # dropout操作
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)  # 处理attention输出

        if self.layer_type == LayerType.decoder:
            attention_output, attention_bias = \
                self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
            # residual connection
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = layernorm_input

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
            layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output) # MLP操作

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm: # 残差操作
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            # yknote代码有不同
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func( # dropout操作
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        return output


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
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()

'''
4.2.3 ParallelTransformer
这里会调用 ParallelTransformerLayer 生成具体的 Transformer层，我们会在后文中进行分析。

即，ParallelTransformer 包括多个 Transformer，其中每层 Transformer 是一个 ParallelTransformerLayer。
'''
'''
0x09 如何把模型分到GPU
我们最后还有一个问题没有涉及，就是如何把模型分块放到对应的GPU之上。
就是如何与最初分成A，B，..., H 的那个图对应起来。
其实，不是根据模型来把模型部分拷贝到对应的rank或者GPU，而是rank或者GPU主动过来拷贝自己对应的层。

    因为调用了 mpu.initialize_model_parallel 来设置模型并行，数据并行等各种进程组，
    所以每个 rank 对应的进程都有自己的全局变量，具体其实就是进程自动就被映射到GPU上了。
    比如 rank 2 对应的进程在启动之后才知道自己是 rank 2，
    然后从初始化的全局变量之中知道自己的 data_parallel group 是 [g0, g2]，
    tensor model-parallel group 是[g2, g3]，pipeline model-parallel group 是 [g2, g6, g10, g14]。
    
    ParallelTransformer 的初始化之中，offset 就是根据 rank 知道自己应该生成模型的那些层，
    然后通过 self.layers = torch.nn.ModuleList([build_layer(i + 1 + offset) for i in range(self.num_layers)]) 来生成对应的层。
    
    get_model 方法也会根据自己的 pipeline rank 和 is_pipeline_first_stage 来知道是不是第一层或者最后一层，
    然后做相应处理。
    
    最后把模型参数拷贝到了自己对应的 GPU 之上。

具体 ParallelTransformer 初始化代码如下：

所以，最终效果如下，其中同名子模块具有同样的参数，可以数据并行，即两个A可以数据并行。
一列上的层之间可以流水线串行，比如 A--> C --> E --> G 就是串行，而一个横行4个是流水线的一个stage，
其中从0开始，横向相邻两个GPU是 tensor model 并行。
57.jpg

'''
class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True, 
                 pre_process=True, post_process=True,
                 drop_path_rate=0.0):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate

        # Store activation checkpoiting flag.
        self.activations_checkpoint_method = args.activations_checkpoint_method
        self.activations_checkpoint_num_layers = args.activations_checkpoint_num_layers
        self.distribute_checkpointed_activations = args.distribute_checkpointed_activations

        # Number of layers. # 获得本Transformer的具体层数
        self.num_layers = mpu.get_num_layers(
            args, args.model_type == ModelType.encoder_and_decoder)

        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, args.num_layers)]

        # Transformer layers.
        def build_layer(layer_number):
            # 返回一层 Transformmer
            return ParallelTransformerLayer(
                init_method,
                output_layer_init_method,
                layer_number,
                layer_type=layer_type,
                self_attn_mask_type=self_attn_mask_type,
                drop_path_rate=self.drop_path_rates[layer_number - 1])

        # 下面 offset 就是根据rank知道自己应该生成模型的那些层
        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
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
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

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
        else:
            self.layers = torch.nn.ModuleList(  # ���� num_layers �� Transformer
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm)
        '''
        目前逻辑如下，我们假定有两个 transformer：26.jpg
        '''

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                attention_mask = inputs[1]
                encoder_output = inputs[2]
                enc_dec_attn_mask = inputs[3]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, attention_mask, encoder_output, enc_dec_attn_mask)
                return x_
            return custom_forward

        if self.activations_checkpoint_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                hidden_states = mpu.checkpoint(
                    custom(l, l + self.activations_checkpoint_num_layers),
                    self.distribute_checkpointed_activations,
                    hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                l += self.activations_checkpoint_num_layers
        elif self.activations_checkpoint_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.activations_checkpoint_num_layers:
                    hidden_states = mpu.checkpoint(
                        custom(l, l + 1),
                        self.distribute_checkpointed_activations,
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                else:
                    hidden_states = custom(l, l + 1)(
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
        else:
            raise ValueError("Invalid activation checkpoint method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    '''
4.2.3.2 前向传播
我们接着看看其前向传播函数，这里主要就是调用内部 ParallelTransformerLayer 的 forward 方法，
如果是第一层或者最后一层，则做特殊处理。
    '''
    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):

        # Checks.
        if inference_params:
            assert self.activations_checkpoint_method is None, \
                'inference does not work with activation checkpointing'

        if self.pre_process:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            # If the input flag for fp32 residual connection is set, convert for float.
            if self.fp32_residual_connection:
                hidden_states = hidden_states.transpose(0, 1).contiguous().float()
            # Otherwise, leave it as is.
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
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
        hidden_states = mpu.make_viewless_tensor(
            hidden_states,
            requires_grad = True,
            keep_graph = True,
        )

        # Transpose encoder output.
        if encoder_output is not None:
            encoder_output = encoder_output.transpose(0, 1).contiguous()

        # Forward pass.
        if self.activations_checkpoint_method is not None:
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask,
                                                       encoder_output,
                                                       enc_dec_attn_mask)
        else:
            for index in range(self.num_layers):
                layer = self._get_layer(index)
                hidden_states = layer(
                    hidden_states, # 调用ParallelTransformerLayer的forward函数
                    attention_mask,
                    encoder_output=encoder_output,
                    enc_dec_attn_mask=enc_dec_attn_mask,
                    inference_params=inference_params)


        # Final layer norm.
        if self.post_process:
            # Reverting data format change [s b h] --> [b s h].
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states) if self.post_layer_norm else hidden_states
        else:
            output = hidden_states

        return output
