# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")

# 2.2.1 获取模型
# model_provider 函数返回的是已经被 Megatron 改造为并行的版本。
def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    gd.debuginfo(prj="mt", info=f'building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model

# 1.2.3.1 广播数据
# forward_step 函数会调用 get_batch 获取 batch 数据，其内部会从 data_iterator 获取数据，
# 然后使用 broadcast_data 函数把输入数据从 rank_0 广播到所有 tensor-model-parallel其他 ranks 之上。
def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator) # 获取数据
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype) # 把数据广播到各个GPU

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}

# 2.2.3 forward 函数
# forward_step 函数接受一个 data_iterator 和 model，并返回一个 loss 标量，该标量带有一个字典，
# 其中key:value 是希望在训练期间监视的信息（例如， {'lm loss': averaged_loss[0]}），
# 还要求此函数将 batch generator 添加到 timers 类中。
def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

# 2.2.2 获取数据集
# train_valid_test_datasets_provider 函数接受 train/valid/test 数据集的大小，
# 并返回 train，valid，test 对应部分的数据集。
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    gd.debuginfo(prj="mt", info=f'> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    gd.debuginfo(prj="mt", info=f"> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

# 2.2 预训练入口
# /Megatron-LM/pretrain_gpt.py 会调用 pretrain 函数进行预训练。
if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
