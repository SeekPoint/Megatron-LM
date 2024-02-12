# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
)
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")

_MAX_DATA_DIM = 5


def _check_data_types(keys, data, target_dtype):
    gd.debuginfo(prj='mt')
    """Check that all the keys have the same target data type."""
    for key in keys:
        assert data[key].dtype == target_dtype, '{} has data type {} which '\
            'is different than {}'.format(key, data[key].dtype, target_dtype)


def _build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]

    gd.debuginfo(prj='mt', info=f'max_dim={max_dim}, sizes={sizes}')

    # Pack the sizes on rank zero.
    if get_tensor_model_parallel_rank() == 0:
        gd.debuginfo(prj='mt')
        offset = 0
        for key in keys:
            assert data[key].dim() < max_dim, 'you should increase MAX_DATA_DIM'
            size = data[key].size()
            for i, s in enumerate(size):
                sizes[i + offset] = s
            offset += max_dim

    gd.debuginfo(prj='mt', info=f'sizes={sizes}')

    # Move to GPU and broadcast.
    sizes_cuda = torch.cuda.LongTensor(sizes)
    gd.debuginfo(prj='mt', info=f'sizes_cuda={sizes_cuda}')

    torch.distributed.broadcast(sizes_cuda,
                                get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    gd.debuginfo(prj='mt', info=f'sizes_cpu={sizes_cpu}')

    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0

    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        gd.debuginfo(prj='mt', info=f'key_size[{key}]={infoTensor(size)}')
        gd.debuginfo(prj='mt', info=f'key_numel[{key}]={infoTensor(numel)}')
        total_numel += numel
        offset += max_dim

    gd.debuginfo(prj='mt', info=f'key_size={key_size}, key_numel={key_numel}, total_numel={total_numel}')

    return key_size, key_numel, total_numel

# tensor_parallel.broadcast_data 函数在每个 model parallel group 之上，把数据从 rank_0 发送到同组其他 GPU 上。
def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Arguments:
        keys: list of keys in the data disctionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)

    gd.debuginfo(prj='mt', info=f'key_size={key_size}')
    gd.debuginfo(prj='mt', info=f'key_numel={key_numel}')
    gd.debuginfo(prj='mt', info=f'total_numel={total_numel}')

    # Pack on rank zero.
    if get_tensor_model_parallel_rank() == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)

        # Flatten the data associated with the keys
        flatten_data = torch.cat([data[key].contiguous().view(-1) for key in keys], dim=0).cuda()
        gd.debuginfo(prj='mt', info=f'flatten_data={infoTensor(flatten_data)}')
    else:
        flatten_data = torch.empty(total_numel,
                                   device=torch.cuda.current_device(),
                                   dtype=datatype)
        gd.debuginfo(prj='mt', info=f'flatten_data={infoTensor(flatten_data)}')

    # Broadcast
    torch.distributed.broadcast(flatten_data,
                                get_tensor_model_parallel_src_rank(),
                                group=get_tensor_model_parallel_group())

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        tmp = flatten_data.narrow(0, offset, numel).view(size)
        output[key] = tmp
        gd.debuginfo(prj='mt', info=f'output[{key}]={infoTensor(tmp)}')
        offset += numel

    gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    return output
