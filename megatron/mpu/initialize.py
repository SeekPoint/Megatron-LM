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


"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility

'''
3.3 初始化进程组全局变量
因为调用了 mpu.initialize_model_parallel 来设置模型并行，数据并行等各种进程组，所以我们假定目前进程组都已经设置成功，
所以每个 rank 对应的进程都有自己的全局变量。
假定目前有16个GPU，属于两个node，rank 0 ～7 属于第一个节点，rank 8 ～ 15 属于第二个节点。
下面的 gi 指的是第 i 个 GPU。

    _TENSOR_MODEL_PARALLEL_GROUP ：当前 rank 所属于的Intra-layer model parallel group，就是tensor 并行进程组。
        假如每一层分为两个tensor，则 _TENSOR_MODEL_PARALLEL_GROUP 例子为：[g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]。
    _PIPELINE_MODEL_PARALLEL_GROUP ：当前 rank 所属于的Intra-layer model parallel group，就是流水线进程组。
        假如流水线深度为4，则例子为 [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]。
    _MODEL_PARALLEL_GROUP ：当前 rank 所属于的模型并行进程组，包括了以上两组。
        针对我们例子，就是完整模型被复制了两份，两份分别对应的 GPU 具体是[0, 1, 4, 5, 8, 9, 12, 13]，[2, 3, 6, 7, 10, 11, 14, 15]
    _EMBEDDING_GROUP ： 嵌入对应的进程组。
    _DATA_PARALLEL_GROUP ：当前 rank 所属于的Data parallel group。
        假如数据并行度数为2，则例子为[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]。
'''
# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None

'''
0x02 初始化
initialize_model_parallel 方法用来设置模型并行，所以我们接下来就具体分析。

2.1 全局变量
因为前文_initialize_distributed之中调用了torch.distributed.init_process_group 初始化分布式环境，所以我们知道，每个进程都有自己的 gloabl rank 和 local rank，都有自己的全局变量。

2.2 初始化代码
我们首先把 initialize_model_parallel 代码摘录出来。initialize_model_parallel 作用就是对模型进行分组，然后初始化进程组相关的各种全局变量。

'''
def initialize_model_parallel(tensor_model_parallel_size_=1,
                              pipeline_model_parallel_size_=1,
                              virtual_pipeline_model_parallel_size_=None,
                              pipeline_model_parallel_split_rank_=None):
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model parallelism.
        virtual_pipeline_model_parallel_size: number of virtual stages (interleaved
                                              pipeline).
        pipeline_model_parallel_split_rank: for models with both encoder and decoder,
                                            rank in pipeline with split point.


    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

0x03 切分样例
我们使用注释内容来进行学习如何切分模型，如何把多种并行模式组合在一起。

3.1 注释
initialize_model_parallel 的注释值得我们深入学习，具体如下：

Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
the model pipeline. The present function will
create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
and 8 data-parallel groups as:
    8 data_parallel groups:
        [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
    8 tensor model-parallel groups:
        [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
    4 pipeline model-parallel groups:
        [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
Note that for efficiency, the caller should make sure adjacent ranks
are on the same DGX box. For example if we are using 2 DGX-1 boxes
with a total of 16 GPUs, rank 0 to 7 belong to the first box and
ranks 8 to 15 belong to the second box.
从注释可以知道如下信息：

假定目前有16个GPU，属于两个node，rank 0 ～7 属于第一个节点，rank 8 ～ 15 属于第二个节点。

create 8 tensor model-parallel groups, 4 pipeline model-parallel groups，这说明将一个完整模型切分如下：

沿着行横向切了一刀：tensor_model_parallel_size = 16 / 8 = 2，就是2个 GPUs 来进行模型张量并行。
沿着列纵向切了三刀：pipeline_model_parallel_size = 16 /4 = 4，就是4个GPUs 进行流水线并行。
因此，一个模型分为8块，每一块放在一个GPU之上，就是8个GPU。而通过如下计算可以知 16 GPUs / 8 GPUs = 2 models。即，16张卡可以放置两个完整模型。
因为张量模型并行组大小是2，即16个GPU被分成8组，则这8组内容是 [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]。

因为流水线并行组大小是4，即16个GPU被分成4组，则这4组内容是[g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]。

因为数据并行组大小是2，16个GPU被分成8组，则这8组内容是[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]。

以上这些进程组都是通过 torch.distributed.new_group 来完成，这样组内进程之间就知道哪些进程是在同一个组内，是在一起训练的，也知道怎么通信。

3.2 切分情况
模型原始图如下



模型切分之后如下，一共被分成8块。其中，第一层被切分为 A，B，所以 A，B 之间就是 Tensor Model parallel。后面 C，D 之间也是 Tensor Model parallel，把两层都做了切分，依次类推。



我们的目标就是用代码来看看如何生成注释里面的各种模型组。

3.3 切分策略
我们接下来看看具体切分的策略，也就是GPU分配策略。切分需要综合考虑多种情况，首先看看模型并行的通信状况。

张量并行：通信发生在每层的前向传播和后向传播过程之中，通信类型是all-reduce，不但单次通信数据量大，并且通信频繁。
流水线并行：通信在流水线阶段相邻的切分点之上，通信类型是P2P通信，单词通信数据量较少但是比较频繁，而且因为流水线的特点，会产生GPU空闲时间，这里称为流水线气泡（Bubble）。
我们接下来看看各种并行机制的对比。

Tensor versus Pipeline Parallelism. 张量模型的并行性在节点内是最好的，因为它会减少通信量。另一方面，流水线模型并行使用更便宜的点对点通信，可以跨节点执行，而不会限制整个计算。然而，流水线并行性会在流水线气泡中花费大量时间，因此，应限制流水线级的总数，以便流水线中的microbatches数量是流水线深度的合理倍数。当张量并行大小等于单个节点中的GPU数量时会达到峰值性能。
Pipeline versus Data Parallelism. 对于每个batch size，吞吐量随着流水线并行规模的增加而降低。流水线模型并行应该主要用于支持不适合单个 worker 的大型模型训练。而数据并行应该用于扩大训练规模。
Tensor versus Data Parallelism. 接下来看看数据和张量模型的并行性对性能的影响。在较大的批处理量和微批处理量为1的情况下，数据并行通信并不频繁；张量模型并行需要对批处理中的每个微批进行all-to-all通信。这种all-to-all的通信主导了端到端的训练时间，特别是当通信需要在多GPU节点上进行时。此外，随着张量模型并行规模的增加，我们在每个GPU上执行较小的矩阵乘法（因为会把模型张量进行切分），这降低了每个GPU的利用率。
最后看看结论

Tensor模型并行被用于intra-node transformer 层，因为张量并行计算密集且是耗费大量带宽，这样会在HGX based系统上高效运行。
Pipeline 模型并行主要被用于inter-node transformer 层，因为Pipeline 并行的通信带宽占用少，其可以有效利用集群中多网卡设计。
数据并行则在前两者基础之上进行加持，使得训练可以扩展到更大规模和更快的速度。我们应该注意到，尽管数据并行可以带来高效的扩展，但我们不能单独使用数据并行来处理训练超大模型，因为 a）内存容量不足，b）数据并行的扩展限制。
    """
    if torch.distributed.get_rank() == 0:
        print('> initializing tensor model parallel with size {}'.format(
            tensor_model_parallel_size_))
        print('> initializing pipeline model parallel with size {}'.format(
            pipeline_model_parallel_size_))
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size)
    ensure_divisibility(world_size,
                        tensor_model_parallel_size * pipeline_model_parallel_size)
    data_parallel_size = world_size // (tensor_model_parallel_size *
                                        pipeline_model_parallel_size)

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    num_data_parallel_groups = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size_ is not None:
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size_

    if pipeline_model_parallel_split_rank_ is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank_

    rank = torch.distributed.get_rank()

    '''
    0x07 Data-parallel
我们接下来看看数据并行。

7.1 分组
对于注释例子，16 / 2 = 8，分成 8 个进程组，每个组 两个 rank。这些分组分别是：[g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]，我们得到了如下信息：

依据上面分析， t * p 就是一个模型所需要的 GPU，因此，d = (总 GPU 数目 / 一个模型需要的 GPU 数目) = n / ( t * p)，就是说，目前提供的这 n 个GPU可以同时训练 d 个模型，就是可以用 d 个 mini-batches 输入到这 d 个模型一起训练，所以数据并行度为 d。
对应注释例子，就是data_parallel_size = 16 / (2 * 4) = 2。
rank 2 对应的数据并行进程组是[g0, g2]。
我们再看看用代码怎么确定有哪些group，每个group里面包含什么。

首先，流水线被分成了 p 个 stage，对于流水线每个stage，其有 n // p 个GPU，stage i 的 rank 范围是：[i * n//p, (i+1) * n//p]，即 rank 2所在的stage 的rank是 [0,1,2,3]。
其次，在每一个stage之中，ranks = range(start_rank + j, end_rank, tensor_model_parallel_size) ，意思是这stage的n//p个GPUs中，每隔 t 个取一个作为数据并行 group 之中的一份子，因此每个data-parallel group大小为 n // p // t = d。
具体代码如下
    '''
    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):   # 遍历流水线深度
        start_rank = i * num_pipeline_model_parallel_groups  # 找到每个stage的起始rank
        end_rank = (i + 1) * num_pipeline_model_parallel_groups  # 找到每个stage的终止rank
        for j in range(tensor_model_parallel_size):  # 遍历tensor model分组size
            ranks = range(start_rank + j, end_rank,  # 每隔 t 个取一个作为数据并行group中的一份子
                          tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group

    '''
    0x08 模型组
前面实验中，我们得到模型并行组如下：[0, 1, 4, 5, 8, 9, 12, 13] [2, 3, 6, 7, 10, 11, 14, 15]。生成代码如下：
    '''
    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)   # 就有生成 8 组
        if rank in ranks:
            # 如果本rank在某一list之中，即1 在 [0,1] 之中，则本 rank 就属于 new_group([0,1])
            _TENSOR_MODEL_PARALLEL_GROUP = group

    '''
    0x06 Pipe-parallel
本节我们分析的是，如何将 Node 上的 GPU 分给 pipeline model 并行组。

6.1 分组
从注释可以看到，流水线分组就是把这个16个GPU 分成 4 组，每组 4 个 GPU，得到 [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]，我们得到了如下信息：

每组的四个GPU进行模型流水线并行，所以 pipeline_model_parallel_size = 4。就是 Notation 之中的 p。其实，就是流水线深度为 4， 每组内 4 个 GPU 是串行的。即， [g0, g4, g8, g12] 这4个 GPU是串行的。

再看看流水线的每一层，含有 16 / 4 = 4 个 GPU，能看到第一层是 0 ~ 4，第二层是 5 ~ 8，......。

可以看到，流水线的 group是隔 n // p个取一个，比如[0, 4, 8, 12]。

对于流水线每个stage，则是stage i 的 rank 范围是：[(i-1) * n//p, (i) * n//p]，即 rank 2 所在的stage 的rank是 [0,1,2,3]。

_PIPELINE_MODEL_PARALLEL_GROUP 得到了本rank对应的流水线进程组。

_PIPELINE_GLOBAL_RANKS 得到了进程组的ranks。

假如本进程是 rank 2，则流水线进程组 ranks 是 [g2, g6, g10, g14]。

具体代码如下：
    '''
    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
        'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, \
        'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, \
        'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):  # 4
        ranks = range(i, world_size,   # 每隔 n // p个取一个
                      num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank_ is not None:
                if ranks[pipeline_model_parallel_split_rank_] not in embedding_ranks:
                    embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank_],
                                       ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank_] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank_]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True

'''
_MODEL_PARALLEL_GROUP 会得到本rank对应的模型组。
'''
def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP

'''
5.2 使用
我们接下来看看如何使用。

get_tensor_model_parallel_group 返回了自己 rank 对应的 tensor model group。
'''
def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP

'''
6.2 使用
接下来看看如何使用。

get_pipeline_model_parallel_group 返回了自己 rank 对应的 pipeline model group。
'''
def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP

'''
7.2 如何使用
我们接下来看看如何使用。

get_data_parallel_group 会得到本rank对应的 _DATA_PARALLEL_GROUP。
'''
def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, \
        'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

#get_pipeline_model_parallel_world_size 获取本流水线组world size数目，就是流水线深度。
#_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE 的意思是流水线深度 p，就是纵向切 p-1刀。
# 比如一共 12 层，纵向切 5 刀，则有 6 个stage，每个 stage 有 2 层。
'''
6.2.2 world size
get_pipeline_model_parallel_world_size 得到了进程组的 world size。
'''
def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank

#其中，get_tensor_model_parallel_rank 作用是获取本进程在tensor并行组的rank。
def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())

'''
4.2.3.1 获取层数
这里一个重点就是获取层数，即获取本模型在并行处理状况下，应该拥有多少层。如果模型一共64层，流水线深度为16，则并行每个阶段有4层，则本子模型拥有4层。
'''
def get_num_layers(args, is_encoder_and_decoder_model):
    """Compute the number of transformer layers resident on the current rank."""
    if get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
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
            assert args.num_layers % num_ranks_in_encoder == 0, \
                    'num_layers (%d) must be divisible by number of ranks given to encoder (%d)' % (args.num_layers, num_ranks_in_encoder)
            assert args.num_layers % num_ranks_in_decoder == 0, \
                    'num_layers (%d) must be divisible by number of ranks given to decoder (%d)' % (args.num_layers, num_ranks_in_decoder)
            if is_pipeline_stage_before_split():
                num_layers = (
                    0
                    if args.standalone_embedding_stage
                    and get_pipeline_model_parallel_rank() == 0 else
                    args.num_layers // num_ranks_in_encoder
                )
            else:
                num_layers = args.num_layers // num_ranks_in_decoder
        else:
            assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'num_layers must be divisible by transformer_pipeline_model_parallel_size'

            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if args.standalone_embedding_stage
                and get_pipeline_model_parallel_rank() == 0 else
                args.num_layers // args.transformer_pipeline_model_parallel_size
            )
    else:
        num_layers = args.num_layers
    return num_layers


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if get_virtual_pipeline_model_parallel_world_size() is not None and \
            get_virtual_pipeline_model_parallel_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = \
            get_virtual_pipeline_model_parallel_world_size()
        if virtual_pipeline_model_parallel_world_size is not None and \
            get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and \
            is_pipeline_stage_after_split(rank+1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE

#get_tensor_model_parallel_src_rank 计算与张量模型并行组中第一个local rank对应的全局rank。
#逻辑图具体如下，三个不同的函数分别为预训练提供不同的功能输入，做到了解耦。 .............有图
def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    data_parallel_size = get_data_parallel_world_size()
    num_data_parallel_groups = torch.distributed.get_world_size() // data_parallel_size
    return global_rank % num_data_parallel_groups


def get_pipeline_model_parallel_first_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]

'''
6.2.1 上下游rank
具体如何得到流水线上下游的rank？是通过 get_pipeline_model_parallel_next_rank 和 get_pipeline_model_parallel_prev_rank 来完成。
其中_PIPELINE_GLOBAL_RANKS 得到了进程组的ranks，假如本进程是 rank 2，则流水线进程组 ranks 是 [g2, g6, g10, g14]。
'''
def get_pipeline_model_parallel_next_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
