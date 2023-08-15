# 3.4 实验
# 我们接下来做一个实验看看。

import torch

world_size = 16
tensor_model_parallel_size = 2 # 2 GPUs to parallelize the model tensor
pipeline_model_parallel_size = 4 # 4 GPUs to parallelize the model pipeline
data_parallel_size = world_size // (tensor_model_parallel_size *
                                    pipeline_model_parallel_size) # 2
num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size # 8
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size # 4
num_data_parallel_groups = world_size // data_parallel_size # 8

# Build the data-parallel groups.
print("------ Build the data-parallel groups -----")
all_data_parallel_group_ranks = []
for i in range(pipeline_model_parallel_size):
    start_rank = i * num_pipeline_model_parallel_groups
    end_rank = (i + 1) * num_pipeline_model_parallel_groups
    for j in range(tensor_model_parallel_size):
        ranks = range(start_rank + j, end_rank,
                      tensor_model_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))
print(all_data_parallel_group_ranks)

# Build the model-parallel groups.
print("------ Build the model-parallel groups -----")
for i in range(data_parallel_size):
    ranks = [data_parallel_group_ranks[i]
             for data_parallel_group_ranks in all_data_parallel_group_ranks]
    print(list(ranks))

# Build the tensor model-parallel groups.
print("------ Build the tensor model-parallel groups -----")
for i in range(num_tensor_model_parallel_groups):
    ranks = range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size)
    print(list(ranks))

# Build the pipeline model-parallel groups and embedding groups
# (first and last rank in each pipeline model-parallel group).
print("------ Build the pipeline model-parallel groups -----")
for i in range(num_pipeline_model_parallel_groups):
    ranks = range(i, world_size,
                  num_pipeline_model_parallel_groups)
    print(list(ranks))
# 输出如下。需要注意，这里都是 GPU 的序列号，[0,2] 就是 [g0, g2]：
#
# ------ Build the data-parallel groups -----
# [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]
# ------ Build the model-parallel groups -----
# [0, 1, 4, 5, 8, 9, 12, 13]
# [2, 3, 6, 7, 10, 11, 14, 15]
# ------ Build the tensor model-parallel groups -----
# [0, 1]
# [2, 3]
# [4, 5]
# [6, 7]
# [8, 9]
# [10, 11]
# [12, 13]
# [14, 15]
# ------ Build the pipeline model-parallel groups -----
# [0, 4, 8, 12]
# [1, 5, 9, 13]
# [2, 6, 10, 14]
# [3, 7, 11, 15]
#
# 我们对比一下注释，发现代码打印结果可以和注释对应上：
#     Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
#     use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
#     the model pipeline. The present function will
#     create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
#     and 8 data-parallel groups as:
#         8 data_parallel groups:
#             [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
#         8 tensor model-parallel groups:
#             [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
#         4 pipeline model-parallel groups:
#             [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
# 我们接下来会进行具体分析。


'''
0x04 起始状态
4.1 GPU 状况
从注释中可以看到：

Note that for efficiency, the caller should make sure adjacent ranks are on the same DGX box. For example if we are using 2 DGX-1 boxes with a total of 16 GPUs, rank 0 to 7 belong to the first box and ranks 8 to 15 belong to the second box.
意思就是：调用者需要确保相邻的rank在同一个节点上，我们例子有两个Node，其中第一个Node拥有 GPU 0 ～ 7，就是 rank 0 ～ 7，第二个Node是 GPU 8～15，就是 rank 8 ～ 15。

具体如下，这里每行4个GPU，是因为 4 GPUs to parallelize the model pipeline，所以流水线每个stage是4个GPU。



4.2 符号说明
下面是论文之中提到的一些符号，这里有必要再取出来温习一下：

(𝑝, 𝑡, 𝑑): Parallelization dimensions.

𝑝 for the pipeline-modelparallel size,

𝑡 for the tensor-model-parallel size, and 𝑑 for the data-parallel size.

𝑛: Number of GPUs. We require 𝑝 · 𝑡 · 𝑑 = 𝑛.

4.3 初始分组
依据注释，我们得出目前分组情况和一些全局信息。

一共16个GPU，所以 world_size 为 16。就是 Notation 之中的 n。
使用两个GPU进行 model tensor 并行，所以 tensor_model_parallel_size = 2。就是 Notation 之中的 t。
使用四个GPU进行模型流水线并行，所以 pipeline_model_parallel_size = 4。就是 Notation 之中的 p。其实，就是流水线深度为 4，即，4 个 GPU 是串行的。
依据上面定义，d = n / ( t * p) = 2，就是 data_parallel_size = 2。因为 t * p 就是一个模型所需要的 GPU，d = (总 GPU / 一个模型需要的 GPU)，结果是这些GPU可以训练 d 个模型，就是可以用 d 个 mini-batches 进行这个 d个模型一起训练，所以数据并行度为 d。
接下来结合代码看看需要分成多少个process groups，他们在代码之中的变量是什么。

num_tensor_model_parallel_groups 就是从 tensor model 并行角度看，分成8 个进程roup。
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size 就是从 model 并行角度看，分成 4 个 进程group。
num_data_parallel_groups = world_size // data_parallel_size 就是从data 并行角度看，分成8 个 进程group。就是会有 8 个 DDP，每个 DDP 包括 2 个 rank。
还有一个 _MODEL_PARALLEL_GROUP，
具体如下：

world_size = 16
tensor_model_parallel_size = 2 # 2 GPUs to parallelize the model tensor
pipeline_model_parallel_size = 4 # 4 GPUs to parallelize the model pipeline
data_parallel_size = world_size // (tensor_model_parallel_size *
                                    pipeline_model_parallel_size) # 2
num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size # 8
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size # 4
num_data_parallel_groups = world_size // data_parallel_size # 8
0x05 Tensor model-parallel
本节我们分析的是，如何将 Node 上的 GPU 分给 tensor model 并行组。

5.1 分组
对于注释例子，16 / 2 = 8，分成 8 个进程组，每个组 两个 rank。这些分组分别是：[g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]，我们得到了如下信息：

[g0, g1] 就是某一层分切为2半，分别被 g0, g1 来执行，[g2, g3] 表示另一层被分为两层，分别被 g2，g3 来执行。

我们可以看到，每一个 tensor-model-parallel group的 rank一定是相邻的，比如 [g0, g1], [g2, g3]。

注意，0 ~ 7 不代表是同一个模型。0 ~ 7 是同一个 Node 上的 GPU，这点容易被混淆。


'''