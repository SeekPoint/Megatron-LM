# 3.3 实验
# 下面引用其他网友的一段实验代码，来模拟模型切分的结果，如下所示：

import torch

#总共有 16 个GPU
world_size = 16

# 对于模型的张量并行，我们使用了 2 个 GPU
tensor_model_parallel_size = 2 # 2 GPUs to parallelize the model tensor

#在模型的流水线并行中，我们使用了 4 个 GPU
pipeline_model_parallel_size = 4 # 4 GPUs to parallelize the model pipeline

'''
d = n / (t * p) = 2。这表示数据并行度为2，其中 n 代表总GPU数，t 代表模型并行组的数量，p 代表流水线并行组的数量。
因为 t * p 就是一个模型所需的GPU数，d 则是总 GPU 数除以一个模型所需的 GPU 数。
这意味着这些 GPU 可以同时训练 d个模型，也就是可以使用 d 个小批次（mini-batches）进行这 d 个模型的联合训练。
'''
data_parallel_size = world_size // (tensor_model_parallel_size *
                                    pipeline_model_parallel_size) # 2

# 表示从张量模型并行角度来看，我们需要分成8个进程组
num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size # 8

# 表示从模型并行角度来看，我们需要分成4个进程组
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size # 4

# 表示从数据并行角度来看，我们需要分成8个进程组。
# 这意味着会有8个数据并行分布式数据并行（DDP）组，每个DDP组包含2个进程
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
# 输出如下：
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
# 代码打印结果可以和上面的代码注释对应上。