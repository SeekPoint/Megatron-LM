#! /bin/bash

# Setting the environment variables
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=WARN
#  File "/usr/local/lib/python3.8/site-packages/torch/distributed/distributed_c10d.py", line 862, in _new_process_group_helper
#    pg = ProcessGroupGloo(prefix_store, group_rank, group_size, timeout=timeout)
#RuntimeError: [enforce fail at ../third_party/gloo/gloo/transport/tcp/device.cc:80] ifa != nullptr. Unable to find address for: bond4/docker0
# bond4 改名还不行，直接去掉！
#export GLOO_SOCKET_IFNAME="docker0"
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST=Ampere

# Distributed training variables
NNODES=1
GPUS_PER_NODE=2
GPU_NUM=$((${GPUS_PER_NODE}*${NNODES}))
WORLD_SIZE=$((${GPUS_PER_NODE}*${NNODES}))
NODE_RANK=0
MASTER_PORT=6000
MASTER_ADDR="localhost"

# Parallelism variables
TP=2
PP=1
DP=$((${GPU_NUM}/${TP}/${PP}))

# Network size variables
MODEL_SIZE=7

if   [[ ${MODEL_SIZE} == 7 ]];   then HIDDEN_SIZE=1024;  NUM_HEAD=16; NUM_QUERY_GROUP=16; NUM_LAYERS=16; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 13 ]];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == 70 ]];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
elif [[ ${MODEL_SIZE} == "tiny" ]]; then HIDDEN_SIZE=128;  NUM_HEAD=4; NUM_QUERY_GROUP=4; NUM_LAYERS=4; FFN_HIDDEN_SIZE=512; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

DROP_OUT=0.0
MAX_LR=3e-5
MIN_LR=3e-6
MAX_SEQ_LEN=1024
MAX_POSITION_EMBEDDINGS=1024

# Paths
BASE_PATH=/workspace/yk_repo/Megatron-LM/tag_23.06
SRC_PATH=./pretrain_llama.py

LOG_NAME=llama2-7b_pretrain_WS${WORLD_SIZE}_TP${TP}_PP${PP}
LOG_PATH=${BASE_PATH}/log/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log/${LOG_NAME}

DATA_PATH=/share/hf_model/oscar-10k-meg-llama_text_document
DATA_CACHE_PATH="./data_cache/${LOG_NAME}"
mkdir -p ${DATA_CACHE_PATH}

SAVE_PATH=${BASE_PATH}/checkpoint-llama2/${LOG_NAME}
LOG_PATH=${BASE_PATH}/log-llama2/${LOG_NAME}/node${NODE_RANK}.log
mkdir -p ${BASE_PATH}/log-llama2/${LOG_NAME}

TOKENIZER_PATH=${BASE_PATH}/Llama2Tokenizer/tokenizer.model

# Set training command
LAUNCHER=" \
       python3.8 -m torch.distributed.launch \
       --nproc_per_node ${GPUS_PER_NODE} \
       --nnodes ${NNODES} \
       --node_rank ${NODE_RANK} \
       --master_addr ${MASTER_ADDR} \
       --master_port ${MASTER_PORT} \
       "

DISTRIBUTED_ARGS=" \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --distributed-backend nccl \
       --use-distributed-optimizer \
       --sequence-parallel \
       "

NETWORK_SIZE_ARGS=" \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_HEAD} \
       --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
       --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
       --make-vocab-size-divisible-by 1 \
       --swiglu \
       --untie-embeddings-and-output-weights \
       --use-flash-attn \
       "

LOGGING_ARGS=" \
       --log-timers-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --log-memory-to-tensorboard \
       "

REGULATIZATION_ARGS=" \
       --attention-dropout ${DROP_OUT} \
       --hidden-dropout ${DROP_OUT} \
       --weight-decay 1e-1 \
       --clip-grad 1.0 \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --adam-eps 1e-8 \
       "

TRAINING_ARGS=" \
       --micro-batch-size 1 \
       --global-batch-size 4 \
       --train-iters 10 \
       --log-interval 1 \
       --disable-bias-linear \
       --no-bias-gelu-fusion \
       --optimizer adam \
       --exit-interval 2 \
       --recompute-activations \
       --recompute-granularity selective \
       "

INITIALIZATION_ARGS=" \
       --seed 1403 \
       --init-method-std 0.02 \
       "

LEARNING_RATE_ARGS=" \
       --lr ${MAX_LR} \
       --lr-decay-style cosine \
       --lr-warmup-fraction 0.1 \
       --min-lr ${MIN_LR} \
       "

CHECKPOINTING_ARGS=" \
       --finetune \
       --no-load-optim \
       --no-load-rng \
       --save ${SAVE_PATH} \
       --save-interval 10 \
       "

MIXED_PRECISION_ARGS=" \
       --no-query-key-layer-scaling \
       "

VALIDATION_ARGS=" \
       --eval-interval 2 \
       --eval-iters 2 \
       "

DATA_ARGS=" \
       --data-path ${DATA_PATH} \
       --split 949,50,1 \
       --seq-length ${MAX_SEQ_LEN} \
       --num-workers 0 \
       --tokenizer-type Llama2Tokenizer \
       --vocab-file Llama2Tokenizer/vocab.json \
       --merge-file Llama2Tokenizer/merges.txt \
       --tokenizer-model ${TOKENIZER_PATH} \
       --data-cache-path ${DATA_CACHE_PATH} \
       "

CMD="${LAUNCHER} \
       ${SRC_PATH} \
       ${DISTRIBUTED_ARGS} \
       ${NETWORK_SIZE_ARGS} \
       ${LOGGING_ARGS} \
       ${REGULATIZATION_ARGS} \
       ${TRAINING_ARGS} \
       ${INITIALIZATION_ARGS} \
       ${LEARNING_RATE_ARGS} \
       ${CHECKPOINTING_ARGS} \
       ${MIXED_PRECISION_ARGS} \
       ${VALIDATION_ARGS} \
       ${DATA_ARGS} \
       ${MOE_ARGS} \
       "
echo ${CMD}
${CMD} 2>&1 | tee ${LOG_PATH}