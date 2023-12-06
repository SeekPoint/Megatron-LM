#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

# TODO not used
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/workspace/yk_repo/Megatron-LM/tag_23.06/fsi-en-t5-8files-bert-large-cased-vocab-bwplc-small3_text_sentence
VOCAB_FILE=/workspace/yk_repo/Megatron-LM//bert-large-cased-vocab.txt

CHECKPOINT_PATH=/share/yk_repo/Megatron-LM/tag_23.06/release_t5_base


vocabfn=/share/yk_repo/Megatron-LM/bert-large-cased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
        --nnodes $NNODES \
        --node_rank $NODE_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT"

T5_ARGS="
    --tensor-model-parallel-size 2 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-num-layers 12 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --micro-batch-size 16 \
    --global-batch-size 128 \
    --lr 0.0001 \
    --train-iters 10 \
    --lr-decay-iters 2 \
    --lr-decay-style linear \
    --min-lr 0.00001 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16  \
    --vocab-extra-ids 100
"
DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1
"


OUTPUT_ARGS="--log-interval 1 \
             --save-interval 5 \
             --eval-interval 1 \
             --eval-iters 1 "
# torchrun $DISTRIBUTED_ARGS pretrain_t5.py 无法指定正确的版本！！！
python3.8 -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_t5.py \
    $T5_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH