export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
CHECKPOINT_PATH=experiments_codeparrot-small/
VOCAB_FILE=/share/Megatron-LM_tag_23.06_idata/gpt2-vocab.json
MERGE_FILE=/share/Megatron-LM_tag_23.06_idata/gpt2-merges.txt
DATA_PATH=/share/Megatron-LM_tag_23.06_idata/codeparrot_content_document
GPT_ARGS="--num-layers 12
--hidden-size 768
--num-attention-heads 12
--seq-length 256
--max-position-embeddings 256
--micro-batch-size 8
--global-batch-size 32
--lr 0.0005
--train-iters 200
--lr-decay-iters 20
--lr-decay-style cosine
--lr-warmup-iters 10
--weight-decay .1
--adam-beta2 .999
--fp16
--log-interval 10
--save-interval 50
--eval-interval 40
--eval-iters 100
"
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"
python3.8 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_gpt.py \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        $GPT_ARGS \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        $TENSORBOARD_ARGS