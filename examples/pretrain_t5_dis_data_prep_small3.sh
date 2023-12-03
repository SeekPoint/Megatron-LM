jsonfile="/workspace/yk_repo/Megatron-LM/eight.files3.json"
vocabfile="/workspace/yk_repo/Megatron-LM/bert-large-cased-vocab.txt"
prefix="fsi-en-t5-8files-bert-large-cased-vocab-bwplc-small3"

#python -m ipdb tools/preprocess_data.py \
python tools/preprocess_data.py \
               --input $jsonfile \
               --output-prefix $prefix \
               --vocab $vocabfile \
               --dataset-impl mmap \
               --tokenizer-type BertWordPieceCase \
               --split-sentences