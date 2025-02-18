python src/build_pretrain.py \
    --data_name Amazon \
    --train_file data/beauty/train_sampled.txt \
    --item_file data/beauty/item.txt \
    --item_ids_file data/beauty/item.jsonl \
    --output beauty_train_sampled.jsonl \
    --output_dir data/pretrain \
    --split_num 499 \
    --sample_num 100 \
    --seed 42 \
    --tokenizer google-t5/t5-base

python src/build_pretrain.py \
    --data_name Amazon \
    --train_file data/beauty/valid_sampled.txt \
    --item_file data/beauty/item.txt \
    --item_ids_file data/beauty/item.jsonl \
    --output beauty_valid_sampled.jsonl \
    --output_dir data/pretrain \
    --split_num 499 \
    --sample_num 100 \
    --seed 42 \
    --tokenizer google-t5/t5-base
