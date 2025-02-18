python src/build_items.py \
    --data_name Amazon \
    --item_file data/beauty_filtered/item.txt \
    --output item.jsonl \
    --output_dir data/beauty_filtered \
    --tokenizer google-t5/t5-base \
    --item_size 32
