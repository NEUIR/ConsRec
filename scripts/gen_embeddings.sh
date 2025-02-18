python src/embedding.py \
    --input_csv data/beauty_item.csv \
    --output_prefix embedding/beauty \
    --model_name ckpt/pretrain/mfilter \
    --batch_size 512 \
    --save_every 2048
