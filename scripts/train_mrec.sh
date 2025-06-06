python src/train.py \
    --output_dir ckpt/beauty_filtered \
    --model_name_or_path google-t5/t5-base \
    --do_train \
    --save_steps 5000 \
    --eval_steps 5000 \
    --train_path data/beauty_filtered/train.jsonl \
    --eval_path data/beauty_filtered/valid.jsonl \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --train_n_passages 10 \
    --num_passages 2 \
    --learning_rate 1e-4 \
    --q_max_len 256 \
    --p_max_len 32 \
    --seed 42 \
    --num_train_epochs 30 \
    --evaluation_strategy steps \
    --logging_dir logs/beauty_filtered
