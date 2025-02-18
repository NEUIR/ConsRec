python src/build_graph.py \
    --user_file data/beauty_inter.csv \
    --item_file data/beauty_item.csv \
    --embedding_path embedding/beauty_embeddings.pkl \
    --output_user_file dataset/beauty_filtered/beauty_filtered.inter \
    --threshold 0.3
