import pandas as pd
import networkx as nx
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from collections import deque


def load_data(user_file, item_file):
    users_df = pd.read_csv(user_file)
    items_df = pd.read_csv(item_file)
    return users_df, items_df


def group_items_by_user(users_df):
    return users_df.groupby("user_id")["item_id"].apply(list).reset_index()


def load_embeddings(embedding_path):
    with open(embedding_path, "rb") as f:
        return pickle.load(f)


def compute_similarity_with_precomputed_embeddings(
    item_list, embeddings_dict, threshold
):
    valid_items = [item_id for item_id in item_list if item_id in embeddings_dict]
    valid_embeddings = [embeddings_dict[item_id] for item_id in valid_items]

    if len(valid_items) < 2:
        return []

    embedding_matrix = np.array(valid_embeddings)
    similarity_matrix = cosine_similarity(embedding_matrix)

    edges = [
        (valid_items[i], valid_items[j])
        for i in range(len(valid_items))
        for j in range(i + 1, len(valid_items))
        if similarity_matrix[i, j] > threshold
    ]
    return edges


def build_graphs(user_items, embeddings_dict, threshold):
    user_graphs = {}
    for _, row in tqdm(
        user_items.iterrows(), total=user_items.shape[0], desc="Building Graph"
    ):
        user_id = row["user_id"]
        item_list = row["item_id"]
        edges = compute_similarity_with_precomputed_embeddings(
            item_list, embeddings_dict, threshold
        )
        G = nx.Graph()
        G.add_edges_from(edges)
        user_graphs[user_id] = G
    return user_graphs


def largest_connected_component(graph):
    if graph.number_of_nodes() == 0:
        return graph

    visited = set()
    largest_cc = set()
    for node in graph.nodes():
        if node not in visited:
            current_component = set()
            queue = deque([node])
            while queue:
                current_node = queue.popleft()
                if current_node not in visited:
                    visited.add(current_node)
                    current_component.add(current_node)
                    queue.extend(graph.neighbors(current_node))
            if len(current_component) > len(largest_cc):
                largest_cc = current_component
    return graph.subgraph(largest_cc)


def save_results(user_graphs, original_users_df, output_user_file):
    user_timestamps = dict(
        zip(original_users_df["user_id"], original_users_df["timestamp"])
    )
    with open(output_user_file, "w") as file:
        file.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for user_id, graph in tqdm(user_graphs.items(), desc="Saving Graph"):
            lcc = largest_connected_component(graph)
            timestamp = user_timestamps.get(user_id)
            for node in lcc.nodes():
                file.write(f"{user_id}\t{node}\t{timestamp}\n")


def main(user_file, item_file, embedding_path, output_user_file, threshold):
    output_dir = os.path.dirname(output_user_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    users_df, _ = load_data(user_file, item_file)
    user_items = group_items_by_user(users_df)
    embeddings_dict = load_embeddings(embedding_path)
    user_graphs = build_graphs(user_items, embeddings_dict, threshold)
    save_results(user_graphs, users_df, output_user_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_file", type=str, required=True)
    parser.add_argument("--item_file", type=str, required=True)
    parser.add_argument("--embedding_path", type=str, required=True)
    parser.add_argument("--output_user_file", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()
    main(
        args.user_file,
        args.item_file,
        args.embedding_path,
        args.output_user_file,
        args.threshold,
    )
