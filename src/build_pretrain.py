import json
import os
from argparse import ArgumentParser
import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer
from src.data_loader import load_item_name, load_item_address, list_split
import random


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_name", type=str, default="Amazon", help="choose Amazon or yelp"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train_sampled_beauty.txt",
        help="Path of the sampled valid/train.txt file",
    )
    parser.add_argument(
        "--item_file",
        type=str,
        default="item_beauty.txt",
        help="Path of the item.txt file",
    )
    parser.add_argument(
        "--item_ids_file",
        type=str,
        default="item_beauty.jsonl",
        help="Path of the item token file",
    )
    parser.add_argument("--output", type=str, default="beauty_sampled.jsonl")
    parser.add_argument(
        "--output_dir", type=str, default="data/pretrain", help="Output data path"
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=499,
        help="token num of seq text without prompt, total num equals to 512",
    )
    parser.add_argument(
        "--sample_num", type=int, default=100, help="the sample num of random negatives"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--tokenizer", type=str, default="google-t5/t5-base")
    return parser.parse_args()


def load_item_input_ids(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return {example["id"]: example["item_ids"] for example in jsonlines.Reader(f)}


def load_data(filename, item_desc):
    data, data_ids = [], []
    with open(filename, "r") as f:
        for line in f.readlines()[1:]:
            example, example2 = [], []
            line = line.strip().split("\t")
            target, seq_id = int(line[-1]), line[1:-1]
            text_list = [item_desc[int(id)] for id in seq_id if int(id) != 0]
            text_list.reverse()
            example.append(text_list)
            example.append(target)
            example2.append(target)
            example2.extend(map(int, seq_id))
            data.append(example)
            data_ids.append(example2)
    return data, data_ids


def load_random_negative_items(args, item_num, data_num, train_data_ids):
    np.random.seed(args.seed)
    negative_samples = {}
    for i in range(data_num):
        negative_samples[i] = [
            np.random.choice(item_num) + 1
            for _ in range(args.sample_num)
            if np.random.choice(item_num) + 1 not in train_data_ids[i]
        ]
    return negative_samples


def mask_random_item(sequences, seed=42):
    random.seed(seed)

    def process_sequence(sequence):
        items = sequence[0]
        if len(items) < 3:
            return None
        mask_index = random.randint(0, len(items) - 1)
        masked_item = items[mask_index - 1]
        masked_sequence = (
            items[: mask_index - 1] + ["<extra_id_0>"] + items[mask_index:]
        )
        return [masked_sequence, str(masked_item)]

    return [seq for seq in (process_sequence(seq) for seq in sequences) if seq]


def main():
    args = get_args()
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    item_input_ids_dict = load_item_input_ids(args.item_ids_file)
    item_num = len(item_input_ids_dict)
    print(f"item num is {item_num}")

    item_desc = (
        load_item_name(args.item_file)
        if args.data_name == "Amazon"
        else load_item_address(args.item_file)
    )
    train_data, train_data_ids = load_data(args.train_file, item_desc)
    train_data_mask = mask_random_item(train_data)
    train_data_all = train_data + train_data_mask
    print(f"data num is {len(train_data_all)}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output)

    template0 = tokenizer.encode(
        "This is Amazon dataset. ", add_special_tokens=False, truncation=False
    )
    template1 = tokenizer.encode(
        "Here is the visit history list of user: ",
        add_special_tokens=False,
        truncation=False,
    )
    template2 = tokenizer.encode(
        ", recommend next item. ", add_special_tokens=False, truncation=False
    )
    template3 = tokenizer.encode(
        "Here is the visit history list of user that have been masked: ",
        add_special_tokens=False,
        truncation=False,
    )
    template4 = tokenizer.encode(
        ", complete the item that is masked", add_special_tokens=False, truncation=False
    )

    with open(output_file, "w") as f:
        for idx, data in enumerate(tqdm(train_data_all)):
            query = ",".join(data[0])
            query_encoded = tokenizer.encode(
                query, add_special_tokens=False, padding=False, truncation=False
            )
            query_list = list_split(query_encoded, args.split_num)

            if idx < len(train_data):
                query_list[0] = template0 + template1 + query_list[0] + template2
                pos = data[1]
                pos_list = [item_input_ids_dict[pos]]
            else:
                query_list[0] = template0 + template3 + query_list[0] + template4
                labels = tokenizer.encode(
                    str(data[1]),
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )
                pos_list = [labels]

            group = {"query": query_list, "positives": pos_list, "negatives": []}
            f.write(json.dumps(group) + "\n")


if __name__ == "__main__":
    main()
