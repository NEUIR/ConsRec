import sys

import json
import os.path
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
        "--output_dir",
        type=str,
        default="data/pretrain",
        help="Output data path",
    )
    parser.add_argument(
        "--split_num",
        type=int,
        default=499,
        help="token num of seq text without prompt, total num equals to 512",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=100,
        help="the sample num of random negatives ",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--tokenizer", type=str, default="google-t5/t5-base")
    args = parser.parse_args()
    return args


def load_item_input_ids(filename):
    item_input_ids_dict = dict()
    with open(filename, "r", encoding="utf-8") as f:
        for example in jsonlines.Reader(f):
            id = example["id"]
            item_ids = example["item_ids"]
            item_input_ids_dict[id] = item_ids
    return item_input_ids_dict


def load_data(filename, item_desc):
    data = []
    data_ids = []
    lines = open(filename, "r").readlines()
    for line in lines[1:]:
        example = list()
        example2 = list()
        line = line.strip().split("\t")
        target = int(line[-1])
        seq_id = line[1:-1]
        text_list = []
        for id in seq_id:
            id = int(id)
            if id == 0:
                break
            text_list.append(item_desc[id])
            example2.append(id)
        text_list.reverse()
        example.append(text_list)
        example.append(target)
        example2.append(target)
        data.append(example)
        data_ids.append(example2)
    return data, data_ids


def load_random_neagtive_items(args, item_num, data_num, train_data_ids):
    np.random.seed(args.seed)
    negative_samples = {}
    for i in range(data_num):
        samples = []
        for _ in range(args.sample_num):
            item = np.random.choice(item_num) + 1
            while item in train_data_ids[i] or item in samples:
                item = np.random.choice(item_num) + 1
            samples.append(item)
        negative_samples[i] = samples
    return negative_samples


def mask_random_item(sequences, seed=2022):
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
        masked_content = f"{masked_item}"
        return [masked_sequence, masked_content]

    processed_sequences = [process_sequence(seq) for seq in sequences]
    return [seq for seq in processed_sequences if seq is not None]


def main():
    args = get_args()
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    item_input_ids_dict = load_item_input_ids(args.item_ids_file)
    item_num = len(item_input_ids_dict)
    print("item num is %d" % item_num)
    if args.data_name == "Amazon":
        item_desc = load_item_name(args.item_file)
    elif args.data_name == "yelp":
        item_desc = load_item_address(args.item_file)
    train_data, train_data_ids = load_data(args.train_file, item_desc)
    train_data_mask = mask_random_item(train_data)
    train_data_all = train_data + train_data_mask
    data_num_1 = len(train_data)
    data_num_2 = len(train_data_mask)
    print("data num is %d" % (data_num_1 + data_num_2))
    output_file = os.path.join(args.output_dir, args.output)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    template0 = "This is Amazon dataset. "
    template1 = "Here is the visit history list of user: "
    template2 = ", recommend next item. "
    template3 = "Here is the visit history list of user that have been masked: "
    template4 = ", complete the item that is masked"

    t0 = tokenizer.encode(template0, add_special_tokens=False, truncation=False)
    t1 = tokenizer.encode(template1, add_special_tokens=False, truncation=False)
    t2 = tokenizer.encode(template2, add_special_tokens=False, truncation=False)
    t3 = tokenizer.encode(template3, add_special_tokens=False, truncation=False)
    t4 = tokenizer.encode(template4, add_special_tokens=False, truncation=False)
    with open(output_file, "w") as f:
        for idx, data in enumerate(tqdm(train_data_all)):
            pos_list = []
            query = ",".join(data[0])
            group = {}
            if idx < data_num_1:
                query = tokenizer.encode(
                    query, add_special_tokens=False, padding=False, truncation=False
                )
                query_list = list_split(query, args.split_num)
                query_list[0] = t0 + t1 + query_list[0] + t2
                pos = data[1]
                pos_list.append(item_input_ids_dict[pos])
            else:
                query = tokenizer.encode(
                    query, add_special_tokens=False, padding=False, truncation=False
                )
                query_list = list_split(query, args.split_num)
                query_list[0] = t0 + t3 + query_list[0] + t4
                labels = data[1]
                labels = tokenizer.encode(
                    labels, add_special_tokens=False, padding=False, truncation=False
                )
                pos_list.append(labels)

            group["query"] = query_list
            group["positives"] = pos_list
            group["negatives"] = []
            f.write(json.dumps(group) + "\n")

    print("-----finish------")


if __name__ == "__main__":
    main()
