import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

filenames = [
    "data/pretrain/beauty_sampled.jsonl",
    "data/pretrain/sports_sampled.jsonl",
    "data/pretrain/toys_sampled.jsonl",
    "data/pretrain/yelp_sampled.jsonl",
]

output_file = "data/pretrain/train.jsonl"


def process_file(filename):
    with open(filename, "r") as file:
        return [
            json.loads(line) for line in file if not line.strip() or json.loads(line)
        ]


def merge_data(filenames):
    with ProcessPoolExecutor() as executor:
        return [
            item for result in executor.map(process_file, filenames) for item in result
        ]


def write_to_file(data, output_file):
    with open(output_file, "w") as file:
        for item in tqdm(data, total=len(data)):
            json.dump(item, file)
            file.write("\n")


def main():
    all_data = merge_data(filenames)
    write_to_file(all_data, output_file)


if __name__ == "__main__":
    main()
