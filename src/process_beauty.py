import pandas as pd
import argparse


def preprocess_interaction(intercation_path, output_path, prefix="books"):
    ratings = pd.read_csv(
        intercation_path,
        sep=",",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    print(f"{prefix} #data points before filter: {ratings.shape[0]}")
    print(f"{prefix} #user before filter: {len(set(ratings['user_id'].values))}")
    print(f"{prefix} #item before filter: {len(set(ratings['item_id'].values))}")

    item_id_count = (
        ratings["item_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="item_count")
    )
    user_id_count = (
        ratings["user_id"]
        .value_counts()
        .rename_axis("unique_values")
        .reset_index(name="user_count")
    )
    ratings = ratings.join(item_id_count.set_index("unique_values"), on="item_id")
    ratings = ratings.join(user_id_count.set_index("unique_values"), on="user_id")
    ratings = ratings[ratings["item_count"] >= 5]
    ratings = ratings[ratings["user_count"] >= 5]
    ratings = ratings.groupby("user_id").filter(lambda x: len(x["item_id"]) >= 5)
    print(f"{prefix} #data points after filter: {ratings.shape[0]}")

    print(f"{prefix} #user after filter: {len(set(ratings['user_id'].values))}")
    print(f"{prefix} #item ater filter: {len(set(ratings['item_id'].values))}")
    ratings = ratings[["item_id", "user_id", "timestamp"]]
    ratings.to_csv(output_path, index=False, header=True)


def preprocess_item(item_path, output_path, prefix="books"):
    data = []
    for line in open(item_path):
        json_data = eval(line)
        item_id = json_data.get("asin", "")
        description = json_data.get("description", "")
        title = json_data.get("title", "")

        data.append({"item_id": item_id, "description": description, "title": title})

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inter_data", type=str)
    parser.add_argument("--item_data", type=str)
    parser.add_argument("--output_inter_data", type=str, default="inter.csv")
    parser.add_argument("--output_item_data", type=str, default="item.csv")
    args = parser.parse_args()
    preprocess_interaction(args.inter_data, args.output_inter_data)
    preprocess_item(args.item_data, args.output_item_data)
