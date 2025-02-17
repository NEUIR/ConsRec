from logging import getLogger
import time
import os
import argparse

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
)
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color


def save_amazon_data(dataset, train_dict, valid_dict, test_dict, save_path):
    item_feat = dataset.item_feat.numpy()
    item_id = item_feat["item_id"]
    title_id = item_feat["title"]
    category_id = item_feat["categories"]
    brand_id = item_feat["brand"]
    price = item_feat["price"]
    rank = item_feat["sales_rank"]

    title = dataset.field2id_token["title"]
    category = dataset.field2id_token["categories"]
    brand = dataset.field2id_token["brand"]

    with open(os.path.join(save_path, "item.txt"), "w", encoding="utf-8") as writer:
        writer.write(
            "%s\t%s\t%s\t%s\t%s\t%s\n"
            % ("item_id", "item_name", "categories", "brand", "price", "sales_rank")
        )
        for id, tid, cid, bid, p, r in zip(
            item_id, title_id, category_id, brand_id, price, rank
        ):
            id = int(id)
            tid = int(tid)
            cid = int(cid)
            bid = int(bid)
            p = float(p)
            p = round(p, 2)
            r = int(r)
            name = str(title[tid])
            cate = str(category[cid])
            bra = str(brand[bid])
            writer.write("%d\t%s\t%s\t%s\t%.2f\t%d\n" % (id, name, cate, bra, p, r))

    for data_dict, filename in zip(
        [train_dict, valid_dict, test_dict],
        ["train.txt", "valid.txt", "test.txt"],
    ):
        with open(os.path.join(save_path, filename), "w", encoding="utf-8") as writer:
            writer.write("%s\t%s\t%s\n" % ("user_id", "seq", "target"))
            for user_id, seq_list, target in zip(
                data_dict["user_id"], data_dict["item_id_list"], data_dict["item_id"]
            ):
                uid = int(user_id)
                writer.write("%d\t" % uid)
                for id in seq_list:
                    writer.write("%d\t" % int(id))
                tid = int(target)
                writer.write("%d\n" % tid)


def save_yelp_data(dataset, train_dict, valid_dict, test_dict, save_path):
    item_feat = dataset.item_feat.numpy()
    item_id = item_feat["business_id"]
    title_id = item_feat["item_name"]
    category_id = item_feat["categories"]
    address_id = item_feat["address"]
    city_id = item_feat["city"]
    state_id = item_feat["state"]

    title = dataset.field2id_token["item_name"]
    category = dataset.field2id_token["categories"]
    address = dataset.field2id_token["address"]
    city = dataset.field2id_token["city"]
    state = dataset.field2id_token["state"]

    with open(os.path.join(save_path, "item.txt"), "w", encoding="utf-8") as writer:
        writer.write(
            "%s\t%s\t%s\t%s\t%s\t%s\n"
            % ("item_id", "item_name", "categories", "address", "city", "state")
        )
        for id, tid, cid, cit, sid, aid in zip(
            item_id, title_id, category_id, city_id, state_id, address_id
        ):
            id = int(id)
            tid = int(tid)
            cid = int(cid)
            cit = int(cit)
            sid = int(sid)
            aid = int(aid)
            name = str(title[tid])
            cate = str(category[cid])
            ccity = str(city[cit])
            sta = str(state[sid])
            add = str(address[aid])
            writer.write("%d\t%s\t%s\t%s\t%s\t%s\n" % (id, name, cate, add, ccity, sta))

    for data_dict, filename in zip(
        [train_dict, valid_dict, test_dict],
        ["train.txt", "valid.txt", "test.txt"],
    ):
        with open(os.path.join(save_path, filename), "w", encoding="utf-8") as writer:
            writer.write("%s\t%s\t%s\n" % ("user_id", "seq", "target"))
            for user_id, seq_list, target in zip(
                data_dict["user_id"],
                data_dict["business_id_list"],
                data_dict["business_id"],
            ):
                uid = int(user_id)
                writer.write("%d\t" % uid)
                for id in seq_list:
                    writer.write("%d\t" % int(id))
                tid = int(target)
                writer.write("%d\n" % tid)


def run_recbole(
    model=None, dataset=None, config_file_list=None, config_dict=None, save_path=None
):
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    dataset = create_dataset(config)

    if config["save_dataset"]:
        dataset.save()

    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if config["save_dataloaders"]:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    train_dict = train_data.dataset.inter_feat.numpy()
    valid_dict = valid_data.dataset.inter_feat.numpy()
    test_dict = test_data.dataset.inter_feat.numpy()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if "amazon" in dataset.dataset_name.lower():
        save_amazon_data(dataset, train_dict, valid_dict, test_dict, save_path)
    elif "yelp" in dataset.dataset_name.lower():
        save_yelp_data(dataset, train_dict, valid_dict, test_dict, save_path)
    else:
        raise ValueError("Unsupported dataset type")


if __name__ == "__main__":
    begin = time.time()
    parameter_dict = {"neg_sampling": None, "train_neg_sample_args": None}
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, default="SASRec", help="name of models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="beauty", help="name of datasets"
    )
    parser.add_argument(
        "--config_files",
        type=str,
        default="configs/amazon.yaml",
        help="config files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/beauty",
        help="save path",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=parameter_dict,
        save_path=args.save_path,
    )
