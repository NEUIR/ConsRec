import glob
import os
import shutil
import faiss
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer
from src.model import TASTEModel
from src.data_loader import (
    load_item_name,
    load_item_address,
    load_data,
    load_item_data,
    SequenceDataset,
    ItemDataset,
)
from src.option import Options
from src.metrics import get_metrics_dict
from src.utils import set_randomseed, init_logger, early_stopping


def evaluate(
    model,
    eval_seq_dataloader,
    eval_item_dataloader,
    device,
    Ks,
    logging,
    tb_logger,
    step,
):
    model.eval()
    model = model.module if hasattr(model, "module") else model
    item_emb_list, seq_emb_list, target_item_list = [], [], []

    with torch.no_grad():
        for batch in eval_item_dataloader:
            item_inputs, item_masks = batch["item_ids"].to(device), batch[
                "item_masks"
            ].to(device)
            _, item_emb = model(item_inputs, item_masks)
            item_emb_list.append(item_emb.cpu().numpy())

        item_emb_list = np.concatenate(item_emb_list, 0)

        for batch in eval_seq_dataloader:
            seq_inputs, seq_masks = batch["seq_ids"].to(device), batch["seq_masks"].to(
                device
            )
            batch_target = batch["target_list"]
            _, seq_emb = model(seq_inputs, seq_masks)
            seq_emb_list.append(seq_emb.cpu().numpy())
            target_item_list.extend(batch_target)

        seq_emb_list = np.concatenate(seq_emb_list, 0)

        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(768)
        cpu_index.add(np.array(item_emb_list, dtype=np.float32))
        D, I = cpu_index.search(np.array(seq_emb_list, dtype=np.float32), Ks)

        n_item, n_seq = item_emb_list.shape[0], seq_emb_list.shape[0]
        metrics_dict = get_metrics_dict(I, n_seq, n_item, Ks, target_item_list)

    logging.info(
        f"current:step: {step} Recall@10: {metrics_dict[10]['recall']:.4f}, Recall@20: {metrics_dict[20]['recall']:.4f}, NDCG@10: {metrics_dict[10]['ndcg']:.4f}, NDCG@20: {metrics_dict[20]['ndcg']:.4f}"
    )

    if tb_logger:
        tb_logger.add_scalars(
            "metrics",
            {
                "recall@10": metrics_dict[10]["recall"],
                "recall@20": metrics_dict[20]["recall"],
                "ndcg@10": metrics_dict[10]["ndcg"],
                "ndcg@20": metrics_dict[20]["ndcg"],
            },
            step,
        )

    return metrics_dict


def main():
    options = Options()
    opt = options.parse()
    set_randomseed(opt.seed)
    logging = init_logger(os.path.join(opt.logging_dir, "eval", "eval.log"))
    tb_logger = SummaryWriter(os.path.join(opt.logging_dir, "eval", "tensorboard"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("***** Running evaluation *****")
    best_dev_recall, stop_step, stop_flag, update_flag = 0.0, 0, False, False

    all_checkpoint_path = opt.all_models_path
    best_checkpoint = os.path.join(all_checkpoint_path, "best_dev")
    len_path = len(all_checkpoint_path) + 1
    id_file = os.path.join(all_checkpoint_path, "checkpoint-*")
    all_checkpoint = sorted(
        glob.glob(id_file), key=lambda name: int(name[len_path + 11 :])
    )
    logging.info(f"all checkpoint amounts: {len(all_checkpoint)}")

    tokenizer = T5Tokenizer.from_pretrained(all_checkpoint[0])

    eval_file = os.path.join(opt.data_dir, "valid.txt")
    item_file = os.path.join(opt.data_dir, "item.txt")

    item_desc = (
        load_item_name(item_file)
        if opt.data_name in ["beauty", "sports", "toys"]
        else load_item_address(item_file)
    )
    logging.info(f"item len: {len(item_desc)}")

    eval_data = load_data(eval_file, item_desc)
    logging.info(f"dev len: {len(eval_data)}")

    item_data = load_item_data(item_desc)

    eval_seq_dataset = SequenceDataset(eval_data, tokenizer, opt)
    eval_seq_dataloader = DataLoader(
        eval_seq_dataset,
        sampler=SequentialSampler(eval_seq_dataset),
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=eval_seq_dataset.collect_fn,
    )

    eval_item_dataset = ItemDataset(item_data, tokenizer, opt)
    eval_item_dataloader = DataLoader(
        eval_item_dataset,
        sampler=SequentialSampler(eval_item_dataset),
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=eval_item_dataset.collect_fn,
    )

    Ks = eval(opt.Ks)

    for check_path in all_checkpoint:
        step = int(check_path[len_path + 11 :])
        model = TASTEModel.from_pretrained(check_path).to(device)
        metrics_dict = evaluate(
            model,
            eval_seq_dataloader,
            eval_item_dataloader,
            device,
            Ks,
            logging,
            tb_logger,
            step,
        )

        cur_recall = metrics_dict[20]["recall"]
        best_dev_recall, stop_step, stop_flag, update_flag = early_stopping(
            cur_recall, best_dev_recall, stop_step, opt.stopping_step
        )

        if update_flag:
            shutil.copytree(check_path, best_checkpoint, dirs_exist_ok=True)
            logging.info(
                f"Saved Best:step:{step}, Recall@10:{metrics_dict[10]['recall']:.4f}, Recall@20:{metrics_dict[20]['recall']:.4f}, NDCG@10:{metrics_dict[10]['ndcg']:.4f}, NDCG@20:{metrics_dict[20]['ndcg']:.4f}"
            )

        if stop_flag:
            logging.info("Early stop! Finished!")
            break

    logging.info("***** Finish evaluation *****")


if __name__ == "__main__":
    main()
