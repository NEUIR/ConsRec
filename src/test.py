import os
import faiss
import numpy as np
import torch
from torch.utils.data import SequentialSampler, DataLoader
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
from src.utils import set_randomseed, init_logger


def evaluate(model, test_seq_dataloader, test_item_dataloader, device, Ks, logging):
    logging.info("***** Running testing *****")
    model.eval()
    model = model.module if hasattr(model, "module") else model
    item_emb_list, seq_emb_list, target_item_list = [], [], []

    with torch.no_grad():
        for batch in test_item_dataloader:
            item_inputs, item_masks = batch["item_ids"].to(device), batch[
                "item_masks"
            ].to(device)
            _, item_emb = model(item_inputs, item_masks)
            item_emb_list.append(item_emb.cpu().numpy())

        item_emb_list = np.concatenate(item_emb_list, 0)

        for batch in test_seq_dataloader:
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
        query_embeds = np.array(seq_emb_list, dtype=np.float32)
        D, I = cpu_index.search(query_embeds, Ks)

        n_item, n_seq = item_emb_list.shape[0], seq_emb_list.shape[0]
        metrics_dict = get_metrics_dict(I, n_seq, n_item, Ks, target_item_list)

        logging.info(
            f"Test: Recall@10: {metrics_dict[10]['recall']:.4f}, Recall@20: {metrics_dict[20]['recall']:.4f}, NDCG@10: {metrics_dict[10]['ndcg']:.4f}, NDCG@20: {metrics_dict[20]['ndcg']:.4f}"
        )
    logging.info("***** Finish test *****")


def main():
    opt = Options().parse()
    set_randomseed(opt.seed)
    logging = init_logger(os.path.join(opt.logging_dir, "test", "test.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(opt.best_model_path)
    model = TASTEModel.from_pretrained(opt.best_model_path).to(device)

    item_desc = (
        load_item_name(os.path.join(opt.data_dir, "item.txt"))
        if opt.data_name in ["beauty", "sports", "toys"]
        else load_item_address(os.path.join(opt.data_dir, "item.txt"))
    )

    logging.info(f"item len: {len(item_desc)}")

    test_data = load_data(os.path.join(opt.data_dir, "test.txt"), item_desc)
    logging.info(f"test len: {len(test_data)}")

    item_data = load_item_data(item_desc)

    test_seq_dataloader = DataLoader(
        SequenceDataset(test_data, tokenizer, opt),
        sampler=SequentialSampler(test_data),
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=SequenceDataset(test_data, tokenizer, opt).collect_fn,
    )

    test_item_dataloader = DataLoader(
        ItemDataset(item_data, tokenizer, opt),
        sampler=SequentialSampler(item_data),
        batch_size=opt.eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=ItemDataset(item_data, tokenizer, opt).collect_fn,
    )

    Ks = eval(opt.Ks)
    evaluate(model, test_seq_dataloader, test_item_dataloader, device, Ks, logging)


if __name__ == "__main__":
    main()
