import random
from typing import Dict, List, Union, Tuple, Any
import torch
from openmatch.trainer import DRTrainer
from transformers import BatchEncoding, DataCollatorWithPadding
from openmatch.dataset.train_dataset import (
    TrainDatasetBase,
    StreamTrainDatasetMixin,
    MappingTrainDatasetMixin,
)
from dataclasses import dataclass


@dataclass
class TasteCollator(DataCollatorWithPadding):
    max_q_len: int = 32
    max_p_len: int = 128
    len_seq: int = 2

    def __call__(self, features):
        qq, dd = [f["query_"] for f in features], [f["passages"] for f in features]

        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated_list = [
            self.tokenizer.pad(
                seq,
                padding="max_length",
                max_length=self.max_q_len,
                return_tensors="pt",
            )
            for seq in qq
        ]

        seq_input_ids, seq_attention_mask = [], []
        for q_collated in q_collated_list:
            item_input_ids = q_collated.data["input_ids"]
            item_attention_mask = q_collated.data["attention_mask"]
            if item_input_ids.size(0) < self.len_seq:
                b, l = self.len_seq - item_input_ids.size(0), item_input_ids.size(1)
                pad = torch.zeros([b, l], dtype=item_input_ids.dtype)
                item_input_ids = torch.cat((item_input_ids, pad), dim=0)
                item_attention_mask = torch.cat((item_attention_mask, pad), dim=0)
            seq_input_ids.append(item_input_ids[None])
            seq_attention_mask.append(item_attention_mask[None])

        query = (torch.cat(seq_input_ids, dim=0), torch.cat(seq_attention_mask, dim=0))

        d_collated = self.tokenizer.pad(
            dd, padding="max_length", max_length=self.max_p_len, return_tensors="pt"
        )
        item_input_ids = torch.unsqueeze(d_collated.data["input_ids"], 1)
        item_attention_mask = torch.unsqueeze(d_collated.data["attention_mask"], 1)

        return query, (item_input_ids, item_attention_mask)


class TasteTrainer(DRTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prepare_inputs(
        self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        return [(x[0].to(self.args.device), x[1].to(self.args.device)) for x in inputs]


class TasteTrainDataset(TrainDatasetBase):
    def create_one_example(
        self, text_encoding: List[int], is_query=False
    ) -> BatchEncoding:
        return self.tokenizer.encode_plus(
            text_encoding,
            truncation="only_first",
            max_length=(
                self.data_args.q_max_len if is_query else self.data_args.p_max_len
            ),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

    def get_process_fn(self, epoch, hashed_seed):
        def process_fn(example):
            encoded_query = [
                self.create_one_example(item, True) for item in example["query"]
            ]
            group_positives = example["positives"]
            group_negatives = example["negatives"]

            pos_psg = (
                group_positives[0]
                if self.data_args.positive_passage_no_shuffle or hashed_seed is None
                else group_positives[(hashed_seed + epoch) % len(group_positives)]
            )
            encoded_passages = [self.create_one_example(pos_psg)]

            negative_size = self.data_args.train_n_passages - 1
            negs = self._get_negatives(
                group_negatives, negative_size, epoch, hashed_seed
            )
            encoded_passages.extend(
                self.create_one_example(neg_psg) for neg_psg in negs
            )

            assert len(encoded_passages) == self.data_args.train_n_passages

            return {"query_": encoded_query, "passages": encoded_passages}

        return process_fn

    def _get_negatives(self, group_negatives, negative_size, epoch, hashed_seed):
        if len(group_negatives) < negative_size:
            return (
                random.choices(group_negatives, k=negative_size)
                if hashed_seed
                else (group_negatives * 2)[:negative_size]
            )
        elif self.data_args.train_n_passages == 1:
            return []
        elif self.data_args.negative_passage_no_shuffle:
            return group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = group_negatives.copy()
            if hashed_seed:
                random.Random(hashed_seed).shuffle(negs)
            return negs * 2[_offset : _offset + negative_size]


class StreamDRTrainDataset(StreamTrainDatasetMixin, TasteTrainDataset):
    pass


class MappingDRTrainDataset(MappingTrainDatasetMixin, TasteTrainDataset):
    pass
