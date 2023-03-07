# -*- encoding: utf-8 -*-
'''
@create_time: 2023/03/06 16:15:47
@author: lichunyu
'''
import os
import sys
from dataclasses import dataclass, field
import json

from tqdm import tqdm
import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    HfArgumentParser,
    TrainingArguments
)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class RerankModel(nn.Module):

    def __init__(
            self,
            model_name_or_path,
            num_labels
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = mean_pooling(model_output=model_output, attention_mask=attention_mask)
        logits = self.classifier(output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {
                "logits": logits,
                "loss": loss
            }
        return {"logits": logits}


class PairDataset(Dataset):

    def __init__(self, df, tokenizer, data_name: list=None, label_name: str="label", max_length=256) -> None:
        super().__init__()
        self.data0 = df[data_name[0]].tolist()
        self.data1 = df[data_name[1]].tolist()
        self.label = df[label_name].tolist()
        assert len(self.data0) == len(self.data1)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        text0 = self.data0[index]
        text1 = self.data1[index]
        inputs = self.tokenizer(
                text0,
                text1, 
                add_special_tokens = True,
                truncation='longest_first',
                max_length = self.max_length,
                padding = 'max_length',
                return_attention_mask = True,
                return_tensors = 'pt',
        )
        label = torch.tensor(self.label[index])
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = label
        return inputs

    def __len__(self):
        return len(self.data0)


if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("/home/search3/lichunyu/pretrain_model/all-MiniLM-L6-v2")
    # df_valid = pd.read_parquet("/home/search3/lichunyu/k12-curriculum-recommendations/data/output/stage2/valid/valid.pqt")
    # valid_dataset = PairDataset(df_valid, tokenizer, data_name=["topic_field", "content_field"])
    # model = RerankModel(
    #     model_name_or_path="/home/search3/lichunyu/pretrain_model/all-MiniLM-L6-v2",
    #     num_labels=2
    # )
    # model.load_state_dict(
    #     torch.load("/home/search3/lichunyu/k12-curriculum-recommendations/src/output_dir/k12_stage2/checkpoint-94444/pytorch_model.bin", map_location=torch.device("cpu"))
    # )
    # dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=64)
    # result = []
    # model.cuda()
    # model.eval()
    # with torch.no_grad():
    #     for batch in tqdm(dataloader):
    #         batch = {k: v.cuda() for k, v in batch.items()}
    #         output = model(**batch)
    #         result += torch.softmax(output["logits"], -1)[:,1].detach().clone().cpu().numpy().tolist()

    # df_retrival = pd.concat([df_valid[["topic_id", "content_ids"]],  pd.DataFrame({"prob": result})], axis=1)
    # df_retrival.to_parquet("retrival.pqt")

    best_f2_score = 0.0
    best_threshold = 0.0
    best_topn = 1
    f2_dict = {}
    for topn in range(1,10):
        for p in np.arange(0.01, 0.3, 0.01):
            df_retrival = pd.read_parquet("retrival_67460.pqt")
            # df_retrival_top = df_retrival.sort_values(["topic_id", "prob"], ascending=[True, False]).groupby("topic_id").head(25)
            df_retrival_top = df_retrival.groupby("topic_id").head(topn)
            df_retrival = df_retrival[df_retrival["prob"]>=p].reset_index(drop=True)[["topic_id", "content_ids"]]
            df_label = pd.read_parquet("/home/search3/lichunyu/k12-curriculum-recommendations/data/input/kflod_data/flod0/valid_correlations_flod0_no_source.pqt")
            df_filler = df_retrival_top[~df_retrival_top["topic_id"].isin(df_retrival["topic_id"].unique())]
            if len(df_filler)>0:
                df_retrival = pd.concat([df_retrival, df_filler])[["topic_id", "content_ids"]]

            df_retrival["recall_label"] = 1
            df_label = df_label.merge(df_retrival, on=["topic_id", "content_ids"], how="left").fillna({"recall_label": 0})
            df_label["recall_label"] = df_label["recall_label"].astype("int")
            df_recall = pl.DataFrame(df_label).groupby("topic_id").agg([(pl.col("recall_label").sum()/pl.col("recall_label").count()).alias("recall")])
            df_label["prec_label"] = 1
            df_retrival = df_retrival[["topic_id", "content_ids"]].merge(
                df_label[["topic_id", "content_ids", "prec_label"]], on=["topic_id", "content_ids"], how="left"
                ).fillna({"prec_label": 0})
            df_retrival["prec_label"] = df_retrival["prec_label"].astype("int")
            df_prec = pl.DataFrame(df_retrival).groupby("topic_id").agg([(pl.col("prec_label").sum()/pl.col("prec_label").count()).alias("prec")])
            df_f2 = df_prec.join(df_recall, on="topic_id")
            f2 = df_f2.with_columns([(5*pl.col("recall")*pl.col("prec")/(5*pl.col("prec")+pl.col("recall"))).alias("f2")]).fill_nan(pl.lit(0))["f2"].mean()

            if f2 > best_f2_score:
                best_f2_score = f2
                best_threshold = p
                best_topn = topn

            # recall = len(df_label.merge(df_retrival, on=["topic_id", "content_ids"], how="inner"))/len(df_label)
            # prec = len(df_label.merge(df_retrival, on=["topic_id", "content_ids"], how="inner"))/len(df_retrival)
            # f2 = 5*(recall*prec)/(4*prec+recall)
            f2_dict[f"F2@{p}&topn@{topn}"] = f2
            print(f"F2@{p}&topn@{topn}: {f2}")
    with open("f2_score.json", "w") as f:
        f.write(json.dumps(f2_dict, ensure_ascii=False, indent=4))
    print(f"Best F2 Score is: {best_f2_score}")
    print(f"Best Strategy is: Threshold={best_threshold}, Fill topn={best_topn}")
    ...