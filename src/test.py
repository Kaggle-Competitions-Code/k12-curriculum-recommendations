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
    #     torch.load("/home/search3/lichunyu/k12-curriculum-recommendations/src/output_dir/k12_stage2/checkpoint-26984/pytorch_model.bin", map_location=torch.device("cpu"))
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
    f2_dict = {}
    for p in np.arange(0.01, 0.15, 0.01):
        df_retrival = pd.read_parquet("retrival.pqt")
        # df_retrival_top = df_retrival.sort_values(["topic_id", "prob"], ascending=[True, False]).groupby("topic_id").head(1)
        df_retrival_top = df_retrival.groupby("topic_id").head(3)
        df_retrival = df_retrival[df_retrival["prob"]>=p].reset_index(drop=True)[["topic_id", "content_ids"]]
        df_label = pd.read_parquet("/home/search3/lichunyu/k12-curriculum-recommendations/data/input/kflod_data/flod0/valid_correlations_flod0_no_source.pqt")
        df_filler = df_retrival_top[~df_retrival_top["topic_id"].isin(df_retrival["topic_id"].unique())]
        if len(df_filler)>0:
            df_retrival = pd.concat([df_retrival, df_filler])[["topic_id", "content_ids"]]
        recall = len(df_label.merge(df_retrival, on=["topic_id", "content_ids"], how="inner"))/len(df_label)
        prec = len(df_label.merge(df_retrival, on=["topic_id", "content_ids"], how="inner"))/len(df_retrival)
        f2 = 5*(recall*prec)/(4*prec+recall)
        f2_dict[str(p)]: f2
        print(f"F2@{p}: {f2}")
    with open("f2_score.json", "w") as f:
        f.write(json.dumps(f2_dict, ensure_ascii=False, indent=4))
    ...