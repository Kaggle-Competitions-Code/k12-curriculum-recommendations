# -*- encoding: utf-8 -*-
'''
@create_time: 2023/02/13 13:58:38
@author: lichunyu
'''
import os
import pathlib

import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


MODEL_PATH = "/home/search3/lichunyu/k12-curriculum-recommendations/data/input/models/stage1/bert-base-multilingual-uncased"
# MODEL_PATH = "/home/search3/lichunyu/k12-curriculum-recommendations/data/input/models/stage1/xlm-roberta-base"
OUTPUT_PATH = "/home/search3/lichunyu/k12-curriculum-recommendations/data/output"

class PlainDataset(Dataset):

    def __init__(self, df, tokenizer, label_name="") -> None:
        super().__init__()
        self.data = df[label_name].tolist()
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = self.data[index]
        inputs = self.tokenizer(
                text, 
                add_special_tokens = True,
                truncation='longest_first',
                max_length = 128,
                padding = 'max_length',
                return_attention_mask = True,
                return_tensors = 'pt',
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def __len__(self):
        return len(self.data)


class Convert2Embed(object):

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # self.model = AutoModel.from_pretrained(MODEL_PATH)
        self.model = AutoModel.from_pretrained(MODEL_PATH).cuda()

    def convert2embeddind(self, df, label_name=""):
        embed: list = []
        dataset = PlainDataset(df, tokenizer=self.tokenizer, label_name=label_name)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)
        for batch in dataloader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                embeddings = self.model(**batch, output_hidden_states=True, return_dict=True).pooler_output
                embed.append(embeddings.cpu().clone().detach().numpy())
        embed = np.concatenate(embed, axis=0)
        return embed

    def get_embed(self):
        with open(os.path.join(OUTPUT_PATH, "valid", "language.txt"), "r") as f:
            valid_language = f.read().splitlines()

        for p in tqdm(valid_language):
            for t in ["content", "topics"]:
                path = os.path.join(OUTPUT_PATH, "valid", p, f"{t}_{p}.pqt")
                df = pd.read_parquet(path)
                embed = self.convert2embeddind(df, label_name=f"{t}_text")
                np.save(path.replace(".pqt", ".npy"), embed)


def valid():
    with open(os.path.join(OUTPUT_PATH, "valid", "language.txt"), "r") as f:
        valid_language = f.read().splitlines()

    recall_amount = 0
    recall_sum = 0
    recall_count = 0
    recall_total = {}
    stage2_data = []
    for p in tqdm(valid_language):
        content_path = os.path.join(OUTPUT_PATH, "valid", p, f"content_{p}.npy")
        topics_path = os.path.join(OUTPUT_PATH, "valid", p, f"topics_{p}.npy")
        correlations_path = os.path.join(OUTPUT_PATH, "valid", p, f"correlations_{p}.pqt")
        content_array = np.load(content_path)
        topics_array = np.load(topics_path)
        model = NearestNeighbors(n_neighbors=50, metric="cosine")
        model.fit(content_array)
        d, r = model.kneighbors(topics_array)
        df_content = pd.read_parquet(content_path.replace(".npy", ".pqt"))
        df_topics = pd.read_parquet(topics_path.replace(".npy", ".pqt"))
        df_correlations = pd.read_parquet(correlations_path).rename({"topic_id": "topics_id", "content_ids": "content_id"}, axis=1)
        pred = {"topics_id": [], "content_id": []}
        for i in range(len(df_topics)):
            r_t = r[i]
            tmp = []
            for c in r_t:
                content_id = df_content.iloc[c]["id"]
                tmp.append(content_id)
            topics_id = df_topics.iloc[i]["id"]
            pred["topics_id"].append(topics_id)
            pred["content_id"].append(tmp)
        df_pred = pd.DataFrame(pred)
        df_pred = df_pred[df_pred["topics_id"].isin(df_correlations["topics_id"].unique())].explode("content_id")
        # Pred
        df_correlations_replica = df_correlations.copy(deep=True)
        df_pred_replica = df_pred.copy(deep=True)
        df_correlations_replica["label"] = 1
        df_pred_replica = df_pred_replica.merge(df_correlations_replica, how="left", on=["topics_id", "content_id"]).fillna({"label": 0})
        stage2_data.append(df_pred_replica)
        # Recall
        df_pred["label"] = 1
        df_pred = df_correlations.merge(df_pred, how="left", on=["topics_id", "content_id"]).fillna({"label": 0})
        df_pred["label"] = df_pred["label"].astype("int")
        s = df_pred["label"].sum()
        c = df_pred["label"].count()
        # recall = df_pred["label"].sum() / df_pred["label"].count()
        recall = s / c
        recall_total[p] = recall
        recall_sum += s
        recall_count += c
    recall_amount = recall_sum / recall_count
    print(f"Recall: {recall_amount}")
    print(f"----------------Details----------------")
    for k, v in recall_total.items():
        print(f"Recall for language {k}: {v}")
    df_stage2_data = pd.concat(stage2_data)[["topics_id", "content_id", "label"]].reset_index(drop=True)
    df_stage2_data["label"] = df_stage2_data["label"].astype("int")
    df_stage2_data.to_parquet("/home/search3/lichunyu/k12-curriculum-recommendations/data/output/stage2/recall.pqt")


if __name__ == "__main__":
    # P = Convert2Embed()
    # P.get_embed()
    valid()