# -*- encoding: utf-8 -*-
'''
@create_time: 2023/03/03 10:34:17
@author: lichunyu
'''
import os
import sys
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    HfArgumentParser,
    TrainingArguments
)
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    fbeta_score
)


_default_tokenizer_param = {
    "add_special_tokens": True,
    "truncation": "longest_first",
    "max_length": 256,
    "padding": "max_length",
    "return_attention_mask": True,
    "return_tensors": "pt"
}


@dataclass
class ModelArguments:

    pooling_strategy: str = field(default="mean")


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

    def __init__(self, df, tokenizer, data_name: list=None, label_name: str="label", max_length=312) -> None:
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


class RerankTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def compute_metrics(predictions):
            all_preds = predictions[0].argmax(axis=-1)
            all_labels = predictions[1]
            f1 = f1_score(all_labels, all_preds, average="micro")
            kappa = cohen_kappa_score(all_labels, all_preds)
            f2 = fbeta_score(all_labels, all_preds, average='macro', beta=2)
            return {"f1": f1, "kappa": kappa, "f2": f2}

        self.compute_metrics = compute_metrics


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[-1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]), allow_extra_keys=True)
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained("/home/search3/lichunyu/pretrain_model/k12-all-MiniLM-L6-v2")
    df_train = pd.read_parquet("/home/search3/lichunyu/k12-curriculum-recommendations/data/output/stage2/train/train.pqt")
    df_valid = pd.read_parquet("/home/search3/lichunyu/k12-curriculum-recommendations/data/output/stage2/valid/valid.pqt")
    train_dataset = PairDataset(df_train, tokenizer, data_name=["topic_field", "content_field"])
    valid_dataset = PairDataset(df_valid, tokenizer, data_name=["topic_field", "content_field"])
    model = RerankModel(
        model_name_or_path="/home/search3/lichunyu/pretrain_model/k12-all-MiniLM-L6-v2",
        num_labels=2
    )
    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    if training_args.do_eval:
        trainer.evaluate()
    ...