# -*- encoding: utf-8 -*-
'''
@create_time: 2023/03/07 10:24:58
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
    DataCollatorForLanguageModeling,
    Trainer,
    HfArgumentParser,
    TrainingArguments
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    fbeta_score
)


@dataclass
class ModelArguments:

    mlm_prob: float = field(default=0.15)


class MLModel(nn.Module):

    def __init__(
        self,
        model_name_or_path
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.cls = BertOnlyMLMHead(self.config)


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def data_collator(tokenizer, mlm=True, mlm_probability=0.15, **kwds):
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
    return collator


class PlainDataset(Dataset):

    def __init__(self, df, tokenizer, data_name="", max_length=156) -> None:
        super().__init__()
        self.data = df[data_name].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        text = self.data[index]
        inputs = self.tokenizer(
                text, 
                add_special_tokens = True,
                truncation='longest_first',
                max_length = self.max_length,
                padding = 'max_length',
                return_attention_mask = True,
                return_tensors = 'pt',
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def __len__(self):
        return len(self.data)


class MLMTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[-1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]), allow_extra_keys=True)
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()
    tokenizer = AutoTokenizer.from_pretrained("/home/search3/lichunyu/pretrain_model/all-MiniLM-L6-v2")
    df_field = pd.read_parquet("/home/search3/lichunyu/k12-curriculum-recommendations/data/output/stage2/pretrain/field.pqt")
    dataset = PlainDataset(df=df_field, data_name="field", tokenizer=tokenizer)
    model = MLModel(
        model_name_or_path="/home/search3/lichunyu/pretrain_model/all-MiniLM-L6-v2"
    )
    trainer = MLMTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator(tokenizer=tokenizer),
        tokenizer=tokenizer
    )
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
    ...