# -*- encoding: utf-8 -*-
'''
@create_time: 2023/02/22 10:13:52
@author: lichunyu
'''
import pathlib
import os

import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


class PlainDataset(Dataset):

    def __init__(self, df, tokenizer, label_name="", max_length=128) -> None:
        super().__init__()
        self.data = df[label_name].tolist()
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


class Stage1(object):

    def __init__(
            self,
            test_topics_path,
            test_content_path,
            submission_path,
            model_path,
            output_path
    ) -> None:
        self.test_topics_path = test_topics_path
        self.test_content_path = test_content_path
        self.submission_path = submission_path
        self.model_path = model_path
        self.output_path = output_path
        self.stage1_path = os.path.join(output_path, "stage1")

    def split_by_language(self):
        print("start to split data by language")
        df_topics = pd.read_parquet(self.test_topics_path) # XXX
        df_submission = pd.read_csv(self.submission_path)
        df_submission = df_submission.merge(df_topics[["id", "language"]], left_on="topic_id", right_on="id", how="left")[["topic_id", "language"]]
        language_list = df_submission["language"].unique().tolist()
        pathlib.Path(self.stage1_path, "stage1").mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.stage1_path, "language.txt"), "w") as f:
            f.write("\n".join(language_list))
        df_content = pd.read_parquet(self.test_content_path) # XXX
        # concat content text
        df_content["title"] = df_content["title"].apply(lambda x: x if x is not None else "")
        df_content["description"] = df_content["description"].apply(lambda x: x if x is not None else "")
        df_content["content_text"] = df_content["title"]+df_content["description"]
        # concat topics text
        df_topics["title"] = df_topics["title"].apply(lambda x: x if x is not None else "")
        df_topics["description"] = df_topics["description"].apply(lambda x: x if x is not None else "")
        df_topics["topics_text"] = df_topics["title"]+df_topics["description"]
        for i in tqdm(language_list):
            p = os.path.join(self.stage1_path, i)
            pathlib.Path(p).mkdir(parents=True, exist_ok=True)
            df_topics[(df_topics["language"]==i)&(df_topics["id"].isin(df_submission[df_submission["language"]==i]["topic_id"].unique()))].to_parquet(os.path.join(p, "topics.pqt"))
            df_content[df_content["language"]==i].to_parquet(os.path.join(p, "content.pqt"))
        ...

    def recall(self):
        print("start to recall topn content")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).cuda()
        self.get_embed()
        with open(os.path.join(self.stage1_path, "language.txt"), "r") as f:
            language_list = f.read().splitlines()
        result = []
        for p in tqdm(language_list):
            content_path = os.path.join(self.stage1_path, p, f"content.npy")
            topics_path = os.path.join(self.stage1_path, p, f"topics.npy")
            content_array = np.load(content_path)
            topics_array = np.load(topics_path)
            model = NearestNeighbors(n_neighbors=50, metric="cosine")
            model.fit(content_array)
            d, r = model.kneighbors(topics_array)
            df_content = pd.read_parquet(content_path.replace(".npy", ".pqt"))
            df_topics = pd.read_parquet(topics_path.replace(".npy", ".pqt"))
            pred = {"topics_id": [], "content_ids": []}
            for i in range(len(df_topics)):
                r_t = r[i]
                tmp = []
                for c in r_t:
                    content_id = df_content.iloc[c]["id"]
                    tmp.append(content_id)
                topics_id = df_topics.iloc[i]["id"]
                pred["topics_id"].append(topics_id)
                pred["content_ids"].append(tmp)
            df_pred = pd.DataFrame(pred)
            df_pred.to_parquet(os.path.join(self.stage1_path, p, f"recall.pqt"))
            result.append(df_pred)
        df_recall = pd.concat(result, ignore_index=True)
        df_recall.to_parquet(os.path.join(self.stage1_path, f"recall.pqt"))
        return df_recall


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
        with open(os.path.join(self.stage1_path, "language.txt"), "r") as f:
            language_list = f.read().splitlines()

        for p in tqdm(language_list):
            for t in ["content", "topics"]:
                path = os.path.join(self.stage1_path, p, f"{t}.pqt")
                df = pd.read_parquet(path)
                df = df[["id", f"{t}_text", "language"]].fillna("")
                embed = self.convert2embeddind(df, label_name=f"{t}_text")
                np.save(path.replace(".pqt", ".npy"), embed)

    def inference(self):
        self.split_by_language()
        df_recall = self.recall()
        return df_recall




class PairDataset(Dataset):

    def __init__(self, df, tokenizer, label_name: list=None, max_length=256) -> None:
        super().__init__()
        self.data0 = df[label_name[0]].tolist()
        self.data1 = df[label_name[1]].tolist()
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
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

    def __len__(self):
        return len(self.data0)


class Stage2(object):

    def __init__(
            self,
            model_name_or_path,
            dp,
    ) -> None:
        self.model = BertForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.topic = dp.topic
        self.content = dp.content

    def inference(self, df_retrival):
        df_retrival = df_retrival.merge(
            self.topic[["id", "field"]], left_on="topic_id", right_on="id", how="left"
        ).merge(
            self.content[["id", "field"]], left_on="content_ids", right_on="id", how="left"
        )
        dataset = PairDataset(
            df=df_retrival,
            tokenizer=self.tokenizer,
            label_name=["field_x", "field_y"]
        )
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=32)
        result = []
        self.model.eval()
        self.model.cuda()
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.cuda() for k, v in batch.items()}
                output = self.model(**batch)
                result += torch.argmax(output.logits, -1).detach().cpu().numpy().tolist()
        df_retrival = pd.concat([df_retrival,  pd.DataFrame({"rank": result})], axis=1)
        df_retrival = df_retrival[df_retrival["rank"]==1].reset_index(drop=True)[["topic_id", "content_ids"]]
        df_retrival = df_retrival.groupby("topic_id").agg(list).reset_index()
        df_retrival["content_ids"] = df_retrival["content_ids"].apply(lambda x: " ".join(x))
        return df_retrival


def get_topic_field(d):
    title = list(filter(lambda x: pd.notna(x), d['title_level']))
    title = ' of '.join(title[-1::-1])
    title = 'No information' if title=='' else title
    title = '[TITLE] ' + title + '. '
    description = d['description'] if pd.notna(d['description']) else 'No information'
    description = '[DESCRIPTION]' + description + '. '
    field = title + description
    return field

def get_content_field(d):
    title = d['title']
    title = 'No information' if pd.isna(title) else title
    title = '[TITLE] ' + title + '. '
    description = d['description'] if pd.notna(d['description']) else 'No information'
    description = '[DESCRIPTION]' + description + '. '
    kind = '[' + d['kind'] + '] '
    field = kind + title + description
    return field


class DataPreparation:
    
    def __init__(self, topic_path, content_path, submission_path):
        self.topic = pd.read_csv(topic_path)
        self.content = pd.read_csv(content_path)
        self.corr = pd.read_csv(submission_path)
        # self.topic = self.topic[self.topic['id'].isin(self.corr['topic_id'].to_list())]
        self.match_dict = None
    
    def prepare_topic(self):
        df_level = self._get_level_features(self.topic)
        self.topic = self.topic.merge(df_level, on='id', how='inner')
        self.topic['field'] = self.topic.apply(lambda x: get_topic_field(x), axis=1)
        return self.topic
    
    def prepare_content(self):
        self.content['field'] = self.content.apply(lambda x: get_content_field(x), axis=1)
        return self.content
    
    def prepare_language_match(self):
        topic = self.topic[['id', 'language']].merge(self.corr, left_on='id', right_on='topic_id', how='right')[['id', 'language']]
        match_dict = {}
        for language in topic['language'].unique():
            match_dict[language] = (topic.query('language==@language')[['id']], self.content.query('language==@language')[['id']])
        self.match_dict = match_dict
        return match_dict
    
    
    def _get_level_features(self, df_topic, level_cols=['title']):
        cols = list(set(level_cols + ['id', 'parent', 'level', 'has_content']))
        df_hier = df_topic[cols]
        
        highest_level = df_hier['level'].max()
        print(f'Highest Level: {highest_level}')

        df_level = df_hier.query('level == 0').copy(deep=True)
        level_list = list()
        for col in level_cols:
            df_level[f'{col}_level'] = df_level[f'{col}'].apply(lambda x: [x])

        for i in tqdm(range(highest_level + 1)):
            level_list.append(df_level[df_level['has_content']])
            df_level_high = df_hier.query('level == @i+1')
            df_level = df_level_high.merge(df_level, left_on='parent', right_on='id', suffixes=['', '_parent'], how='inner')
            for col in level_cols:
                df_level[f'{col}_level'] = df_level[f'{col}_level'] + df_level[f'{col}'].apply(lambda x: [x])
            for col in df_level.columns:
                if col.endswith('_parent'):
                    df_level.drop(columns=col, inplace=True)
        df = pd.concat(level_list).reset_index(drop=True)
        return df[set(['id'] + [f'{col}_level' for col in level_cols])]
    
    def prepare(self):
        self.prepare_topic()
        self.prepare_content()
        self.prepare_language_match()





if __name__ == "__main__":
    # stage1 = Stage1(
    #     test_topics_path="/home/search3/lichunyu/k12-curriculum-recommendations/data/input/kflod_data/flod1/valid_topics_flod1.pqt",
    #     test_content_path="/home/search3/lichunyu/k12-curriculum-recommendations/data/input/kflod_data/flod1/valid_content_flod1.pqt",
    #     submission_path="/home/search3/lichunyu/k12-curriculum-recommendations/data/input/kflod_data/flod1/sample_submission.csv",
    #     model_path="/home/search3/lichunyu/k12-curriculum-recommendations/data/input/models/stage1/bert-base-multilingual-uncased",
    #     output_path="/home/search3/lichunyu/k12-curriculum-recommendations/tmp/output"
    # )
    # stage2 = Stage2()
    # df_recall = stage1.inference()
    # df_submission = stage2.inference(df_recall)
    # df_submission["content_ids"] = df_submission["content_ids"].apply(lambda x: " ".join(x))
    # df_submission.to_csv("submission.csv", index=False)

    # ----------------
    INPUT_DIR = '/home/search3/lichunyu/k12-curriculum-recommendations/data/input/raw'
    OUTPUT_PATH = '/home/search3/lichunyu/k12-curriculum-recommendations/data/output'

    TOPIC_DIR = os.path.join(INPUT_DIR, 'topics.csv')
    CONTENT_DIR = os.path.join(INPUT_DIR, 'content.csv')
    CORR_DIR = os.path.join(INPUT_DIR, 'sample_submission.csv')

    dp = DataPreparation(TOPIC_DIR, CONTENT_DIR, CORR_DIR)
    dp.prepare()

    df_submission = pd.read_csv("submission.csv")[["topics_id", "content_id"]].rename({"topics_id": "topic_id", "content_id": "content_ids"}, axis=1)
    df_submission["content_ids"] = df_submission["content_ids"].apply(lambda x: " ".join(eval(x)))
    df_retrival = df_submission
    df_retrival["content_ids"] = df_retrival["content_ids"].apply(lambda x: x.split(" "))
    df_retrival = df_retrival.explode("content_ids").reset_index(drop=True)

    stage2 = Stage2(
        model_name_or_path="/home/search3/lichunyu/k12-curriculum-recommendations/data/input/models/stage2/bert-base-multilingual-uncased",
        dp=dp
    )
    df_sub = stage2.inference(df_retrival)


    ...