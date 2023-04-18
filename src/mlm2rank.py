# -*- encoding: utf-8 -*-
'''
@create_time: 2023/03/08 09:12:37
@author: lichunyu
'''
from collections import OrderedDict

import torch


mlm_state_dict = torch.load("/home/search3/lichunyu/k12-curriculum-recommendations/src/output_dir/k12_stage2_pretrain/checkpoint-166716/pytorch_model.bin", map_location=torch.device("cpu"))
raw = torch.load("/home/search3/lichunyu/pretrain_model/all-MiniLM-L6-v2/pytorch_model.bin", map_location=torch.device("cpu"))
rank_state_dict = OrderedDict()
for k, v in mlm_state_dict.items():
    if k.startswith("model.") and k.replace("model.", "") in raw.keys():
        rank_state_dict[k.replace("model.", "")] = v
assert len(list(raw.keys())) == len(list(rank_state_dict.keys()))
torch.save(rank_state_dict, "pytorch_model.bin")
...