# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 11:05
# @Author  :
# @Email   :
# @File    : model.py
# @Software: PyCharm
# @Note    :
import torch
from torch import nn
from transformers import BertConfig, BertModel


class BertNoPRP(torch.nn.Module):
    def __init__(self, config: BertConfig):
        super(BertNoPRP, self).__init__()

        self.bert = BertModel(config)
        self.mlm = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        mlm_logits = self.mlm(outputs.last_hidden_state)
        return mlm_logits

    def save_model(self, save_path):
        self.bert.save_pretrained(save_path)

    def load_model(self, load_path):
        self.bert = BertModel.from_pretrained(load_path)


class BertWithPRP(torch.nn.Module):
    def __init__(self, config: BertConfig):
        super(BertWithPRP, self).__init__()

        self.bert = BertModel(config)
        self.mlm = nn.Linear(config.hidden_size, config.vocab_size)

        self.root_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.parent_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.branch_projection = nn.Linear(config.hidden_size, config.hidden_size)

    def pooler(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def root_logits(self, pooler_output1, pooler_output2):
        return torch.matmul(self.root_projection(pooler_output1), self.root_projection(pooler_output2).transpose(0, 1))

    def branch_logits(self, pooler_output1, pooler_output2):
        return torch.matmul(self.branch_projection(pooler_output1),
                            self.branch_projection(pooler_output2).transpose(0, 1))

    def parent_logits(self, pooler_output1, pooler_output2):
        return torch.matmul(self.parent_projection(pooler_output1),
                            self.parent_projection(pooler_output2).transpose(0, 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        mlm_logits = self.mlm(outputs.last_hidden_state)
        return mlm_logits

    def save_model(self, save_path):
        self.bert.save_pretrained(save_path)

    def load_model(self, load_path):
        self.bert = BertModel.from_pretrained(load_path)
