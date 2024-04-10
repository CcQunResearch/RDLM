# -*- coding: utf-8 -*-
# @Time    : 2023/4/17 19:17
# @Author  :
# @Email   :
# @File    : dataset.py
# @Software: PyCharm
# @Note    :
import os
import os.path as osp
import json
import torch
import numpy as np
from torch.utils.data import Dataset


class PEPDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_position_embeddings, post_num_limit=128, chunk_size=2048,
                 preload_num=128):
        # path
        self.data_path = data_path
        self.raw_path = osp.join(data_path, 'raw')
        self.processed_data_path = osp.join(data_path, 'processed', f'{post_num_limit}')

        self.tokenizer = tokenizer
        self.max_position_embeddings = max_position_embeddings
        self.post_num_limit = post_num_limit

        self.chunk_size = chunk_size
        self.preload_num = preload_num
        if not osp.exists(self.processed_data_path):
            self.load_data()

        self.features_files = sorted(os.listdir(self.processed_data_path))
        self.lengths = []
        self.preloaded_features = []

        for file in self.features_files[:preload_num]:
            features = torch.load(os.path.join(self.processed_data_path, file))
            self.lengths.append(len(features))
            self.preloaded_features.append(features)

        for file in self.features_files[preload_num:]:
            features = torch.load(os.path.join(self.processed_data_path, file))
            self.lengths.append(len(features))

    def load_data(self):
        print('loading data...', flush=True)
        os.makedirs(self.processed_data_path)

        self.data = []
        self.chunk_num = 0
        for i, file_name in enumerate(os.listdir(self.raw_path)):
            file_path = osp.join(self.raw_path, file_name)
            post = json.load(open(file_path, 'r', encoding='utf-8'))
            one_data = {'label': post['label']} if 'label' in post.keys() else {}

            input_id_list = []
            sentences = post['sentences']
            for sen in sentences[:self.post_num_limit]:
                input_id_list.append(self.add_special_tokens_and_convert_to_ids(sen))
            one_data['input id list'] = input_id_list

            one_data["yrop"] = np.array(post["yrop"])[:self.post_num_limit, :self.post_num_limit].tolist()
            one_data["ybrp"] = np.array(post["ybrp"])[:self.post_num_limit, :self.post_num_limit].tolist()
            one_data["ypap"] = np.array(post["ypap"])[:self.post_num_limit, :self.post_num_limit].tolist()

            self.data.append(one_data)

            if len(self.data) == self.chunk_size:
                torch.save(self.data, osp.join(self.processed_data_path, f'{self.chunk_num}.pt'))
                self.data = []
                self.chunk_num += 1

        if len(self.data) > 0:
            torch.save(self.data, osp.join(self.processed_data_path, f'{self.chunk_num}.pt'))
            self.chunk_num += 1

    def add_special_tokens_and_convert_to_ids(self, sentence):
        tokens = self.tokenizer.tokenize(sentence.strip())[:self.max_position_embeddings - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        return {"input_ids": input_ids}

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        for i, length in enumerate(self.lengths):
            if index < length:
                if i < self.preload_num:
                    features = self.preloaded_features[i]
                else:
                    features_file = os.path.join(self.processed_data_path, self.features_files[i])
                    features = torch.load(features_file)
                return features[index]
            index -= length
