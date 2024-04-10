# -*- coding: utf-8 -*-
# @Time    : 2023/3/22 11:43
# @Author  :
# @Email   :
# @File    : tokenizer.py
# @Software: PyCharm
# @Note    :

# import sys
# import tokenizers
#
# print("Python executable location:", sys.executable)
# print("tokenizers version:", tokenizers.__version__)

import os.path as osp
from transformers import PreTrainedTokenizer


class TwitterTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, special_tokens, **kwargs):
        super().__init__(**kwargs)
        self.special_tokens = special_tokens
        self.load_vocab(vocab_file)

        self.add_special_tokens({
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "additional_special_tokens": special_tokens
        })

    def load_vocab(self, vocab_file):
        with open(vocab_file, "r", encoding='utf-8') as f:
            self.vocab = {line.strip(): i for i, line in enumerate(f.readlines() + ["[UNK]"])}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        for token, idx in self.vocab.items():
            if idx == index:
                return token
        return self.unk_token

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        assert already_has_special_tokens and token_ids_1 is None, (
            "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
            "Please use a slow (full python) tokenizer to activate this argument."
            "Or set `return_special_tokens_mask=True` when calling the encoding method "
            "to get the special tokens mask in any tokenizer. "
        )

        all_special_ids = self.all_special_ids
        special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
        return special_tokens_mask

    def save_vocabulary(self, save_directory):
        save_path = osp.join(save_directory, 'vocab.txt')
        with open(save_path, "w", encoding='utf-8') as f:
            for token in self.vocab.keys():
                f.write(token + "\n")
            for token in self.get_added_vocab().keys():
                f.write(token + "\n")


def count_words(file_paths):
    word_counts = {}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().lower().split()
                for word in words:
                    if word in word_counts:
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
    return word_counts


def create_vocab_file(word_counts, output_path, vocab_size=52000):
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for word, count in sorted_words[:vocab_size - 1 + 2]:  # -1是给[UNK]留位置，+2是跳过<@user>和<url>两个特殊token
            if word != '<@user>' and word != '<url>':
                f.write(f"{word}\n")
