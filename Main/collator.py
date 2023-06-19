# -*- coding: utf-8 -*-
# @Time    : 2023/3/27 10:24
# @Author  :
# @Email   :
# @File    : collator.py
# @Software: PyCharm
# @Note    :
import torch
from transformers import DataCollatorForLanguageModeling


class DataCollatorForPaddingAndMasking(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.12):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

    def __call__(self, examples):
        # Padding
        batch = self.tokenizer.pad(examples, return_tensors="pt")

        # Masking
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(batch["input_ids"])
            # batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"])

        return batch


class PRPDataCollatorForPaddingAndMasking(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=True, mlm_probability=0.12):
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)

    def __call__(self, claim):
        # Padding
        posts = self.tokenizer.pad(claim['input id list'], return_tensors="pt")

        # Masking
        if self.mlm:
            posts["input_ids"], posts["labels"] = self.torch_mask_tokens(posts["input_ids"])
            # posts["input_ids"], posts["labels"] = self.mask_tokens(posts["input_ids"])

        Y_RoP = torch.tensor(claim["yrop"])
        Y_BrP = torch.tensor(claim["ybrp"])
        Y_PaP = torch.tensor(claim["ypap"])
        return posts, Y_RoP, Y_BrP, Y_PaP
