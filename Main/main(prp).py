# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 15:18
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main(prp).py
# @Software: PyCharm
# @Note    :
import sys
import os
import os.path as osp
import glob
import warnings
from datasets import load_dataset

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Main.pargs import pargs
from Main.tokenizer import count_words, create_vocab_file, TwitterTokenizer
from Main.model import BertWithPRP
from Main.utils import write_log
from Main.collator import PRPDataCollatorForPaddingAndMasking
from Main.dataset import PRPDataset
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

if __name__ == '__main__':
    args = pargs()

    vocab_size = args.vocab_size
    special_tokens = args.special_tokens

    unsup_dataset = args.unsup_dataset
    post_num_limit = args.post_num_limit

    prp_epochs = args.prp_epochs
    prp_batch_size = args.prp_batch_size
    run_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    load_name = args.load_name

    max_position_embeddings = args.max_position_embeddings
    num_attention_heads = args.num_attention_heads
    num_hidden_layers = args.num_hidden_layers
    type_vocab_size = args.type_vocab_size

    prp_dataset_path = osp.join(dirname, '..', 'Data(PRP)', unsup_dataset, 'dataset')
    nost_vocab_file_path = osp.join(dirname, '..', 'Save', 'Twitter', 'vocab_nost.txt')
    save_path = osp.join(dirname, '..', 'Save', 'Twitter', f'[PRP]{run_name}')
    load_path = osp.join(dirname, '..', 'Save', 'Twitter', load_name)
    log_path = osp.join(dirname, '..', 'Log', f'[PRP]{run_name}.log')

    log = open(log_path, 'w')
    os.makedirs(save_path, exist_ok=True)

    tokenizer = TwitterTokenizer(nost_vocab_file_path, special_tokens)
    tokenizer.save_vocabulary(save_path)


    def add_special_tokens_and_convert_to_ids(examples):
        tokens = tokenizer.tokenize(examples["text"].strip())[:max_position_embeddings - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        return {"input_ids": input_ids}


    dataset = PRPDataset(prp_dataset_path, tokenizer, max_position_embeddings, post_num_limit)
    collator = PRPDataCollatorForPaddingAndMasking(tokenizer=tokenizer, mlm=True, mlm_probability=0.12)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

    config = BertConfig(
        vocab_size=vocab_size + len(tokenizer.all_special_ids) - 1,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
    )

    model = BertWithPRP(config)
    model.load_model(load_path)
    device = args.gpu if args.cuda else 'cpu'
    model.to(device)

    grad_accumulation_steps = prp_batch_size // 1
    total_steps = len(dataloader) * prp_epochs // prp_batch_size
    warmup_steps = int(0.1 * total_steps)
    optimizer = AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    step = 0
    model.train()
    grad_accumulation_counter = 0
    for epoch in range(prp_epochs):
        print(f"epoch {epoch + 1} start", flush=True)
        for i, claim in enumerate(dataloader):
            posts, Y_RoP, Y_BrP, Y_PaP = collator(claim[0])
            posts = {k: v.to(device) for k, v in posts.items()}
            Y_RoP = Y_RoP.to(device)
            Y_BrP = Y_BrP.to(device)
            Y_PaP = Y_PaP.to(device)

            mlm_logits = model(posts['input_ids'], posts['attention_mask'])
            mlm_loss = F.cross_entropy(mlm_logits.view(-1, config.vocab_size), posts['labels'].view(-1),
                                       ignore_index=-100)

            pooler_output1 = model.pooler(posts['input_ids'], posts['attention_mask'])
            pooler_output2 = model.pooler(posts['input_ids'], posts['attention_mask'])
            root_logits = model.root_logits(pooler_output1, pooler_output2)
            branch_logits = model.branch_logits(pooler_output1, pooler_output2)
            parent_logits = model.parent_logits(pooler_output1, pooler_output2)
            root_loss = F.binary_cross_entropy_with_logits(root_logits, Y_RoP.float())
            branch_loss = F.binary_cross_entropy_with_logits(branch_logits, Y_BrP.float())
            parent_loss = F.binary_cross_entropy_with_logits(parent_logits, Y_PaP.float())
            prp_loss = root_loss + branch_loss + parent_loss

            loss = mlm_loss + prp_loss
            loss = loss / grad_accumulation_steps  # 梯度累积
            loss.backward()

            grad_accumulation_counter += 1
            if grad_accumulation_counter == grad_accumulation_steps:
                optimizer.step()
                scheduler.step()  # 更新学习率
                optimizer.zero_grad()
                grad_accumulation_counter = 0
                step += 1

            if (i + 1) % 20000 == 0:
                print(
                    f"  epoch {epoch + 1} have trained {i + 1} samples, loss: {loss.item()}, training Step: {step}",
                    flush=True)

        model.save_model(save_path)

        write_log(log, f"Epoch: {epoch + 1}, Loss: {loss.item()}, Training Step: {step}")
    model.save_model(save_path)
