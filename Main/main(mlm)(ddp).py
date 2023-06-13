# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 13:14
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main(mlm)(ddp).py.py
# @Software: PyCharm
# @Note    :
import sys
import os
import os.path as osp
import warnings

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))

import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import os.path as osp
import glob
from datasets import load_dataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from Main.pargs import pargs
from Main.tokenizer import count_words, create_vocab_file, TwitterTokenizer
from Main.model import BertNoPRP
from Main.utils import write_log
from Main.collator import DataCollatorForPaddingAndMasking
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup


def main():
    args = pargs()

    # 初始化进程组
    dist.init_process_group(backend='nccl')

    # 创建本地进程ID和全局进程ID
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    global_rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    vocab_size = args.vocab_size
    special_tokens = args.special_tokens

    epochs = args.epochs
    accumulation_batch_size = args.accumulation_batch_size
    batch_size = args.batch_size
    run_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))

    max_position_embeddings = args.max_position_embeddings
    num_attention_heads = args.num_attention_heads
    num_hidden_layers = args.num_hidden_layers
    type_vocab_size = args.type_vocab_size

    nost_vocab_file_path = osp.join(dirname, '..', 'Save', 'Twitter', 'vocab_nost.txt')
    save_path = osp.join(dirname, '..', 'Save', 'Twitter', f'[MLM]{run_name}')
    twitter_nopt_text_path = osp.join(dirname, '..', 'Data', 'TwitterCorpus(EN)')
    twitter_nopt_text_cache_path = osp.join(dirname, '..', 'Data', 'TwitterCorpus(EN)cache')
    nopt_text_file_paths = glob.glob(osp.join(twitter_nopt_text_path, '*.txt'))
    log_path = osp.join(dirname, '..', 'Log', f'[MLM]{run_name}.log')

    log = open(log_path, 'w')
    os.makedirs(save_path, exist_ok=True)

    # 如果没有词汇表文件就创建一个
    if not osp.exists(nost_vocab_file_path):
        word_counts = count_words(nopt_text_file_paths)
        create_vocab_file(word_counts, nost_vocab_file_path, vocab_size)

    tokenizer = TwitterTokenizer(nost_vocab_file_path, special_tokens)
    tokenizer.save_vocabulary(save_path)

    def add_special_tokens_and_convert_to_ids(examples):
        tokens = tokenizer.tokenize(examples["text"].strip())[:max_position_embeddings - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
        return {"input_ids": input_ids}

    # 改变dataset，使其适应分布式环境
    dataset = load_dataset('text', data_files={'train': nopt_text_file_paths}, cache_dir=twitter_nopt_text_cache_path)
    tokenized_dataset = dataset.map(add_special_tokens_and_convert_to_ids, remove_columns=["text"], num_proc=16)
    collator = DataCollatorForPaddingAndMasking(tokenizer=tokenizer, mlm=True, mlm_probability=0.12)
    sampler = DistributedSampler(tokenized_dataset['train'])
    dataloader = DataLoader(tokenized_dataset['train'], batch_size=batch_size, shuffle=False, sampler=sampler,
                            collate_fn=collator)

    config = BertConfig(
        vocab_size=vocab_size + len(tokenizer.all_special_ids) - 1,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
    )

    # 更改model为DistributedDataParallel模型
    model = BertNoPRP(config)
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                    find_unused_parameters=True)

    grad_accumulation_steps = accumulation_batch_size // batch_size
    total_steps = len(dataloader) * epochs // grad_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    optimizer = AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    step = 0
    model.train()
    grad_accumulation_counter = 0
    for epoch in range(epochs):
        if global_rank == 0:
            print(f"epoch {epoch + 1} start", flush=True)
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            mlm_logits = model(batch['input_ids'], batch['attention_mask'])
            loss = F.cross_entropy(mlm_logits.view(-1, config.vocab_size), batch['labels'].view(-1), ignore_index=-100)
            loss = loss / grad_accumulation_steps  # 梯度累积
            loss.backward()

            grad_accumulation_counter += 1
            if grad_accumulation_counter == grad_accumulation_steps:
                optimizer.step()
                scheduler.step()  # 更新学习率
                optimizer.zero_grad()
                grad_accumulation_counter = 0
                step += 1

            if (i + 1) * batch_size % 8064 == 0 and global_rank == 0:
                print(
                    f"  epoch {epoch + 1} have trained {(i + 1) * batch_size} samples, loss: {loss.item()}, training Step: {step}",
                    flush=True)

        if global_rank == 0:
            model.module.save_model(save_path)

            write_log(log, f"Epoch: {epoch + 1}, Loss: {loss.item()}, Training Step: {step}")
    if global_rank == 0:
        model.module.save_model(save_path)


if __name__ == '__main__':
    # gpu_list = [int(gpu.strip()) for gpu in '0,1,2,3,4,5'.split(',')]
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    main()
