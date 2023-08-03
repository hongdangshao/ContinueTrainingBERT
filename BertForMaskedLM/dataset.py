#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/8/2 20:15
# @Author : Gangao Liu
import torch
import random
import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, DataCollatorForLanguageModeling


class BERTDataset(Dataset):

    def __init__(self, corpus_path, vocab="bert-base-chinese", encoding="utf-8", corpus_lines=None, on_memory=True):
        self.tokenizer = BertTokenizer.from_pretrained(vocab)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000, 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1 = self.get_corpus_line(item)
        inputs = self.tokenizer(t1, max_length=512, add_special_tokens=True, padding="max_length", truncation=True, return_tensors='pt')
        input_ids0 = inputs['input_ids']
        attention_mask = inputs['attention_mask'].squeeze(0)
        token_type_ids = inputs['token_type_ids'].squeeze(0)
        labels = self.data_collator([input_ids0])['labels'].squeeze(0).squeeze(0)
        input_ids = self.data_collator([input_ids0])['input_ids'].squeeze(0).squeeze(0)
        output = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  "token_type_ids": token_type_ids,
                  "labels": labels}

        return {key: value for key, value in output.items()}

    def get_corpus_line(self, index):
        if self.on_memory:
            return self.lines[index]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()
            return line



# class BERTDataloader(Dataloader):
