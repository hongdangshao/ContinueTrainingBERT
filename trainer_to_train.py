#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/8/4 17:12
# @Author : Gangao Liu
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from itertools import chain
from datasets import load_dataset


class TokenizedDatasets:
    def __init__(self, train_file: str = "/data/corpus.txt", val_file: str = None,
                 max_seq_length: int = 512, tokenizer: BertTokenizer = None):
        self.data_files = {"train": train_file}
        if val_file is not None:
            self.data_files['val'] = val_file
        self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length

        self.raw_datasets = load_dataset(
            "text",
            data_files=self.data_files
        )

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], max_length=self.max_seq_length, padding="max_length", truncation=True,
                              return_tensors='pt')

    def group_texts(self, examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.max_seq_length) * self.max_seq_length
        result = {
            k: [t[i: i + self.max_seq_length] for i in range(0, total_length, self.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def tokenized_datasets(self):
        datasets = self.raw_datasets.map(
            self.tokenize_function,
            batched=True,
            remove_columns='text',
            desc="Running tokenizer on every text in dataset",
        )
        datasets = datasets.map(self.group_texts, batched=True, )
        return datasets


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

model = BertForMaskedLM.from_pretrained("bert-base-chinese")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./results",  # 保存模型和日志的目录
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=100,
)

Datasets = TokenizedDatasets(tokenizer=tokenizer)
tokenized_datasets = Datasets.tokenized_datasets()
train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['val'] if 'val' in tokenized_datasets.keys() else None

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()