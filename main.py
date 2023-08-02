#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/8/2 19:57
# @Author : Gangao Liu
from ContinueTrainingBERT.utils.config import BERTConfig
from ContinueTrainingBERT.BertForPreTraining.dataset import BERTDataset
from torch.utils.data import DataLoader
from ContinueTrainingBERT.BertForPreTraining.trainer import BERTTrainer
Config = BERTConfig()
train_dataset = BERTDataset(Config.path_datasets)
train_data_loader = DataLoader(train_dataset, batch_size=64, num_workers=5)

Trainer = BERTTrainer(config=Config, train_dataloader=train_data_loader)

for epoch in range(10):
    Trainer.train(epoch)