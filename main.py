#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/8/2 19:57
# @Author : Gangao Liu
from ContinueTrainingBERT.utils.config import BERTConfig
from ContinueTrainingBERT.BertForPreTraining.dataset import BERTDataset as PreDataset
from ContinueTrainingBERT.BertForMaskedLM.dataset import BERTDataset as MaskedLMDataset
from torch.utils.data import DataLoader
from ContinueTrainingBERT.BertForPreTraining.trainer import BERTTrainer as PreTraining
from ContinueTrainingBERT.BertForMaskedLM.trainer import BERTTrainer as MaskedLM
Config = BERTConfig()
pre_train_dataset = PreDataset(Config.path_datasets)
lm_train_dataset = MaskedLMDataset(Config.path_datasets)
pretrain_data_loader = DataLoader(pre_train_dataset, batch_size=64, num_workers=5)
lm_train_data_loader = DataLoader(lm_train_dataset, batch_size=64, num_workers=5)
PreTrainer = PreTraining(config=Config, train_dataloader=pretrain_data_loader)
LMTrainer = MaskedLM(config=Config, train_dataloader=lm_train_data_loader)

for epoch in range(10):
    LMTrainer.train(epoch)

LMTrainer.save()
