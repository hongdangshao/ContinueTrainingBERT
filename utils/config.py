#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/8/2 19:56
# @Author : Gangao Liu
class BERTConfig(object):

    def __init__(self):
        """
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param learning_rate: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        # 训练配置
        self.learning_rate = 1e-3  # 学习率
        self.warmup_steps = 10000  # warm up步数
        self.with_cuda = True
        self.cuda_devices = [0, 1, 2, 3]
        self.log_freq = 10
        self.betas = (0.9, 0.999)
        self.vocab_size = 1000
        self.weight_decay = 0.01
        # 模型及路径配置
        self.initial_pretrain_model = 'bert-base-chinese'
        self.initial_pretrain_tokenizer = 'bert-base-chinese'
        self.path_model_save = '/data/agl/_nlp/BERT-pytorch/output/checkpoint/'  # 模型保存路径
        self.path_datasets = "/data/agl/_nlp/BERT-pytorch/data/corpus.txt"  # 数据集
        self.path_log = 'output/logs/'