#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/8/2 20:15
# @Author : Gangao Liu
import torch
import torch.nn as nn
from torch.optim import Adam
import tqdm
from ContinueTrainingBERT.utils.optim_schedule import ScheduledOptim
from ContinueTrainingBERT.utils.config import BERTConfig
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM


class BERTTrainer():
    """
    BERTTrainer make the pretrained BERT model with two LM training method.
        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction
    please check the details on README.md with simple example.
    """

    def __init__(self, config: BERTConfig, train_dataloader: DataLoader):
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and config.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Initialize the BERT Language Model, with BERT model
        self.model = BertForMaskedLM.from_pretrained(config.initial_pretrain_model).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if config.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=config.cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = None

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, 768, n_warmup_steps=config.warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = config.log_freq
        self.path_model_save = config.path_model_save
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        print(">>>>>>>> Model Structure >>>>>>>>")
        for name, parameters in self.model.named_parameters():
            print(name, ':', parameters.size())
        print(">>>>>>>> Model Structure >>>>>>>>\n")

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        self.model.train()
        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            output = self.model(**data)
            # def forward(
            #         self,
            #         input_ids: Optional[torch.Tensor] = None,
            #         attention_mask: Optional[torch.Tensor] = None,
            #         token_type_ids: Optional[torch.Tensor] = None,
            #         position_ids: Optional[torch.Tensor] = None,
            #         head_mask: Optional[torch.Tensor] = None,
            #         inputs_embeds: Optional[torch.Tensor] = None,
            #         encoder_hidden_states: Optional[torch.Tensor] = None,
            #         encoder_attention_mask: Optional[torch.Tensor] = None,
            #         labels: Optional[torch.Tensor] = None,
            #         output_attentions: Optional[bool] = None,
            #         output_hidden_states: Optional[bool] = None,
            #         return_dict: Optional[bool] = None,
            # )

            # loss = next_loss + mask_loss
            loss = output.loss.mean()

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))
        # , "total_acc=",total_correct * 100.0 / total_element)

    def save(self):
        output_path = self.path_model_save
        self.model.module.bert.cpu().save_pretrained(output_path)
        # torch.save(self.model.module.bert.cpu(), output_path)
        self.model.module.bert.to(self.device)
