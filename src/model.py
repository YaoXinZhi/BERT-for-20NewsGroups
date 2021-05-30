# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 14:41
@Author: XINZHI YAO
"""


import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import BertConfig
from transformers import BertPreTrainedModel

class BertClassifier(BertPreTrainedModel):

    def __init__(self, bert_config: BertConfig, bert, hidden_size, label_number, dropout):
        super(BertClassifier, self).__init__(bert_config)
        self.bert = bert
        self.fc = nn.Linear(hidden_size, label_number)
        self.dropout = nn.Dropout(dropout)

        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.Softmax = nn.Softmax(dim=1)


    def forward(self, encoded_input, labels=None):

        bert_output = self.bert(**encoded_input)

        cls_state = bert_output['last_hidden_state'][:, 0, :]

        predicted = self.dropout(cls_state)

        predicted = self.fc(predicted)
        if labels is not None:
            loss = self.CrossEntropyLoss(predicted.to('cpu'), labels)
            return loss
        else:
            predicted = self.Softmax(predicted)
            predicted_label = torch.argmax(predicted, dim=1)
            return predicted_label

