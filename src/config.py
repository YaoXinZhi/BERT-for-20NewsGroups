# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 12/04/2021 11:06
@Author: XINZHI YAO
"""

import logging
import torch
from transformers import WEIGHTS_NAME, CONFIG_NAME

class config:
    def __init__(self):

        # global parameters
        self.mode = 'train'
        self.seed = 26
        self.use_cpu = False
        self.device = 'cuda' if torch.cuda.is_available() and not self.use_cpu else 'cpu'

        # Preprocessed data

        self.train_data = '../data/20news.train.txt'
        self.test_data = '../data/20news.test.txt'

        self.label_file = '../data/label.txt'

        # Data loading parameters
        self.unknown_token = '[UNK]'
        # fixme: max_sequence_length
        # self.max_sequence_length = 512
        self.max_sequence_length = 512
        self.load_denotation = True
        self.add_denotation_span = True

        self.train_size = 2000
        self.test_size = 1000

        # Model initialization parameters
        self.model_name = 'bert-base-cased'
        self.hidden_size = 768
        self.label_number = 20
        self.dropout_prob = 0.3

        # Training parameters
        self.num_train_epochs = 20
        self.batch_size = 10
        self.shuffle = True
        self.drop_last = False

        # Optimizer parameters
        self.optimizer = 'Adam'
        self.learning_rate = 2e-5
        self.adam_epsilon = 1e-8

        # Logging parameters.
        self.save_log_file = False
        self.logging_level = logging.NOTSET
        self.log_save_path = '../logging'
        self.model_save_path = '../models'

        self.checkpoint_file = 'checkpoint.news.txt'
        self.train_log_file = 'BertClassifier.news.log'

        self.model_save_name = WEIGHTS_NAME
        self.config_save_name = CONFIG_NAME

Best F1-score: 0.9843
Precision: 0.9892
Recall: 0.9795
Accuracy: 0.9791
Epoch-Average Loss: 0.6392
