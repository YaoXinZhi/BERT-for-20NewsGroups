# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 29/05/2021 23:09
@Author: yao
"""

import re
import logging

from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

# from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)


class News_Dataset(Dataset):
    def __init__(self, paras, mode:str='train'):
        if mode == 'train':
            logger.info(f'loading {mode} dataset.')
            self.mode = mode
            self.data_file = paras.train_data
        elif mode == 'test':
            logger.info(f'loading {mode} dataset.')
            self.mode = mode
            self.data_file = paras.test_data
        else:
            raise ValueError(f'mode must be "train" or "test",'
                           f' but got "{mode}".')

        self.label_file = paras.label_file

        self.data = []
        self.label = []

        self.label_set = set()
        self.label_to_index = {}
        self.index_to_label = {}

        self.load_label()
        # print(self.label_to_index)
        self.load_data()
        logger.info(f'{self.mode} data loaded, data size: {len(self.data):,}.')
        logger.info(f'label count: {len(self.label):,}.')

        if (not paras.train_size is None) and self.mode == 'train':
            self.data = self.data[:paras.train_size]
            self.label = self.label[:paras.train_size]
        if (not paras.test_size is None) and self.mode == 'test':
            self.data = self.data[:paras.test_size]
            self.label = self.label[:paras.test_size]


    def load_label(self):

        with open(self.label_file) as f:
            for line in f:
                label = line.strip()
                self.label_set.add(label)
                label_idx = len(self.label_to_index)
                self.label_to_index[label] = label_idx
                self.index_to_label[label_idx] = label

    @staticmethod
    def data_preprocess(text: str):
        text = text.replace('\n', ' ')
        text = re.sub(r' +', ' ', text)
        return text

    def load_data(self):

        with open(self.data_file) as f:
            for line in f:
                l = line.strip().split('\t')
                # if len(l) != 2:
                #     print(l)
                #     exit()
                try:
                    data = ' '.join(l[:-1])
                    label = l[-1]
                    label = int(label)
                    if label >= 20:
                        raise TypeError
                except:
                    print(l)
                    continue

                data_len = len(data.split())
                if data_len > 500:
                    continue
                data = self.data_preprocess(data)
                self.data.append(data)
                # fixme: label index may not right
                self.label.append(self.index_to_label[label])
                # self.label.append(label)
                # self.label.append(label)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

if __name__ == '__main__':
    args = config()

    test_dataset = News_Dataset(args, 'test')
