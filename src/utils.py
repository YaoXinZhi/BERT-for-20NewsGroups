# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 15:45
@Author: XINZHI YAO
"""

import os
import torch
import random
import numpy as np


def batch_label_to_idx(batch_label: list, label_to_index: dict,
                       return_tensor=True):

    label_idx = [label_to_index[label] for label in batch_label]

    if return_tensor:
        return torch.LongTensor(label_idx)
    else:
        return label_idx

def loss_to_int(loss):
    return loss.detach().cpu().item()

def tensor_to_list(tensor):
    return tensor.numpy().tolist()

def input_ids_to_token(tokenizer, input_ids):

    token_list = [''.join(tokenizer.decode(idx).split()) for idx in input_ids]
    return token_list

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fuck_metric(y_true: list, y_pred: list):
    TP = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 1)))
    FP = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)))
    FN = np.sum(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)))
    TN = np.sum(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 0)))

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    acc = (TP+TN)/(TP+FP+TN+FN)
    f1 = 2*precision*recall / (precision+recall)

    return precision, recall, acc, f1
