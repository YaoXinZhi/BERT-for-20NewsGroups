# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 13/04/2021 15:43
@Author: XINZHI YAO
"""

import os
import time
import random
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AdamW

#from sklearn import metrics
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import BertClassifier
from utils import *
from news_dataloader import News_Dataset
from config import config

# from src.model import BertClassifier
# from src.utils import *
# from src.dataloader import RE_Dataset
# from src.config import config


def evaluation(model: BertModel, tokenizer: BertTokenizer, dataloader: DataLoader,
               max_sequence_length: int, label_to_index: dict, device='cuda'):
    model.eval()
    total_pred_label = []
    total_ture_label = []
    with torch.no_grad():
        for batch in tqdm(dataloader):

            batch_data, batch_label = batch
            label_idx = batch_label_to_idx(batch_label, label_to_index, False)
            encoded_data = tokenizer(batch_data,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt',
                                     max_length=max_sequence_length)
            encoded_data.to(device)
            predict_label = model(encoded_data)
            predict_label = tensor_to_list(predict_label.to('cpu'))

            total_ture_label.extend(label_idx)
            total_pred_label.extend(predict_label)

        precision, recall, acc, f1 = fuck_metric(total_ture_label, total_pred_label)
 

    return acc, precision, recall, f1


def _save_model(save_path: str, model: BertModel, tokenizer: BertTokenizer,
                model_save_name: str, config_save_name: str):

    model_save_Path = os.path.join(save_path, model_save_name)
    config_save_path = os.path.join(save_path, config_save_name)

    # torch.save(model.state_dict(), model_save_Path)
    model_to_save = model.module if hasattr(model, 'module') else model

    torch.save(model_to_save.state_dict(), model_save_Path)
    model_to_save.config.to_json_file(config_save_path)
    tokenizer.save_vocabulary(save_path)

def _load_fine_tuning_model(model: BertModel, save_path: str, model_save_name: str):

    model_save_path = os.path.join(save_path, model_save_name)
    model.load_state_dict(torch.load(model_save_path))

    tokenizer = BertTokenizer.from_pretrained(save_path)
    return model, tokenizer

def _load_pre_trained_model(paras):
    tokenizer = BertTokenizer.from_pretrained(paras.model_name)
    bert_config = BertConfig.from_pretrained(paras.model_name)
    bert = BertModel.from_pretrained(paras.model_name, output_hidden_states=True)
    model = BertClassifier(bert_config, bert, paras.hidden_size, paras.label_number,
                           paras.dropout_prob)
    return model, tokenizer

def _train(paras):

    logger = logging.getLogger(__name__)
    if paras.save_log_file:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=paras.logging_level,
                            filename=f'{paras.log_save_path}/{paras.mode}-{paras.train_log_file}',
                            filemode='w')
    else:
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=paras.logging_level,)

    device = 'cuda' if torch.cuda.is_available() and (not paras.use_cpu) else 'cpu'
    if device == 'cuda':
        logging.info('----- cuda for training -----')
    else:
        logging.info('----- cpu for training -----')    


    logger.info(f'Loading model: {paras.model_name}')

    bert_classifier, tokenizer = _load_pre_trained_model(paras)
    bert_classifier.to(device)

    train_dataset = News_Dataset(paras, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=paras.batch_size,
                                 shuffle=paras.shuffle, drop_last=paras.drop_last)
    label_to_index = train_dataset.label_to_index
    index_to_label = train_dataset.index_to_label


    test_dataset = News_Dataset(paras, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=paras.batch_size,
                                 shuffle=paras.shuffle, drop_last=paras.drop_last)

    if paras.optimizer.lower() == 'adam':
        logger.info('Loading Adam optimizer.')
        optimizer = torch.optim.Adam(bert_classifier.parameters(), lr=paras.learning_rate)
    elif paras.optimizer.lower() == 'adamw':
        logger.info('Loading AdamW optimizer.')
        no_decay = [ 'bias', 'LayerNorm.weight' ]
        optimizer_grouped_parameters = [
            {'params': [ p for n, p in bert_classifier.named_parameters() if not any(nd in n for nd in no_decay) ],
             'weight_decay': 0.01},
            {'params': [ p for n, p in bert_classifier.named_parameters() if any(nd in n for nd in no_decay) ],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=paras.learning_rate,
                          eps=args.adam_epsilon)
    else:
        logger.warning(f'optimizer must be "Adam" or "AdamW", but got {paras.optimizer}.')
        logger.info('Loading Adam optimizer.')
        optimizer = torch.optim.Adam(bert_classifier.parameters(),
                                     lr=paras.learning_rate)

    logger.info('Training Start.')
    best_eval = {'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'loss': 0}
    for epoch in range(paras.num_train_epochs):
        epoch_loss = 0
        bert_classifier.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch_data, batch_label = batch
            print(batch_data)
            print(batch_label)
            input('continue')
            encoded_data = tokenizer(batch_data,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt',
                                     max_length=paras.max_sequence_length)
            encoded_data.to(device)

            label_tensor = batch_label_to_idx(batch_label, label_to_index)
            label_tensor.to(device)

            loss = bert_classifier.forward(encoded_data, label_tensor)

            epoch_loss += loss_to_int(loss)

            logging.info(f'epoch: {epoch}, step: {step}, loss: {loss:.4f}')

            loss.backward()
            optimizer.step()

            # logging.debug('evaluationg.')
            # evaluation(bert_classifier, tokenizer, test_dataloader,
            #            paras.max_sequence_length, label_to_index)

        epoch_loss = epoch_loss / len(train_dataloader)

        logging.info('Evaluating.')
        acc, precision, recall, f1 = evaluation(bert_classifier, tokenizer, test_dataloader,
                                                paras.max_sequence_length, label_to_index)

        logging.info(f'Epoch: {epoch}, Epoch-Average Loss: {epoch_loss:.4f}')
        logger.info(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, '
                    f'Recall: {recall:.4f}, F1-score: {f1:.4f}')

        if best_eval['loss'] == 0 or f1 > best_eval['f1']:
            best_eval['loss'] = epoch_loss
            best_eval['acc'] = acc
            best_eval['precision'] = precision
            best_eval['recall'] = recall
            best_eval['f1'] = f1
            _save_model(paras.model_save_path, bert_classifier, tokenizer, paras.model_save_name,
                        paras.config_save_name)


            with open(f'{paras.log_save_path}/{paras.checkpoint_file}', 'w') as wf:
                wf.write(f'Save time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
                wf.write(f'Best F1-score: {best_eval["f1"]:.4f}\n')
                wf.write(f'Precision: {best_eval["precision"]:.4f}\n')
                wf.write(f'Recall: {best_eval["recall"]:.4f}\n')
                wf.write(f'Accuracy: {best_eval["acc"]:.4f}\n')
                wf.write(f'Epoch-Average Loss: {best_eval["loss"]:.4f}\n')

            logger.info(f'Updated model, best F1-score: {best_eval["f1"]:.4f}\n')

    logger.info(f'Train complete, Best F1-score: {best_eval["f1"]:.4f}.')

if __name__ == '__main__':
    args = config()

    set_seed(args.seed)

    if args.mode == 'train':
        _train(args)
    else:
        raise TypeError(f'mode must be "train", but get "{args.mode}".')
