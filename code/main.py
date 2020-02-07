# -*- encoding: utf-8 -*-

import model.config.conf as conf
#conf to change: main,trainer,counter,data_loader,crf,model
import torch
import torch.nn as nn
import torch.optim as optim

from model.modules.bilstm_model import Bilstm_LR_Model
from utils.data_loader import DataSet

from utils.fuzzy_trainer import Fuzzy_Trainer
import datetime
import numpy as np
import os
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def main():
    np.random.seed(77777)
    torch.manual_seed(77777)
    torch.cuda.manual_seed(77777)
    restore = False
    eva = False
    dataset = DataSet(conf.embedding_path,conf.alphabet_path,
        conf.train_path,conf.test_path,conf.valid_path,conf.local_train_path)

    model = Bilstm_LR_Model(word_embeddings=dataset.word_embeddings, word_require_grad=False,
            char_embedding_shape=(len(dataset.alphabet2id),conf.char_dim), filter_sizes=conf.filter_sizes,
            filter_nums=conf.filter_nums, target_size = len(conf.type2id), average_batch=True, use_cuda=True)
    model = model.cuda()
    if restore:
        model.load_state_dict(torch.load('../log/saved_models/model_10'))
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate, weight_decay=conf.l2_rate)
    local_parameters = parameters       
    local_optimizer = optim.Adam(model.parameters(), lr=conf.local_learning_rate, weight_decay=conf.l2_rate)
    
    trainer = Fuzzy_Trainer(model=model, optimizer=optimizer, local_optimizer=local_optimizer, learning_rate=conf.learning_rate,
        local_learning_rate=conf.local_learning_rate,lr_decay=conf.lr_decay,local_lr_decay=conf.local_lr_decay,
        path_save_model=conf.path_save_model, max_patience=conf.max_patience, local_max_patience=conf.local_max_patience, dataset=dataset,
        eva = eva)
    start = datetime.datetime.now()
    if conf.use_local:
        trainer.local_fit()
    trainer.fit()
    end = datetime.datetime.now()
    print(end-start)
    fout = open(conf.valid_log_dir+'time','w')
    fout.write(str(end-start))
if __name__ == '__main__':
    for th in [0.1]:
    conf.test_log_dir = '../log/test/'
    create_dir(conf.test_log_dir)
    conf.valid_log_dir = '../log/valid/'
    create_dir(conf.valid_log_dir)
    conf.local_test_log_dir = '../log/local_test/'
    create_dir(conf.local_test_log_dir)
    conf.local_valid_log_dir = '../log/local_valid/'
    create_dir(conf.local_valid_log_dir)
    conf.path_save_model = '../log/saved_models/'
    create_dir(conf.path_save_model)
    conf.record_path = '../log/record'
    
    conf.train_path = '../files/train.txt'
    conf.test_path = '../files/test.txt'
    conf.valid_path = '../files/valid.txt'
    conf.local_train_path = '../files/local_train.txt'
    conf.nofuzzy_train_path = '../files/nofuzzy_train.txt'
    conf.embedding_path = '../files/word_embedding'
    conf.entity_dict = '../files/entity_dict'
    conf.word_idf_dict = '../files/word_idf_dict'
    conf.alphabet_path = '../files/alphabet'
    conf.sample_ratio = 0.9
    conf.local_sample_ratio = 0.9
    conf.use_local = True
    conf.use_fuzzy = True
    conf.joint = True
    conf.use_nil = False
    main()