# -*- encoding: utf-8 -*-
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim

from model.config import conf as conf
from utils.counter_entity import PrecRecallCounter
import sys
import numpy as np

import codecs
import torch
import datetime
import json
class Fuzzy_Trainer(object):

    def __init__(self,**kwargs):
        """
        Args of data:
            data_iter_train: 训练数据迭代器
            data_iter_dev: 开发数据迭代器
            feature_names: list(str), 特征名称, 没有`label`和`char`
            use_char: bool, 是否使用char feature
            max_len_char: int, 单词最大长度

        Args of train:
            model: 初始化之后的模型
            optimizer: model arguments optimizer
            lr_decay: float, 学习率衰减率
            learning_rate: float, 初始学习率
            path_save_model: str, 模型保存的路径

            nb_epoch: int, 迭代次数上限
            max_patience: int, 开发集上连续mp次没有提升即停止训练
        #model, optimizer, learning_rate, lr_decay, path_save_model, max_patience,
        dataset,use_trans
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        self.current_patience = 0
        self.dev_loss = self.train_loss = 0.
        self.best_dev_loss = 1.e10
        self.num_epoch = 0
        self.local_num_epoch = 0
        self.evalu_num = 0
        self.local_evalu_num = 0
        self.best_f1_type = 0.
        self.local_best_epoch = 0
        self.best_epoch = 0
        self.batch_per_epoch_train = self.dataset.num_train/conf.batch_size + \
        int((self.dataset.num_train % conf.batch_size) > 0)
        self.batch_per_epoch_train = self.batch_per_epoch_train/conf.evaluate_times
        self.nofuzzy_batch_per_epoch_train = self.dataset.num_nofuzzy_train/conf.nofuzzy_batch_size + \
        int((self.dataset.num_nofuzzy_train % conf.nofuzzy_batch_size) > 0)

        print('batch_per_epoch_train',self.batch_per_epoch_train)
        print('nofuzzy_batch_per_epoch_train',self.nofuzzy_batch_per_epoch_train)
        self.local_batch_per_epoch_train = conf.local_batch_per_eva
        if conf.use_local:
            self.local_batch_per_epoch_train = self.dataset.num_local_train/conf.local_batch_size + \
            int((self.dataset.num_train % conf.local_batch_size) > 0)
            print('local_batch_per_epoch_train',self.local_batch_per_epoch_train)

        self.batch_per_eva = conf.batch_per_eva
        #self.batch_per_eva = self.batch_per_epoch_train/3
        self.local_batch_per_eva = conf.local_batch_per_eva
        self.local_step = self.prev_local_step = self.nofuzzy_step = self.prev_nofuzzy_step = 0
        self.use_trans = conf.use_trans
        self.record = {'full_f1':[[],[]],'full_f1_full':[[],[]],'full_f1_local':[[],[]],
        'local_f1':[[],[]],'local_f1_full':[[],[]],'local_f1_local':[[],[]],'train_loss':[]}
    def fit(self):
        """训练模型
        """
        self.step = 0
        self.current_patience = 0

        for batch in self.dataset.train_data:
            self.step += 1
            if (self.step % self.batch_per_epoch_train == 0) or self.eva:
                f1 = self.validate(self.dataset.get_valid_iter(),conf.valid_log_dir,'full_fullphase')
                if self.eva:
                    exit(0)
                self.record['full_f1'][0].append(f1)
                self.record['full_f1'][1].append(self.num_epoch)
                self.record['full_f1_full'][0].append(f1)
                self.record['full_f1_full'][1].append(self.num_epoch)
                
                print('training loss:',self.train_loss)
                self.record['train_loss'].append(self.train_loss)
                json.dump(self.record,open(conf.record_path,'w'))
                self.train_loss = 0
                if self.max_patience <= self.current_patience and conf.early_stop:
                    print('finished training! (early stopping, max patience: {0})'.format(self.max_patience))
                    print('best_f1',self.best_f1_type)
                    return
                if self.lr_decay != 0.:
                    self.optimizer = self.decay_learning_rate(self.num_epoch, self.learning_rate)
            
            if (self.step % (self.batch_per_epoch_train*3) == 0):
                if conf.joint:
                    self.nofuzzy_fit_step()
                else:
                    pass
            self.model.train()
            self.optimizer.zero_grad()
            x_w,x_c,y,length,locations = zip(*batch)
            x_w,x_c,y,mask = self.dataset.pad_sentence(x_w,x_c,y,length)
            x_w = self.tensor_from_numpy(np.array(x_w))
            x_c = self.tensor_from_numpy(np.array(x_c))
            y = self.tensor_from_numpy(np.array(y))
            length = self.tensor_from_numpy(np.array(length))
            mask = self.tensor_from_numpy(np.array(mask))
            
            if self.use_trans:
                logits = self.model(x_w,x_c,mask)
            else:
                logits = self.model(x_w,x_c,mask,length)
            if conf.use_fuzzy:
                if conf.use_nil:
                    loss = self.model.fuzzy_loss_nil(logits, mask, y, locations)
                else:
                    y = (y >= 0).long()*y
                    loss = self.model.fuzzy_loss(logits, mask, y, locations, ratio = False)
            else:
                y = (y >= 0).long()*y
                loss = self.model.loss(logits, mask, y)
            if loss.item() != loss.item():
                print ('x_w',x_w)
                print ('x_c',x_c)
                print ('y',y)
                print ('mask',mask)
                print ('length',length)
                print ('locations',locations)
                exit(0)
            self.train_loss = self.train_loss+loss.item()
            loss.backward()
            self.optimizer.step()
        print('finished training!','best_f1_type',self.best_f1_type)
    def nofuzzy_fit_step(self):
        right = total = 0.
        self.prev_nofuzzy_step = self.nofuzzy_step
        for nofuzzy_batch in self.dataset.nofuzzy_train_data:
            self.nofuzzy_step+=1
            if self.nofuzzy_step%self.nofuzzy_batch_per_epoch_train == 0:
                f1=self.validate(self.dataset.get_valid_iter(),conf.valid_log_dir,'nofuzzy_phase')
            self.model.train()
            self.optimizer.zero_grad()
            nofuzzy_x_w,nofuzzy_x_c,nofuzzy_y,nofuzzy_length = zip(*nofuzzy_batch)
            nofuzzy_x_w,nofuzzy_x_c,nofuzzy_y,nofuzzy_mask = self.dataset.pad_sentence(nofuzzy_x_w,nofuzzy_x_c,nofuzzy_y,nofuzzy_length)
            nofuzzy_y_ = np.array(nofuzzy_y)
            nofuzzy_x_w = self.tensor_from_numpy(np.array(nofuzzy_x_w))
            nofuzzy_x_c = self.tensor_from_numpy(np.array(nofuzzy_x_c))
            nofuzzy_y = self.tensor_from_numpy(np.array(nofuzzy_y))

            nofuzzy_length = self.tensor_from_numpy(np.array(nofuzzy_length))
            nofuzzy_mask = self.tensor_from_numpy(np.array(nofuzzy_mask))
            if self.use_trans:
                nofuzzy_logits = self.model(nofuzzy_x_w,nofuzzy_x_c,nofuzzy_mask)
            else:
                nofuzzy_logits = self.model(nofuzzy_x_w,nofuzzy_x_c,nofuzzy_mask,nofuzzy_length)            
            nofuzzy_y = (nofuzzy_y >= 0).long()*nofuzzy_y
            if self.use_trans:
                nofuzzy_loss = self.model.loss(nofuzzy_logits,nofuzzy_y)
            else:
                nofuzzy_loss = self.model.loss(nofuzzy_logits,nofuzzy_mask,nofuzzy_y)
            nofuzzy_loss.backward()
            self.optimizer.step()
            
            if self.nofuzzy_step % (3*self.nofuzzy_batch_per_epoch_train) == 0:
                break
        if self.nofuzzy_step == self.prev_nofuzzy_step:
            print('nofuzzy_step data all trained')
            self.dataset.nofuzzy_train_data = self.dataset.bucket_batch_iter(self.dataset.nofuzzy_train_data_,conf.nofuzyy_batch_size,conf.nofuzzy_num_epochs,True)
        
    def local_fit(self):
        self.step = 0
        right = total = 0.
        for batch in self.dataset.local_train_data:
            self.step += 1
            if (self.step % self.local_batch_per_epoch_train == 0) or self.eva:
                self.eva = False
                local_f1 = self.local_validate(self.dataset.get_valid_iter(),conf.local_valid_log_dir,'onlylocal',onlylocal=True)
                self.record['local_f1'][0].append(local_f1)
                self.record['local_f1'][1].append(self.local_num_epoch)
                self.record['train_loss'].append(self.train_loss)
                print('local training loss:',self.train_loss)
                json.dump(self.record,open(conf.record_path,'w'))
                self.train_loss = 0.
                if self.max_patience <= self.current_patience:
                    print('finished training! (early stopping, max patience: {0})'.format(self.max_patience))
                    print('best_f1_type',self.best_f1_type)
                    return
            self.model.train()

            self.local_optimizer.zero_grad()
            local_x_w,local_x_c,local_y,local_length,locations = zip(*batch)
            local_x_w,local_x_c,local_y,local_mask = self.dataset.pad_sentence(local_x_w,local_x_c,local_y,local_length)
            local_x_w = self.tensor_from_numpy(np.array(local_x_w))
            local_x_c = self.tensor_from_numpy(np.array(local_x_c))
            local_y = self.tensor_from_numpy(np.array(local_y))
            local_length = self.tensor_from_numpy(np.array(local_length))
            local_mask = self.tensor_from_numpy(np.array(local_mask))
            if self.use_trans:
                local_logits = self.model(local_x_w,local_x_c,local_mask)
            else:
                local_logits = self.model(local_x_w,local_x_c,local_mask,local_length)
            if conf.weighted_local_loss:
                local_loss = self.model.weighted_local_loss(local_logits,local_y,local_y_,locations,self.dataset.local_weight)
            else:
                local_loss = self.model.local_loss(local_logits,local_y,locations)
            self.train_loss += local_loss.item()
            local_loss.backward()
            self.local_optimizer.step()
        print('finished training!','best_f1_type',self.best_f1_type)
    def local_validate(self, dataset, log_dir, name, onlylocal=False):
        print('local validation')
        self.model.eval()
        self.local_dev_loss = 0.
        counter = PrecRecallCounter(len(conf.id2type),log_dir,name,self.local_num_epoch,self.dataset.entity_train)
        for test_batch in dataset:
            x_w_dev_,x_c_dev_,y_dev_,length_dev_ = zip(*test_batch)
            x_w_dev,x_c_dev,y_dev,mask_dev = self.dataset.pad_sentence(x_w_dev_,x_c_dev_,y_dev_,length_dev_)
            x_w_dev = self.tensor_from_numpy(np.array(x_w_dev))
            x_c_dev = self.tensor_from_numpy(np.array(x_c_dev))
            y_dev_ = y_dev
            y_dev = self.tensor_from_numpy(np.array(y_dev))
            length_dev = self.tensor_from_numpy(np.array(length_dev_))
            mask_dev = self.tensor_from_numpy(np.array(mask_dev))
            if self.use_trans:
                logits = self.model(x_w_dev,x_c_dev,mask_dev)
            else:
                logits = self.model(x_w_dev,x_c_dev,mask_dev,length_dev)

            logits[:,:,-1] = -1000.
            logits[:,:,-2] = -1000.
            prediction = self.model.local_predict(logits)
            prediction = prediction.view(y_dev.size(0),y_dev.size(1)).cpu().data.numpy()
            counter.count_sequence(prediction,y_dev_,length_dev_)
            counter.count_entity_overlap(prediction,y_dev_,length_dev_,x_w_dev_)
        counter.compute()
        counter.compute_entity()
        counter.output_entity()
        if counter.f1_type > self.best_f1_type:
            self.best_f1_type = counter.f1_type
            print('best local f1_type:',self.best_f1_type)
            self.current_patience = 0
            self.local_best_epoch = self.local_num_epoch
            if onlylocal:
                self.save_model(self.local_num_epoch)
                self.local_evaluate(self.dataset.get_test_iter(),conf.local_test_log_dir,name)
        elif onlylocal:
            self.current_patience += 1
            print('no improvement, current patience: {0} / {1}'.format(
                self.current_patience, self.max_patience), 'best_f1_type:', self.best_f1_type, 'best_epoch', self.local_best_epoch)
            print ('')
        self.local_num_epoch += 1
        return counter.f1_type
    def validate(self, dataset, log_dir,name):
        print('validation',self.num_epoch)
        self.model.eval()
        self.dev_loss = 0
        counter = PrecRecallCounter(len(conf.id2type),log_dir,name,self.num_epoch,self.dataset.entity_train)
        if self.eva:
            f = codecs.open(log_dir +'output'+str(self.num_epoch), 'w','utf-8')
        for test_batch in dataset:
            x_w_dev_,x_c_dev_,y_dev_,length_dev_ = zip(*test_batch)
            
            x_w_dev,x_c_dev,y_dev,mask_dev = self.dataset.pad_sentence(x_w_dev_,x_c_dev_,y_dev_,length_dev_)
            x_w_dev = self.tensor_from_numpy(np.array(x_w_dev))
            x_c_dev = self.tensor_from_numpy(np.array(x_c_dev))
            y_dev_ = y_dev
            y_dev = self.tensor_from_numpy(np.array(y_dev))
            length_dev = self.tensor_from_numpy(np.array(length_dev_))
            mask_dev = self.tensor_from_numpy(np.array(mask_dev))
            if self.use_trans:
                logits = self.model(x_w_dev,x_c_dev,mask_dev)
            else:
                logits = self.model(x_w_dev,x_c_dev,mask_dev,length_dev)
            y_dev = (y_dev >= 0).long()*y_dev
            loss = self.model.loss(logits, mask_dev, y_dev)
            prediction = self.model.predict(logits,length_dev,mask_dev)

            batch_size = len(length_dev_)
            if self.eva:
                for b in range(batch_size):
                    towrite = ''
                    for i in range(length_dev_[b]):
                        towrite += (self.dataset.id2word[x_w_dev_[b][i]]+':'+str(y_dev_[b][i])+':'+str(prediction[b][i])+' ')
                    
                    towrite += '\n'
                    f.write(towrite)
            counter.count_sequence(prediction,y_dev_,length_dev_)
            
            counter.count_entity_overlap(prediction,y_dev_,length_dev_,x_w_dev_)
            self.dev_loss += loss.item()
        print('dev_loss',self.dev_loss,'train_loss',self.train_loss)
        counter.compute()
        counter.compute_entity()
        counter.output_entity()
        if counter.f1_type > self.best_f1_type:
            self.current_patience = 0
            self.best_f1_type = counter.f1_type
            self.best_epoch = self.num_epoch
            # 保存模型
            print('best_f1_type',self.best_f1_type)
            self.save_model(self.num_epoch)
            print('model has saved to {0}!'.format(self.path_save_model+'_'+str(self.num_epoch)))
            print ('')
            self.evaluate(self.dataset.get_test_iter(),conf.test_log_dir,name)
        else:
            self.current_patience += 1
            print('no improvement, current patience: {0} / {1}'.format(
                self.current_patience, self.max_patience), 'best_f1_type:', self.best_f1_type, 'best_epoch', self.best_epoch)
            print ('')
        self.num_epoch += 1
        return counter.f1_type
    def evaluate(self, dataset, log_dir, name):
        self.model.eval()
        self.evalu_num += 1
        print('evaluation')
        valid_step = 0
        valid_loss = 0
        counter = PrecRecallCounter(len(conf.id2type),log_dir,name,self.num_epoch*100+self.evalu_num,self.dataset.entity_train)
        f_result = open('result_output','w')
        for test_batch in dataset:
            valid_step += 1
            x_w_dev_,x_c_dev_,y_dev_,length_dev_  = zip(*test_batch)
            x_w_dev,x_c_dev,y_dev,mask_dev = self.dataset.pad_sentence(x_w_dev_,x_c_dev_,y_dev_,length_dev_)
            x_w_dev = self.tensor_from_numpy(np.array(x_w_dev))
            x_c_dev = self.tensor_from_numpy(np.array(x_c_dev))
            y_dev_ = y_dev
            y_dev = self.tensor_from_numpy(np.array(y_dev))

            length_dev = self.tensor_from_numpy(np.array(length_dev_))
            mask_dev = self.tensor_from_numpy(np.array(mask_dev))
            if self.use_trans:
                logits = self.model(x_w_dev,x_c_dev,mask_dev)
            else:
                logits = self.model(x_w_dev,x_c_dev,mask_dev,length_dev)
            y_dev = (y_dev >= 0).long()*y_dev
            loss = self.model.loss(logits, mask_dev, y_dev)
            prediction = self.model.predict(logits,length_dev,mask_dev)
            counter.count_sequence(prediction,y_dev_,length_dev_)
            counter.count_entity_overlap(prediction,y_dev_,length_dev_,x_w_dev_)
            valid_loss += loss.item()
            batch_size = len(length_dev_)
            for b in range(batch_size):
                pb = prediction[b].tolist()[:length_dev_[b]]
                yb = y_dev_[b][:length_dev_[b]]
                for w in range(length_dev_[b]):
                    pb[w] = str(pb[w])
                    yb[w] = str(yb[w])
                f_result.write(' '.join(pb)+'\t'+' '.join(yb)+'\n')
        print('evaluate_loss',valid_loss)
        counter.compute()
        counter.compute_entity()
        counter.output_entity()
        print ('')
        return counter.f1_type
    def local_evaluate(self, dataset, log_dir, name):
        print('local evaluation')
        self.model.eval()
        self.local_evalu_num += 1
        self.local_dev_loss = 0.

        counter = PrecRecallCounter(len(conf.id2type),log_dir,name,self.local_num_epoch*100+self.local_evalu_num,self.dataset.entity_train)
        for test_batch in dataset:
            x_w_dev_,x_c_dev_,y_dev_,length_dev_ = zip(*test_batch)
            x_w_dev,x_c_dev,y_dev,mask_dev = self.dataset.pad_sentence(x_w_dev_,x_c_dev_,y_dev_,length_dev_)
            x_w_dev = self.tensor_from_numpy(np.array(x_w_dev))
            x_c_dev = self.tensor_from_numpy(np.array(x_c_dev))
            y_dev_ = y_dev
            y_dev = self.tensor_from_numpy(np.array(y_dev))
            length_dev = self.tensor_from_numpy(np.array(length_dev_))
            mask_dev = self.tensor_from_numpy(np.array(mask_dev))
            if self.use_trans:
                logits = self.model(x_w_dev,x_c_dev,mask_dev)
            else:
                logits = self.model(x_w_dev,x_c_dev,mask_dev,length_dev)

            logits[:,:,-1] = -1000.
            logits[:,:,-2] = -1000.
            prediction = self.model.local_predict(logits)
            prediction = prediction.view(y_dev.size(0),y_dev.size(1)).cpu().data.numpy()
            counter.count_sequence(prediction,y_dev_,length_dev_)
            counter.count_entity_overlap(prediction,y_dev_,length_dev_,x_w_dev_)
        counter.compute()
        counter.compute_entity()
        counter.output_entity()
        print ('')
        return counter.f1_type
    def predict(self, data_iter, has_label=False):
        """预测
        Args:
            data_iter: 数据迭代器
            has_label: bool, 是否带有label

        Returns:
            labels: list of int
        """
        labels_pred, labels_gold = [], []
        for batch in data_iter:
            if has_label:
                x_w,x_c,y,length  = zip(*batch)
                x_w,x_c,y,mask = self.dataset.pad_sentence(x_w,x_c,y,length)
                labels_gold_batch = y
                labels_gold.extend(labels_gold_batch)
                y = self.tensor_from_numpy(np.array(y))
            else:
                x_w,x_c,length = zip(*batch)
                y = copy.deepcopy(x_w)
                x_w,x_c,y,mask = self.dataset.pad_sentence(x_w,x_c,y,length)
            x_w = self.tensor_from_numpy(np.array(x_w))
            x_c = self.tensor_from_numpy(np.array(x_c))
            length_ = length
            length = self.tensor_from_numpy(np.array(length))
            mask = self.tensor_from_numpy(np.array(mask))
            if self.use_trans:
                logits = self.model(x_w,x_c,mask)
            else:
                logits = self.model(x_w,x_c,mask,length)
            labels_batch = self.model.predict(logits, length, mask)

            for tempi in range(len(labels_batch)):
                labels_batch[tempi] = labels_batch[tempi][:length_[tempi]]
            labels_pred.extend(labels_batch)
        if has_label:
            return labels_gold, labels_pred
        return labels_pred

    def decay_learning_rate(self, epoch, init_lr):
        """衰减学习率

        Args:
            epoch: int, 迭代次数
            init_lr: 初始学习率
        """
        lr = init_lr / (1+self.lr_decay*epoch)
        print('learning rate: {0}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    @staticmethod
    def tensor_from_numpy(data, dtype='long', use_cuda=True):
        """将numpy转换为tensor
        Args:
            data: numpy
            dtype: long or float
            use_cuda: bool
        """
        assert dtype in ('long', 'float')
        if dtype == 'long':
            data = torch.from_numpy(data).long()
        else:
            data = torch.from_numpy(data).float()
        if use_cuda:
            data = data.cuda()
        return data
    def save_model(self,epoch):
        """保存模型
        """
        torch.save(self.model.state_dict(), self.path_save_model+'_'+str(epoch))

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.data_iter_train.batch_size = batch_size
        self.data_iter_dev.batch_size = batch_size

    def set_max_patience(self, max_patience):
        self.max_patience = max_patience

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate
