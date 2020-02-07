#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    Sequence Labeling Model.
"""
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.config import conf as conf
from model.modules.bilstm import Bilstm
from model.modules.crf import CRF
from model.modules.feature import CharFeature, WordFeature, PositionFeature
import numpy as np
class Bilstm_LR_Model(nn.Module):

    def __init__(self, word_embeddings, word_require_grad,
        char_embedding_shape, filter_sizes, filter_nums, 
        target_size, average_batch=True, use_cuda=True):
        """
        Args:
            feature_names: list(str), 特征名称, 不包括`label`和`char`

            feature_size_dict: dict({str: int}), 特征表大小字典
            feature_dim_dict: dict({str: int}), 输入特征dim字典
            pretrained_embed_dict: dict({str: np.array})
            require_grad_dict: bool, 是否更新feature embedding的权重

            # char parameters
            use_char: bool, 是否使用字符特征, default is False
            filter_sizes: list(int), 卷积核尺寸, default is [3]
            filter_nums: list(int), 卷积核数量, default is [32]

            # rnn parameters
            rnn_unit_type: str, options: ['rnn', 'lstm', 'gru']
            num_rnn_units: int, rnn单元数
            num_layers: int, 层数
            bi_flag: bool, 是否双向, default is True

            use_crf: bool, 是否使用crf层

            dropout_rate: float, dropout rate

            average_batch: bool, 是否对batch的loss做平均
            use_cuda: bool
        """
        super(Bilstm_LR_Model, self).__init__()
        word_embedding_shape = (len(word_embeddings),len(word_embeddings[0]))
        # word level feature layer
        self.word_feature_layer = WordFeature(word_embedding_shape, 
            word_embeddings, word_require_grad)

        self.char_feature_layer = CharFeature(char_embedding_shape, 
            filter_sizes, filter_nums)
        trans_input_dim = word_embedding_shape[1] + sum(filter_nums)
        # feature dropout
        self.dropout_feature = nn.Dropout(conf.dropout_rate)
        input_size = trans_input_dim
        # trans layer
        self.bilstm_layer = Bilstm(input_size, conf.trans_output_size)

        # trans dropout
        self.dropout_trans = nn.Dropout(conf.dropout_rate)

        # crf layer
        self.crf_layer = CRF(target_size, average_batch, use_cuda)

        # dense layer
        hidden_input_dim = conf.trans_output_size * 3
        ex_target_size = target_size + 2
        self.hidden2tag = nn.Linear(hidden_input_dim, ex_target_size)

        # loss
        self.loss_function = self.crf_layer.neg_log_likelihood_loss
        self.local_loss_function = nn.CrossEntropyLoss(reduce=False)
        self.local_loss_function_nil = self.CrossEntropyLoss_nil
        self.loss_function_ratio = self.crf_layer.neg_log_likelihood_loss_ratio
        self.loss_function_nil = self.crf_layer.neg_log_likelihood_loss_nil
        self.average_batch = average_batch
        self.begins = [0. for i in range(ex_target_size)]
        for i in conf.begin:
            self.begins[i] = 1.
        if conf.only_nil:
            self.begins[0] = 1.
        self.begins = torch.tensor(self.begins, dtype=torch.float32).cuda().view(1,ex_target_size)
        self.insides = [0. for i in range(ex_target_size)]
        for i in conf.inside:
            self.insides[i] = 1.
        if conf.only_nil:
            self.insides[0] = 1.
        self.insides = torch.tensor(self.insides, dtype=torch.float32).cuda().view(1,ex_target_size)
        
    def CrossEntropyLoss_nil(self, feats, tags):
        isBNil = (tags == conf.fuzzy2id['B-Nil'])
        isINil = (tags == conf.fuzzy2id['I-Nil'])
        notfuzzy = 1 - (isBNil + isINil)
        todiv = float(len(conf.begin))
        if conf.only_nil:
            todiv += 1
        loss = torch.sum(-feats*self.begins*(isBNil.float().unsqueeze(1))/todiv, dim=1)
        loss = loss+torch.sum(-feats*self.insides*(isINil.float().unsqueeze(1))/todiv, dim=1)
        temp = tags*notfuzzy.long()
        temp = temp.unsqueeze(1)

        temp = -torch.gather(feats,1,temp)
        temp = temp.squeeze()
        temp = temp*notfuzzy.float()
        loss = loss+temp
        loss = loss+ torch.log(torch.sum(torch.exp(feats), dim=1))
        return loss

    def loss(self, feats, mask, tags):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        loss_value = self.loss_function(feats, mask, tags)
        if self.average_batch:
            batch_size = feats.size(0)
            loss_value = loss_value/float(batch_size)
        return loss_value
    
    def fuzzy_loss(self, feats, mask, tags, locations, ratio = False):
        batch_size = tags.size(0)
        seq_len = tags.size(1)
        confirmed = np.zeros((batch_size,seq_len))
        for l in range(len(locations)):
            confirmed[l][locations[l]] = 1
        tags2 = tags*(torch.tensor(confirmed,dtype=torch.long).cuda())+torch.tensor(conf.NOT_CONFIRM_IDX*(1-confirmed),dtype=torch.long).cuda()
        
        if ratio:
            loss_value = self.loss_function_ratio(feats,mask,tags2,tags)
        else:
            loss_value = self.loss_function(feats,mask,tags2, fuzzy=True)
        if self.average_batch:
            loss_value = loss_value/float(batch_size)
        return loss_value

    def fuzzy_loss_nil(self, feats, mask, tags, locations):
        batch_size = tags.size(0)
        seq_len = tags.size(1)
        confirmed = np.zeros((batch_size,seq_len))
        for l in range(len(locations)):
            confirmed[l][locations[l]] = 1
        confirmed = torch.tensor(confirmed,dtype=torch.long).cuda()
        tags_notconf = tags*confirmed+conf.fuzzy2id['not_conf']*(1-confirmed)
        not_nil = (tags >= 0).long()

        tags_nofuzzy = tags*not_nil
        loss_value = self.loss_function_nil(feats,mask,tags_notconf,tags_nofuzzy)
        if self.average_batch:
            loss_value = loss_value/float(batch_size)
        return loss_value
    def local_loss(self, feats, tags, locations):
        seq_len = feats.size(1)
        flat_feats = feats.view(-1,feats.size(-1))
        flat_tags = tags.view(-1)
        
        if conf.use_nil:
            losses = self.local_loss_function_nil(flat_feats,flat_tags)
        else:
            losses = self.local_loss_function(flat_feats,flat_tags)
        flat_locations = []
        local_mask = torch.zeros(losses.size()).float().cuda()
        location_count = 0.
        for tempi in range(len(locations)):
            start = tempi * seq_len
            location_count = location_count+len(locations[tempi])
            for loc in locations[tempi]:
                local_mask[start + loc] = 1.
        losses = local_mask * losses
        local_losses = torch.sum(losses)/location_count
        return local_losses
    
    def weighted_local_loss(self,feats,tags,tags_np,locations,weight):
        seq_len = feats.size(1)
        flat_feats = feats.view(-1,feats.size(-1))
        flat_tags = tags.view(-1)
        tags_np = tags_np.reshape(-1)
        losses = self.local_loss_function(flat_feats,flat_tags)
        flat_locations = []
        local_mask = torch.zeros(losses.size()).float().cuda()
        location_count = 0.
        for tempi in range(len(locations)):
            start = tempi * seq_len
            location_count =location_count+ len(locations[tempi])
            for loc in locations[tempi]:
                local_mask[start + loc] = weight[tags_np[start+loc]]
        losses = local_mask * losses
        local_losses = torch.sum(losses)/location_count
        return local_losses
    def forward(self, word_input, char_input, mask, length):
        """
        Args:
             inputs: list
        """
        batch_size = word_input.size(0)
        max_len = word_input.size(1)

        # word level feature
        word_feature = self.word_feature_layer(word_input)

        # char level feature
        char_feature = self.char_feature_layer(char_input)

        try:
            word_feature = torch.cat([word_feature, char_feature], 2)
        except:
            print (word_feature.shape)
            print (char_feature.shape)
            print (word_input.shape)
            print (char_input.shape)
            print (mask.shape)
            print (word_input)
            print (char_input)
            print (mask)
            exit(0)
        word_feature = self.dropout_feature(word_feature)
        # transformer layer
        bilstm_outputs = self.bilstm_layer(word_feature, length)
        lefts = [torch.zeros(bilstm_outputs.size(0),1,bilstm_outputs.size(-1)).cuda()]
        rights = [torch.zeros(bilstm_outputs.size(0),1,bilstm_outputs.size(-1)).cuda()]
        for tempi in range(bilstm_outputs.size(1)-1):
            lefts.append(torch.max(lefts[-1],bilstm_outputs[:,tempi:tempi+1,:]))
        for tempi in range(bilstm_outputs.size(1)-1,0,-1):
            rights.append(torch.max(rights[-1],bilstm_outputs[:,tempi:tempi+1,:]))
        rights.reverse()
        trans_outputs_lr = torch.cat([torch.cat(lefts,dim=1),bilstm_outputs,torch.cat(rights,dim=1)],dim=2)
        trans_outputs_lr = self.dropout_trans(trans_outputs_lr.view(-1, trans_outputs_lr.size(-1)))
        trans_feats = self.hidden2tag(trans_outputs_lr)
        return trans_feats.view(batch_size, max_len, trans_feats.size(-1))

    def predict(self, bilstm_outputs, actual_lens, mask=None):
        batch_size = bilstm_outputs.size(0)
        tags_list = []
        path_score, best_paths = self.crf_layer(bilstm_outputs, mask)
        return best_paths.cpu().data.numpy()
    def local_predict(self, logits):
        return torch.argmax(logits,dim=2).view(-1)
