#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from model.config import conf as conf
def log_sum_exp(vec, m_size):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim

    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)

def log_sum_exp_fuzzy(vec, m_size, before_mask, current_mask):
    """
    Args:
        vec: size=(batch_size, vanishing_dim, hidden_dim)
        m_size: hidden_dim
        before_mask: batch_size, vanishing_dim, 1
        current_mask: batch_size, vanishing_dim, hidden_dim
    Returns:
        size=(batch_size, hidden_dim)
    """
    _, idx = torch.max(vec, 1)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
    #print (vec.is_cuda, max_score.is_cuda, before_mask.is_cuda, current_mask.is_cuda)
    if torch.sum(torch.sum(before_mask*current_mask,1) == 0) != 0:
        print (before_mask)
        print (current_mask)
        print (before_mask*current_mask)
    score_exp = torch.exp(vec - max_score.expand_as(vec))*before_mask*current_mask
    return max_score.view(-1, m_size) + torch.log(torch.sum(score_exp, 1)).view(-1, m_size)

class CRF(nn.Module):

    def __init__(self, target_size, average_batch = True, use_cuda = True):
        """
        Args:
            target_size: int, target size
            use_cuda: bool, 是否使用gpu, default is True
            average_batch: bool, loss是否作平均, default is True
        """
        super(CRF, self).__init__()
        self.target_size = target_size
        self.use_cuda = use_cuda
        self.average_batch = average_batch
        # init transitions
        self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
        self.NOT_CONFIRM_IDX = -3
        init_transitions = torch.zeros(self.target_size+2, self.target_size+2)
        init_transitions[:, self.START_TAG_IDX] = -1000.
        init_transitions[self.END_TAG_IDX, :] = -1000.
        if self.use_cuda:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)
        all_fuzzy = [1. for i in range(self.target_size+2)]
        begin_fuzzy = [0. for i in range(self.target_size+2)]
        for i in conf.begin:
            begin_fuzzy[i] = 1.
        inside_fuzzy = [0. for i in range(self.target_size+2)]
        for i in conf.inside:
            inside_fuzzy[i] = 1.
        if conf.only_nil:
            begin_fuzzy[0] = inside_fuzzy[0] = 1.
        self.before_masks = torch.tensor([all_fuzzy, begin_fuzzy, inside_fuzzy],dtype=torch.float).view(3,self.target_size+2,1)
        self.before_masks = self.before_masks.cuda()
        current_masks = [[1. for j in range(self.target_size+2)]for i in range(self.target_size+2)]
        common = [[1. for j in range(self.target_size+2)]for i in range(self.target_size+2)]
        for i in range(self.target_size+2):
            for j in conf.inside:
                if i != j and i != j-1:
                    current_masks[i][j] = 0.
        if conf.only_nil:
            for i in range(1,self.target_size+2):
                current_masks[i][0] = 0.
            current_masks[0][0] = 1.
        self.current_masks = torch.tensor([common, common, current_masks],dtype=torch.float)
        self.current_masks = self.current_masks.cuda()


    def _forward_alg(self, feats, mask):
        """
        Do the forward algorithm to compute the partition function (batched).

        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            xxx
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)
        mask = mask.byte()
        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats
        temp = self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores + temp
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)

            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)

            masked_cur_partition = cur_partition.masked_select(mask_idx)
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                partition.masked_scatter_(mask_idx, masked_cur_partition)

        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
            batch_size, tag_size, tag_size) + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, self.END_TAG_IDX]
        return final_partition.sum(), scores
    def fuzzy_forward(self,feats,mask,tags):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            xxx
        """

        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)
        mask = mask.byte()
        mask = mask.transpose(1, 0).contiguous()
        tags = tags.transpose(1,0).contiguous()
        isconfirmed = (tags != self.NOT_CONFIRM_IDX).long()
        notconfirmed = (tags == self.NOT_CONFIRM_IDX).long()      
        tags_2 = tags -self.NOT_CONFIRM_IDX*notconfirmed
        ins_num = batch_size * seq_len

        new_tags = Variable(torch.LongTensor(seq_len, batch_size))
        if self.use_cuda:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[0, :] = (tag_size - 2) * tag_size + tags_2[0, :]
            else:
                new_tags[idx,:] = tags_2[idx-1,:] * tag_size + tags_2[idx,:]
        new_tags = new_tags.view(seq_len, batch_size, 1)

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats
        temp = self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores + temp
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        new_scores = scores.view(seq_len, batch_size, -1)
        tg_energy = torch.gather(new_scores, 2, new_tags).view(seq_len, batch_size)

        seq_iter = enumerate(scores)
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        notconf_partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        conf_partition = torch.gather(notconf_partition.view(batch_size,tag_size),1,tags_2[0,:].view(batch_size,1))
        
        for idx,cur_values in seq_iter:
            conf_notconf = conf_partition.view(batch_size,1,1).expand(batch_size,1,tag_size) + \
            torch.gather(cur_values,1,tags_2[idx-1,:].view(batch_size,1,1).expand(batch_size,1,tag_size))
            conf_notconf = conf_notconf.transpose(1,2).contiguous()

            cur_values = cur_values + notconf_partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            notconf_notconf = log_sum_exp(cur_values, tag_size).view(batch_size,tag_size,1)
            temp_notconf = isconfirmed[idx-1,:].view(batch_size,1,1).expand(batch_size,tag_size,1).float()*conf_notconf\
             + notconfirmed[idx-1,:].view(batch_size,1,1).expand(batch_size,tag_size,1).float()*notconf_notconf

            isconf_idx = tags_2[idx,:].view(batch_size,1,1)
            notconf_conf = torch.gather(notconf_notconf,1,isconf_idx).view(batch_size,1)
            conf_conf = conf_partition + tg_energy[idx,:].view(batch_size,1)
            temp_conf = isconfirmed[idx-1,:].view(batch_size,1).float()*conf_conf +\
                    notconfirmed[idx-1,:].view(batch_size,1).float()*notconf_conf

            mask_idx = mask[idx, :].view(batch_size, 1, 1).expand(batch_size, tag_size, 1)
            masked_temp_notconf = temp_notconf.masked_select(mask_idx)
            if masked_temp_notconf.dim() != 0:
                notconf_partition.masked_scatter_(mask_idx, masked_temp_notconf)

            mask_idx = mask[idx, :].view(batch_size, 1)
            masked_temp_conf = temp_conf.masked_select(mask_idx)
            if masked_temp_conf.dim() != 0:
                conf_partition.masked_scatter_(mask_idx, masked_temp_conf)
        
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        cur_values_notconf = cur_values + notconf_partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        final_notconf_partition = log_sum_exp(cur_values_notconf, tag_size)[:, self.END_TAG_IDX].view(batch_size,1)
        
        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=0).view(1,batch_size).long()
        end_ids = torch.gather(tags_2, 0, length_mask-1).view(batch_size,1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        final_conf_partition = conf_partition + end_energy
        final_confirmed = torch.gather(isconfirmed,0,length_mask-1).view(batch_size,1).float()
        final_partition = final_confirmed*final_conf_partition+(1-final_confirmed)*final_notconf_partition
        return final_partition.sum()
    def fuzzy_forward_nil(self, feats, mask, tags_notconf, tags_nofuzzy):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            xxx
        """

        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)
        mask = mask.byte()
        mask = mask.transpose(1, 0).contiguous()
        tags_notconf = tags_notconf.transpose(1,0).contiguous()
        tags_nofuzzy = tags_nofuzzy.transpose(1,0).contiguous()

        isconfirmed = (tags_notconf >= 0).long()
        notconfirmed = (tags_notconf < 0).long()      
        ins_num = batch_size * seq_len
        tags_notconf = (tags_notconf-conf.fuzzy2id['not_conf'])*notconfirmed

        all_currents = self.current_masks[tags_notconf]
        all_befores = self.before_masks[tags_notconf]
        new_tags = Variable(torch.LongTensor(seq_len, batch_size))
        if self.use_cuda:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[0, :] = (tag_size - 2) * tag_size + tags_nofuzzy[0, :]
            else:
                new_tags[idx,:] = tags_nofuzzy[idx-1,:] * tag_size + tags_nofuzzy[idx,:]
        new_tags = new_tags.view(seq_len, batch_size, 1)

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats
        temp = self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores + temp
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        new_scores = scores.view(seq_len, batch_size, -1)
        tg_energy = torch.gather(new_scores, 2, new_tags).view(seq_len, batch_size)

        seq_iter = enumerate(scores)
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        notconf_partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        conf_partition = torch.gather(notconf_partition.view(batch_size,tag_size),1,tags_nofuzzy[0,:].view(batch_size,1))
        
        for idx,cur_values in seq_iter:
            conf_notconf = conf_partition.view(batch_size,1,1).expand(batch_size,1,tag_size) + \
            torch.gather(cur_values,1,tags_nofuzzy[idx-1,:].view(batch_size,1,1).expand(batch_size,1,tag_size))
            conf_notconf = conf_notconf.transpose(1,2).contiguous()

            cur_values = cur_values + notconf_partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            before_mask = all_befores[idx-1]
            current_mask = all_currents[idx]
            notconf_notconf = log_sum_exp_fuzzy(cur_values, tag_size, before_mask, current_mask).view(batch_size,tag_size,1)
            temp_notconf = isconfirmed[idx-1,:].view(batch_size,1,1).expand(batch_size,tag_size,1).float()*conf_notconf\
             + notconfirmed[idx-1,:].view(batch_size,1,1).expand(batch_size,tag_size,1).float()*notconf_notconf

            isconf_idx = tags_nofuzzy[idx,:].view(batch_size,1,1)
            notconf_conf = torch.gather(notconf_notconf,1,isconf_idx).view(batch_size,1)
            conf_conf = conf_partition + tg_energy[idx,:].view(batch_size,1)
            temp_conf = isconfirmed[idx-1,:].view(batch_size,1).float()*conf_conf +\
                    notconfirmed[idx-1,:].view(batch_size,1).float()*notconf_conf

            mask_idx = mask[idx, :].view(batch_size, 1, 1).expand(batch_size, tag_size, 1)
            masked_temp_notconf = temp_notconf.masked_select(mask_idx)
            if masked_temp_notconf.dim() != 0:
                notconf_partition.masked_scatter_(mask_idx, masked_temp_notconf)

            mask_idx = mask[idx, :].view(batch_size, 1)
            masked_temp_conf = temp_conf.masked_select(mask_idx)
            if masked_temp_conf.dim() != 0:
                conf_partition.masked_scatter_(mask_idx, masked_temp_conf)
        
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        cur_values_notconf = cur_values + notconf_partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        length_mask = torch.sum(mask, dim=0).view(1,batch_size).long()
        fuzzy_end_ids = torch.gather(tags_notconf, 0, length_mask-1).view(batch_size)
        before_mask = self.before_masks[fuzzy_end_ids]
        current_mask = self.current_masks[[0 for temp in range(batch_size)]]
        final_notconf_partition = log_sum_exp_fuzzy(cur_values_notconf, tag_size, before_mask, current_mask)[:, self.END_TAG_IDX].view(batch_size,1)
        
        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        end_ids = torch.gather(tags_nofuzzy, 0, length_mask-1).view(batch_size,1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        final_conf_partition = conf_partition + end_energy
        final_confirmed = torch.gather(isconfirmed,0,length_mask-1).view(batch_size,1).float()
        final_partition = final_confirmed*final_conf_partition+(1-final_confirmed)*final_notconf_partition
        return final_partition.sum()
    def fuzzy_forward_ratio(self,feats,mask,tags):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            xxx
        """
        #feats = feats.softmax(2)
        pred_index = feats.argmax(2)
        pred_notO = (pred_index != conf.type2id['O'])
        print (pred_notO.sum())
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)
        mask = mask.byte()
        mask = mask.transpose(1, 0).contiguous()
        tags = tags.transpose(1,0).contiguous()
        isconfirmed = (tags != self.NOT_CONFIRM_IDX).long()
        notconfirmed = (tags == self.NOT_CONFIRM_IDX).long()      
        tags_2 = tags -self.NOT_CONFIRM_IDX*notconfirmed
        ins_num = batch_size * seq_len

        new_tags = Variable(torch.LongTensor(seq_len, batch_size))
        if self.use_cuda:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[0, :] = (tag_size - 2) * tag_size + tags_2[0, :]
            else:
                new_tags[idx,:] = tags_2[idx-1,:] * tag_size + tags_2[idx,:]
        new_tags = new_tags.view(seq_len, batch_size, 1)

        feats_expanded = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats_expanded
        temp = self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores + temp
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        new_scores = scores.view(seq_len, batch_size, -1)
        tg_energy = torch.gather(new_scores, 2, new_tags).view(seq_len, batch_size)

        seq_iter = enumerate(scores)
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        notconf_partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        #ratio on notO
        change_count = 0
        p_0 = feats[:,0,:].softmax(dim=1)
        max_score, max_id = torch.max(p_0,dim=1)
        notconf_partition_ratio = (notconf_partition.clone() * p_0.view(batch_size,tag_size,1))/max_score.view(batch_size,1,1)
        notO_0 = pred_notO[:,0].view(batch_size,1,1).expand(batch_size,tag_size,1)
        masked_notconf_partition_ratio = notconf_partition_ratio.masked_select(notO_0)
        if masked_notconf_partition_ratio.dim() != 0:
            change_count += masked_notconf_partition_ratio.dim()
            notconf_partition.masked_scatter_(notO_0, masked_notconf_partition_ratio)
        
        conf_partition = torch.gather(notconf_partition.view(batch_size,tag_size),1,tags_2[0,:].view(batch_size,1))
        
        for idx,cur_values in seq_iter:
            conf_notconf = conf_partition.view(batch_size,1,1).expand(batch_size,1,tag_size) + \
            torch.gather(cur_values,1,tags_2[idx-1,:].view(batch_size,1,1).expand(batch_size,1,tag_size))
            conf_notconf = conf_notconf.transpose(1,2).contiguous()

            cur_values = cur_values + notconf_partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            notconf_notconf = log_sum_exp(cur_values, tag_size).view(batch_size,tag_size,1)
            temp_notconf = isconfirmed[idx-1,:].view(batch_size,1,1).expand(batch_size,tag_size,1).float()*conf_notconf\
             + notconfirmed[idx-1,:].view(batch_size,1,1).expand(batch_size,tag_size,1).float()*notconf_notconf
            #ratio on notO
            p_idx = feats[:,idx,:].softmax(dim=1)
            max_score, max_id = torch.max(p_idx,dim=1)
            temp_notconf_ratio = (temp_notconf.clone()*p_idx.view(batch_size,tag_size,1))/max_score.view(batch_size,1,1)
            notO_idx = pred_notO[:,idx].view(batch_size,1,1).expand(batch_size,tag_size,1)
            masked_temp_notconf_ratio = temp_notconf_ratio.masked_select(notO_idx)
            if masked_temp_notconf_ratio.dim() != 0:
                temp_notconf.masked_scatter_(notO_idx, masked_temp_notconf_ratio)

            isconf_idx = tags_2[idx,:].view(batch_size,1,1)
            notconf_conf = torch.gather(notconf_notconf,1,isconf_idx).view(batch_size,1)
            conf_conf = conf_partition + tg_energy[idx,:].view(batch_size,1)
            temp_conf = isconfirmed[idx-1,:].view(batch_size,1).float()*conf_conf +\
                    notconfirmed[idx-1,:].view(batch_size,1).float()*notconf_conf

            mask_idx = mask[idx, :].view(batch_size, 1, 1).expand(batch_size, tag_size, 1)
            masked_temp_notconf = temp_notconf.masked_select(mask_idx)
            if masked_temp_notconf.dim() != 0:
                notconf_partition.masked_scatter_(mask_idx, masked_temp_notconf)

            mask_idx = mask[idx, :].view(batch_size, 1)
            masked_temp_conf = temp_conf.masked_select(mask_idx)
            if masked_temp_conf.dim() != 0:
                change_count += masked_temp_conf.dim()
                conf_partition.masked_scatter_(mask_idx, masked_temp_conf)
        
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        cur_values_notconf = cur_values + notconf_partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        final_notconf_partition = log_sum_exp(cur_values_notconf, tag_size)[:, self.END_TAG_IDX].view(batch_size,1)
        
        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=0).view(1,batch_size).long()
        end_ids = torch.gather(tags_2, 0, length_mask-1).view(batch_size,1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        final_conf_partition = conf_partition + end_energy
        final_confirmed = torch.gather(isconfirmed,0,length_mask-1).view(batch_size,1).float()
        final_partition = final_confirmed*final_conf_partition+(1-final_confirmed)*final_notconf_partition
        print (change_count)
        return final_partition.sum()
    def _viterbi_decode(self, feats, mask):
        """
        Args:
            feats: size=(batch_size, seq_len, self.target_size+2)
            mask: size=(batch_size, seq_len)

        Returns:
            decode_idx: (batch_size, seq_len), viterbi decode结果
            path_score: size=(batch_size, 1), 每个句子的得分
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)
        mask = mask.byte()
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()

        mask = mask.transpose(1, 0).contiguous()
        ins_num = seq_len * batch_size

        feats = feats.transpose(1, 0).contiguous().view(
            ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(
            1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)
        # record the position of the best score
        back_points = list()
        partition_history = list()

        # mask = 1 + (-1) * mask
        mask = (1 - mask.long()).byte()
        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()

        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        partition_history.append(partition)

        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(
                batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition.unsqueeze(-1))

            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            back_points.append(cur_bp)

        partition_history = torch.cat(partition_history).view(
            seq_len, batch_size, -1).transpose(1, 0).contiguous()

        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        last_partition = torch.gather(
            partition_history, 1, last_position).view(batch_size, tag_size, 1)

        last_values = last_partition.expand(batch_size, tag_size, tag_size) + \
            self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = Variable(torch.zeros(batch_size, tag_size)).long()
        if self.use_cuda:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        pointer = last_bp[:, self.END_TAG_IDX]
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        back_points = back_points.transpose(1, 0).contiguous()

        back_points.scatter_(1, last_position, insert_last)

        back_points = back_points.transpose(1, 0).contiguous()

        decode_idx = Variable(torch.LongTensor(seq_len, batch_size))
        if self.use_cuda:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.view(-1).data
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        return path_score, decode_idx

    def forward(self, feats, mask):
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    def _score_sentence(self, scores, mask, tags):
        """
        Args:
            scores: size=(seq_len, batch_size, tag_size, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)

        Returns:
            score:
        """
        mask = mask.byte()
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)

        new_tags = Variable(torch.LongTensor(batch_size, seq_len))
        if self.use_cuda:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx-1] * tag_size + tags[:, idx]

        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(
            1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask-1)

        end_energy = torch.gather(end_transition, 1, end_ids)
        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(
            seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))

        gold_score = tg_energy.sum() + end_energy.sum()

        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags, fuzzy = False):
        """
        Args:
            feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        """
        
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        if fuzzy:
            gold_score = self.fuzzy_forward(feats,mask,tags)
            #notfuzzy_score = self._score_sentence(scores, mask, tags2)
            #print ('fuzzy',forward_score,gold_score,notfuzzy_score)
        else:
            #fuzzy_score = self.fuzzy_forward(feats,mask,tags)
            gold_score = self._score_sentence(scores, mask, tags)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score
    def neg_log_likelihood_loss_ratio(self, feats, mask, tags, tags_):
        batch_size = feats.size(0)
        pred_index = feats.argmax(2)
        isconfirmed = (tags != self.NOT_CONFIRM_IDX).long()
        notconfirmed = (tags == self.NOT_CONFIRM_IDX).long()      
        tags_2 = tags*isconfirmed + pred_index*notconfirmed

        forward_score, scores = self._forward_alg(feats, mask)
        #print ('single_gold_score')
        gold_score = self._score_sentence(scores,mask,tags_)
        single_gold_score = self._score_sentence(scores, mask, tags_)
        #gold_score = self.fuzzy_forward_ratio(feats,mask,tags)
        
        print (forward_score,gold_score,single_gold_score)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score
    def neg_log_likelihood_loss_nil(self, feats, mask, tags_notconf, tags_nofuzzy):
        batch_size = feats.size(0)
        forward_score, scores = self._forward_alg(feats, mask)
        gold_score = self.fuzzy_forward_nil(feats, mask, tags_notconf, tags_nofuzzy)
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        return forward_score - gold_score