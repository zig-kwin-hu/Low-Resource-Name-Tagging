#! -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
'''
inputs是一个形如(batch_size, seq_len, word_size)的张量；
函数返回一个形如(batch_size, seq_len, position_size)的位置张量。
'''
class Bilstm(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Bilstm, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True,batch_first=True)
        self.hidden_dim = hidden_dim

    '''
    inputs是一个二阶以上的张量，代表输入序列，比如形如(batch_size, seq_len, input_size)的张量；
    seq_len是一个形如(batch_size,)的张量，代表每个序列的实际长度，多出部分都被忽略；
    mode分为mul和add，mul是指把多出部分全部置零，一般用于全连接层之前；
    add是指把多出部分全部减去一个大的常数，一般用于softmax之前。
    '''
    def Mask(self, inputs, mask, mode='mul'):
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
                return inputs - (1 - mask) * 1e12
    def init_hidden(self,batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).cuda(),
                torch.randn(2, batch_size, self.hidden_dim // 2).cuda())
    def forward(self,inputs,length):
        batch_size = inputs.size(0)
        self.hidden = self.init_hidden(batch_size)
        packed_inputs = pack_padded_sequence(inputs,length,batch_first = True)
        lstm_out, self.hidden = self.lstm(packed_inputs, self.hidden)
        lstm_out,_ = pad_packed_sequence(lstm_out,batch_first=True)
        return lstm_out
