#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.functional.initialize import init_embedding
import numpy as np

class CharFeature(nn.Module):
    def __init__(self, embedding_shape, filter_sizes, filter_nums):
        super(CharFeature, self).__init__()
        self.feature_size = embedding_shape[0]
        self.feature_dim = embedding_shape[1]
        # char embedding layer
        self.char_embedding = nn.Embedding(self.feature_size, self.feature_dim)
        init_embedding(self.char_embedding.weight)
        
        # cnn
        self.char_encoders = nn.ModuleList()
        for i, filter_size in enumerate(filter_sizes):
            f = nn.Conv3d(
                in_channels=1, out_channels=filter_nums[i], kernel_size=(1, filter_size, self.feature_dim))
            self.char_encoders.append(f)

    def forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # [bs, 1, max_len, max_len_char, feature_dim]
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.feature_dim)

        # conv
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0])
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)

        # size=[bs, max_len, output_dim]
        char_conv_outputs = char_conv_outputs.transpose(-2, -1).contiguous()
        if len(char_conv_outputs.shape) == 2:
            char_conv_outputs = char_conv_outputs.unsqueeze(0)
        return char_conv_outputs


class WordFeature(nn.Module):

    def __init__(self, embedding_shape, pretrained_embedding, require_grad):
        super(WordFeature, self).__init__()
        num_of_words = embedding_shape[0]
        word_dim = embedding_shape[1]
        # feature embedding layer
        embed = nn.Embedding(num_of_words, word_dim)
        if pretrained_embedding is not None:  # 预训练向量
            # print('预训练:', feature_name)
            embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        else:  # 随机初始化
            # print('随机初始化:', feature_name)
            init_embedding(embed.weight)
        # 是否需要根据embedding的权重
        embed.weight.requires_grad = require_grad
        self.embed = embed

    def forward(self, word_input):
        return self.embed(word_input)
class PositionFeature(nn.Module):
    def __init__(self, position_size):
        super(PositionFeature, self).__init__()
        self.position_size = position_size
    def forward(self, inputs):
        batch_size,seq_len = inputs.size()[0],inputs.size()[1]
        position_j = 1. / np.power(10000., 2 * np.arange(self.position_size / 2, dtype=float) / self.position_size)
        position_j = torch.FloatTensor(np.expand_dims(position_j, 0))
        position_i = np.arange(seq_len, dtype=float)
        position_i = torch.FloatTensor(np.expand_dims(position_i, 1))
        position_ij = torch.matmul(position_i, position_j)
        position_ij = torch.cat([position_ij.cos(), position_ij.sin()], 1)
        position_embedding = position_ij.unsqueeze(0) \
                             + torch.zeros((batch_size, seq_len, self.position_size))
        return position_embedding.cuda()