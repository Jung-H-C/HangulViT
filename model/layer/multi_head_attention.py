import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model.encoder import CNNEncoder


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % num_heads == 0

        self.head_dim = embedding_dim // num_heads

        self.q_fc = nn.Linear(embedding_dim, embedding_dim)
        self.k_fc = nn.Linear(embedding_dim, embedding_dim)
        self.v_fc = nn.Linear(embedding_dim, embedding_dim)

        self.fc_out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, query, key, value, mask=None, dropout=None):
        # query, key, value의 input size = [n_batch, head, seq_len, head_dim]
        # query의 size [n_batch, 1, 12, 512], key의 size [n_batch, 1, 121, 512]
        print("key의 size: {}".format(key.size()))
        print("query의 size: {}".format(query.size()))
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        print("scores의 size: {}".format(scores.size()))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # 0인 부분을 -inf로 채움
            print("mask done")
        attention_prob = F.softmax(scores, dim=-1)

        if dropout is not None:
            attention_prob = dropout(attention_prob)

        return torch.matmul(attention_prob, value) # [batch_size, head, seq_len, d_k]

    def forward(self, query, key, value, mask=None, dropout=None):
        n_batch = query.size(0)

        def transform(x, fc):
            out = x.view(n_batch, -1, self.num_heads, self.head_dim).transpose(1, 2) # 각 q, k, v를 [n_batch, seq_len, h, d_k]로 변환
            return out

        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = self.calculate_attention(query, key, value, mask=mask, dropout=dropout)
        out = out.transpose(1, 2) # [batch_size, seq_len, head, d_k]로 변환
        out = out.contiguous().view(n_batch, -1, self.embedding_dim) # [batch_size, seq_len, embedding_dim]
        out = self.fc_out(out) # [batch_size, seq_len, embedding_dim]
        return out

# random_image_feature = CNNEncoder().forward(torch.randn(32, 1, 360, 360)) # [32, 121, 512]
# random_query = torch.randn(32, 11, 512)
# attention_output = MultiHeadAttention(embedding_dim=512, num_heads=1).forward(query = random_query, key = random_image_feature, value = random_image_feature)
#
# print(attention_output.size())
