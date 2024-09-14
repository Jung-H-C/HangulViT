import math
import torch.nn as nn

class JamoEmbedding(nn.Module):
    def __init__(self, vocab_size = 54, embedding_dim = 512, pad_idx = 53):
        super(JamoEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx)
        # 패딩 인덱스에 대해서는 모든 dimension을 고정된 값 0으로 설정
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)