import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding.embedding import JamoEmbedding
from model.model.encoder import CNNEncoder


class HangulViT(nn.Module):
    def __init__(self, input_embed, decoder, embedding_dim, vocab_size = 54):
        super(HangulViT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = CNNEncoder().to(self.device)
        self.decoder = decoder.to(self.device)
        self.input_embed = input_embed
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size


        self.generator = nn.Linear(embedding_dim, vocab_size)

    def encode(self, image):
        print('dsfsda')
        return self.encoder(image)

    def decode(self, embedded_label, encoder_out, mask):
        return self.decoder(embedded_label, encoder_out, mask)

    def make_pad_mask(self, query, key, pad_idx = 53):
        # query: [batch_num, query_seq_len]
        # key: [batch_num, key_seq_len]

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = (key != pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        # query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        # print('ㅁ')
        # query_mask = query_mask.repeat(1, 1, 1, key_seq_len)
        print(key_mask.size())

        mask = key_mask
        print('ㅅ')
        mask.requires_grad = False
        print('ㅇ')
        return mask

    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k = 0).astype(np.uint8)
        mask = torch.tensor(tril, dtype=torch.uint8, requires_grad=False, device=query.device)
        return mask

    def make_mask(self, input):
        print('A')
        pad_mask = self.make_pad_mask(input, input)
        print('B')
        subsequent_mask = self.make_subsequent_mask(input, input)
        print('C')
        mask = pad_mask & subsequent_mask
        # print(mask)
        # print(mask.size())
        print('D')
        return mask

    def forward(self, image, input_label):
        image = image.to(self.device)
        input_label = input_label.to(self.device)
        print(1)
        # print(input_label)
        print(input_label.size())
        mask = self.make_mask(input_label)
        print("mask의 size: {}".format(mask.size()))
        print(mask[..., :, :]) # 마지막 두개의 차원에 대해서만 출력

        print("input_label 의 size: {}".format(input_label.size()))
        embedded_label = self.input_embed(input_label)
        # embedded_label = self.input_embed(input_label)
        print("embedded_label 의 size: {}".format(embedded_label.size()))

        print(2)
        encoder_out = self.encode(image)
        print('F')
        print(encoder_out.size())
        print(3)
        decoder_out = self.decode(embedded_label, encoder_out, mask)
        print(decoder_out)
        print(4)
        out = self.generator(decoder_out)
        print(5)
        out = F.log_softmax(out, dim=-1)
        print(6)
        return out, decoder_out



def generate_random_sequence(size):

    batch_num = size[0]
    max_seq = size[1]

    random_sequence = torch.randint(50, 54, (batch_num, max_seq))

    for i in range(batch_num):
        for j in range(max_seq):
            if random_sequence[i, j] == 53:
                for k in range(j+1, max_seq):
                    random_sequence[i, k] = 53
            else:
                continue

    return random_sequence

size = [32, 16]
# np_size = np.array(size)
# print(generate_random_sequence(size))


def make_pad_mask(query, key, pad_idx=53):
    # query: [batch_num, query_seq_len]
    # key: [batch_num, key_seq_len]

    query_seq_len, key_seq_len = query.size(1), key.size(1)

    key_mask = (key != pad_idx).unsqueeze(1).unsqueeze(2)
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

    # query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(2)
    # print('ㅁ')
    # query_mask = query_mask.repeat(1, 1, 1, key_seq_len)
    # print(key_mask.size())

    mask = key_mask
    # print('ㅅ')
    mask.requires_grad = False
    # print('ㅇ')
    return mask
key = generate_random_sequence(size)
print(key)
mask = make_pad_mask(key, key)
print(mask.size())
print(mask[0,...])


def make_subsequent_mask(query, key):
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype(np.uint8)
    mask = torch.tensor(tril, dtype=torch.uint8, requires_grad=False, device=query.device)
    return mask

mask_b = make_subsequent_mask(key, key)
print(mask_b)

print(mask & mask_b)
# 이제 [32, 1, max_seq, max_seq] 크기의 mask를 [32, 1, max_seq, 121] 크기로 변환하고




