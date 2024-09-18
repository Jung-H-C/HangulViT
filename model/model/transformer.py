import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding.embedding import JamoEmbedding
from model.model.encoder import CNNEncoder
from data import return_one_batch


class HangulViT(nn.Module):
    def __init__(self, input_embed, encoder, decoder, embedding_dim, device, generator, vocab_size = 54):
        super(HangulViT, self).__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.input_embed = input_embed
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.generator = generator

    def encode(self, image):
        print('from forward: encode function start')
        return self.encoder(image)

    def decode(self, embedded_label, encoder_out, self_mask, cross_mask):
        print('from forward: decode function start')
        return self.decoder(embedded_label, encoder_out, self_mask, cross_mask)

    def make_pad_mask(self, query, pad_idx=53):
        # query: [batch_num, query_seq_len]
        # key: [batch_num, key_seq_len]

        print('make_pad_mask function start')
        batch_num, query_seq_len = query.size()

        key_mask = (query != pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        end_mask = key_mask

        for i in range(batch_num):
            for j in range(query_seq_len):
                if query[i][j] == pad_idx:
                    for k in range(j, query_seq_len):
                        for l in range(query_seq_len):
                            end_mask[i, 0, k, l] = False

        mask = end_mask & key_mask
        mask.requires_grad = False
        return mask

    def make_subsequent_mask(self, query):
        print('make_subsequent_mask function start')
        query_seq_len = query.size(1)

        tril = np.tril(np.ones((query_seq_len, query_seq_len)), k=0).astype(np.uint8)
        mask = torch.tensor(tril, dtype=torch.uint8, requires_grad=False, device=query.device)
        return mask

    def make_self_attn_mask(self, input):
        print('make_self_attn_mask function start')
        pad_mask = self.make_pad_mask(input)
        subsequent_mask = self.make_subsequent_mask(input)
        mask = pad_mask & subsequent_mask
        print('make_self_attn_mask_fin')
        return mask

    def make_cross_attn_mask(self, query, pad_idx = 53, num_feature = 121):
        # return: [batch_num, 1, 12, 121]
        print('make_cross_attn_mask function start')
        batch_num = query.size(0)
        query_seq_len = query.size(1)

        mask = np.ones((batch_num, 1, query_seq_len, num_feature), dtype=np.uint8)
        mask = torch.tensor(mask, dtype=torch.bool, requires_grad=False, device=query.device)

        for i in range(batch_num):
            for j in range(query_seq_len):
                if query[i][j] == pad_idx:
                    for k in range(j, query_seq_len):
                        for l in range(num_feature):
                            mask[i, 0, k, l] = False

        # print('ㅅ')
        mask.requires_grad = False
        print('make_cross_mask_fin')
        return mask

    def forward(self, image, input_label):
        print("batch input에 들어옴")
        image = image.to(self.device)
        input_label = input_label.to(self.device)

        self_attn_mask = self.make_self_attn_mask(input_label) # self-attention 용 마스크 생성
        cross_attn_mask = self.make_cross_attn_mask(input_label) # cross-attention 용 마스크 생성

        embedded_label = self.input_embed(input_label) # embedding
        encoder_out = self.encode(image) # encoder
        decoder_out = self.decode(embedded_label, encoder_out, self_attn_mask, cross_attn_mask) # decoder (ff, attention, layer-normalization, residual)
        out = self.generator(decoder_out) # generator (embedding_dim -> vocab_size)
        out = F.log_softmax(out, dim=-1) # 각 sequence softmax
        return out



# def generate_random_sequence(size):
#
#     batch_num = size[0]
#     max_seq = size[1]
#
#     random_sequence = torch.randint(50, 54, (batch_num, max_seq))
#
#     for i in range(batch_num):
#         for j in range(max_seq):
#             if random_sequence[i, j] == 53:
#                 for k in range(j+1, max_seq):
#                     random_sequence[i, k] = 53
#             else:
#                 continue
#
#     return random_sequence

# size = [32, 16]
# # np_size = np.array(size)
# # print(generate_random_sequence(size))
#
#


# def make_pad_mask(query, pad_idx=53):
#     # query: [batch_num, query_seq_len]
#     # key: [batch_num, key_seq_len]
#
#     print('make_pad_mask function start')
#     batch_num, query_seq_len = query.size()
#
#     key_mask = (query != pad_idx).unsqueeze(1).unsqueeze(2)
#     key_mask = key_mask.repeat(1, 1, query_seq_len, 1)
#
#     end_mask = key_mask
#
#     for i in range(batch_num):
#         for j in range(query_seq_len):
#             if query[i][j] == pad_idx:
#                 for k in range(j, query_seq_len):
#                     for l in range(query_seq_len):
#                         end_mask[i, 0, k, l] = False
#
#     mask = end_mask & key_mask
#     mask.requires_grad = False
#     return mask
#
# def make_subsequent_mask(query):
#     print('make_subsequent_mask function start')
#     query_seq_len = query.size(1)
#
#     tril = np.tril(np.ones((query_seq_len, query_seq_len)), k=0).astype(np.uint8)
#     mask = torch.tensor(tril, dtype=torch.uint8, requires_grad=False, device=query.device)
#     return mask
#
# def make_self_attn_mask(input):
#     print('make_self_attn_mask function start')
#     pad_mask = make_pad_mask(input)
#     subsequent_mask = make_subsequent_mask(input)
#     mask = pad_mask & subsequent_mask
#     print('make_self_attn_mask_fin')
#     return mask
#
# def make_cross_attn_mask(query, pad_idx = 53, num_feature = 121):
#     # return: [batch_num, 1, 12, 121]
#     print('make_cross_attn_mask function start')
#     batch_num = query.size(0)
#     query_seq_len = query.size(1)
#
#     mask = np.ones((batch_num, 1, query_seq_len, num_feature), dtype=np.uint8)
#     mask = torch.tensor(mask, dtype=torch.bool, requires_grad=False, device=query.device)
#
#     for i in range(batch_num):
#         for j in range(query_seq_len):
#             if query[i][j] == pad_idx:
#                 for k in range(j, query_seq_len):
#                     for l in range(num_feature):
#                         mask[i, 0, k, l] = False
#
#         # print('ㅅ')
#     mask.requires_grad = False
#     print('make_cross_mask_fin')
#     return mask

# mask 잘 생성되는지 test
# images, input_labels, output_labels = return_one_batch()
#
# self_mask = make_self_attn_mask(input_labels)
# cross_mask = make_cross_attn_mask(input_labels)
# print(input_labels[0, ...])
# print(self_mask[0, ...])
# print(cross_mask[0, ...])
# print(input_labels[1, ...])
# print(self_mask[1, ...])
# print(cross_mask[1, ...])

# key = generate_random_sequence(size)
# print(key)



