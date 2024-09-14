import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embedding.embedding import JamoEmbedding

class HangulViT(nn.Module):
    def __init__(self, input_embed, encoder, decoder, embedding_dim, vocab_size = 54):
        super(HangulViT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embed = input_embed
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.generator = nn.Linear(embedding_dim, vocab_size)

    def encode(self, image):
        return self.encoder(image)

    def decode(self, input_label, encoder_out, mask):
        return self.decoder(self.input_embed(input_label), encoder_out, mask)

    def make_pad_mask(self, query, key, pad_idx = 53):
        # query: [batch_num, query_seq_len]
        # key: [batch_num, key_seq_len]

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask

    def make_subsequent_mask(self, query, key):
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k = 0).astype(np.uint8)
        mask = torch.tensor(tril, dtype=torch.uint8, requires_grad=False, device=query.device)
        return mask

    def make_mask(self, input):
        pad_mask = self.make_pad_mask(input, input)
        subsequent_mask = self.make_subsequent_mask(input, input)
        mask = pad_mask & subsequent_mask
        return mask

    def forward(self, image, input_label):
        mask = self.make_mask(input_label)
        encoder_out = self.encoder(image)
        decoder_out = self.decoder(input_label, encoder_out, mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)
        return out, decoder_out

