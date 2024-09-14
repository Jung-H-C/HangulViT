import torch.nn as nn
import copy
class Decoder(nn.Module):
    def __init__(self, decoder_block, n_layer, norm):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)]) # mask 기본적으로 pad_mask랑 seq_mask & 연산 하면 됨. (역삼각이랑 4차원)
        self.norm = norm

    def forward(self, input_label, encoder_out, mask):
        out = input_label
        for layer in self.layers: # layer가 하나의 decoder_block 단위임 (residual 3개)
            out = layer(out, encoder_out, mask)
        out = self.norm(out)
        return out