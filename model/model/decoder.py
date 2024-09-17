import torch.nn as nn
import copy
from model.block.decoder_block import DecoderBlock
class Decoder(nn.Module):
    def __init__(self, n_layer, decoder_block, norm = None):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.norm = norm
        self.decoder_block = decoder_block
        self.layers = nn.ModuleList([copy.deepcopy(self.decoder_block) for _ in range(self.n_layer)]) # mask 기본적으로 pad_mask랑 seq_mask & 연산 하면 됨. (역삼각이랑 4차원)


    def forward(self, input_label, encoder_out, self_mask, cross_mask):
        out = input_label
        print('ii')
        for layer in self.layers: # layer가 하나의 decoder_block 단위임 (residual 3개)
            print('i')
            out = layer(out, encoder_out, self_mask, cross_mask)
        # out = self.norm(out)
        return out