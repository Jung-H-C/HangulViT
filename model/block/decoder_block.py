import copy
import torch.nn as nn

from model.layer.residual_connection import Residual

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate = 0):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [Residual(copy.deepcopy(norm), dr_rate) for _ in range(3)] # out, self_attn, cross_attn, position_ff

    def forward(self, x, encoder_out, self_mask, cross_mask):
        out = x
        print("ㄱ")
        out = self.residuals[0](out, lambda out: self.self_attention(query = out, key = out, value = out, mask = self_mask))
        print("ㄴ")
        out = self.residuals[1](out, lambda out: self.cross_attention(query = out, key = encoder_out, value = encoder_out, mask = cross_mask)) # 두 mask 다 tgt_mask로 사용하면 됨! (두개 and)
        print("ㄷ")
        out = self.residuals[2](out, lambda out: self.position_ff(out))
        print("ㄹ")
        return out
