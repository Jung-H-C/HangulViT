import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, norm, dr_rate = 0):
        super(Residual, self).__init__()
        self.norm = norm
        self.dr_rate = dr_rate

    def forward(self, x, sub_layer):
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x
        return out