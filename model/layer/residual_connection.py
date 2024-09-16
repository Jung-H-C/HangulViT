import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, norm, dr_rate = 0):
        super(Residual, self).__init__()
        self.norm = norm
        self.dropout = nn.Dropout(dr_rate)

    def forward(self, x, sub_layer):
        out = x
        print("a")
        out = self.norm(out)
        print("b")
        out = sub_layer(out)
        print("c")
        out = self.dropout(out)
        print("d")
        out = out + x
        print("현재까지의 shape:{}".format(out.shape))
        return out