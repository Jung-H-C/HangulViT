import torch.nn as nn

class PositionWiseFeedForwardLayer(nn.Module):
    # decoder layer에 있는 PositionWiseFeedForward Layer
    def __init__(self, embedding_dim, hidden_dim, dropout = 0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        out = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
        