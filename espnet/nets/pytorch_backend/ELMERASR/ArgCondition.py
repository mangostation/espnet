import torch

class Argmax_Condition(torch.nn.Module):
    def __init__(self, attention_dim, odim, dropout_rate=0.1):
        super().__init__()
        self.li = torch.nn.Linear(attention_dim, odim)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, hs_pad):
        return torch.argmax(self.li(self.dropout(hs_pad)), dim=2)
