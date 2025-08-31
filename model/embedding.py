import torch
import torch.nn as nn
import torch.nn.functional as F


#class SinusoidalPE(nn.Module):
#    """
#    Positional Encoding
#    """
#    def __init__(self, seq_len, d_model):
#        super().__init__()
#        position_idx = torch.arange(0, seq_len).reshape(-1, 1)
#        freq = torch.pow(10000, -torch.arange(0, d_model, 2, dtype=torch.float)/d_model)
#        self.pe = torch.zeros(seq_len, d_model)
#        self.pe[:, 0::2] = torch.sin(position_idx * freq)
#        self.pe[:, 1::2] = torch.cos(position_idx * freq)
#
#    def forward(self):
#        return self.pe


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.linear = nn.Linear(patch_size, d_model)
        self.sep = nn.Parameter(torch.randn(d_model))

    def forward(self, x: torch.Tensor, sep_pos=None):
        x = self.linear(x)
        if sep_pos:
            self.process(x, sep_pos)
        else:
            # concat a sep in front
            sep = self.sep.reshape(1, 1, -1).repeat(x.shape[0], 1, 1)
            x = torch.cat([sep, x], dim=1)
        return x

    def process(self, x: torch.Tensor, sep_pos: list):
        # TODO: CAREFULLY CHECK THIS; USEFUL IN GENERATIVE DOWNSTREAM
        assert sep_pos[0] == 0, "sep position must be in order and contain 0 and last index"
        new_data = []
        sep = self.sep.reshape(1, 1, -1).repeat(x.shape[0], 1, 1)
        for i in range(len(sep_pos)-1):
            new_data.append(sep)
            new_data.append(x[sep_pos[i]:sep_pos[i+1]])
        new_data.append(sep)
        x = torch.cat(new_data, dim=1)
        return x

