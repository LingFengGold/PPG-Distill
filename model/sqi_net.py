import torch
import torch.nn as nn
import torch.nn.functional as F


class SQINet(nn.Module):
    def __init__(self, CC=128):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, CC, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(CC),
            nn.ReLU(),
            nn.Dropout(0.0)
        )
        self.dilated_blocks = nn.Sequential(
            *[self._make_dilated_block(CC, 2**i) for i in range(5)]
        )
        # Final convolution to map 64 channels to 1 channel, preserving the input length.
        self.final_conv = nn.Conv1d(CC, 1, kernel_size=1)

    def _make_dilated_block(self, channels, dilation):
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # x: (batch, L) -> add channel dimension
        x = x.unsqueeze(1)         # becomes (batch, 1, L)
        x = self.initial_conv(x)     # becomes (batch, 64, L)
        x = self.dilated_blocks(x)   # becomes (batch, 64, L)
        x = self.final_conv(x)       # becomes (batch, 1, L)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        return x        # returns (batch, L)
