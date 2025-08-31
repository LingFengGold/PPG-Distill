import torch
import torch.nn as nn
import torch.nn.functional as F


class ACCNet(nn.Module):
    """PPG-to-Accelerometer Generator with internal pad-to-4096 then crop-to-3840."""
    def __init__(self, M=2):
        super().__init__()
        self.init_conv = nn.Conv1d(1, 32*M, 15, padding=7)

        # Encoder with dilated convolutions
        self.encoder = nn.Sequential(
            *[ResidualBlock(32*M, 32*M, dilation=2**i, norm='switch') for i in range(3)]
        )

        # Temporal processing
        self.tcn = nn.Sequential(
            nn.Conv1d(32*M, 32*M, 3, padding=1, groups=32*M),
            nn.ReLU(),
            nn.Conv1d(32*M, 64*M, 1),
            SwitchNorm1d(64*M)
        )

        # Separate decoders per channel - this multi-tail model works quite well
        self.channel_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(64*M, 32*M, 3, padding=1),
                SwitchNorm1d(32*M),
                nn.ReLU(),
                nn.Conv1d(32*M, 1, 1)
            ) for _ in range(3)
        ])

        # Attention
        self.attention = nn.Sequential(
            nn.Conv1d(64*M, 64*M, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch_size, 1, 3840)
        1) Reflect-pad to length=4096
        2) Forward pass
        3) Crop the output back to 3840
        """
        # 1) Pad from 3840 to 4096
        x = F.pad(x, (128, 128), mode='reflect')  # (B,1,4096)

        # 2) Forward pass
        x = self.init_conv(x)           # (B, 32*M, 4096)
        x = self.encoder(x)             # (B, 32*M, 4096)
        x = self.tcn(x)                 # (B, 64*M, 4096)

        attn = self.attention(x)        # (B, 64*M, 4096)
        x = x * attn + x                # Residual attention

        outputs = []
        for decoder in self.channel_decoders:
            outputs.append(decoder(x))
        x = torch.cat(outputs, dim=1)

        # 3) Crop center 3840 out of 4096: [128 : 128+3840]
        x = x[:, :, 128:128+3840]       # final shape (B, 3, 3840)
        return x


class SwitchNorm1d(nn.Module):
    """Custom Switchable Normalization layer"""
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

        self.mean_weight = nn.Parameter(torch.ones(3))
        self.var_weight = nn.Parameter(torch.ones(3))

        self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1))

    def forward(self, x):
        batch_size, channels, length = x.size()

        # Instance statistics
        instance_mean = x.mean(dim=2, keepdim=True)
        instance_var = x.var(dim=2, keepdim=True, unbiased=False)

        # Layer statistics
        layer_mean = x.mean(dim=[1,2], keepdim=True)
        layer_var = x.var(dim=[1,2], keepdim=True, unbiased=False)

        # Batch statistics
        batch_mean = x.mean(dim=[0,2], keepdim=True)
        batch_var = x.var(dim=[0,2], keepdim=True, unbiased=False)

        # Softmax weights
        mean_weight = F.softmax(self.mean_weight, dim=0)
        var_weight = F.softmax(self.var_weight, dim=0)

        # Combined statistics
        combined_mean = mean_weight[0]*instance_mean + mean_weight[1]*layer_mean + mean_weight[2]*batch_mean
        combined_var = var_weight[0]*instance_var + var_weight[1]*layer_var + var_weight[2]*batch_var

        # Normalize
        x = (x - combined_mean) / torch.sqrt(combined_var + self.eps)

        # Affine transformation
        return x * self.weight + self.bias


class ResidualBlock(nn.Module):
    """Dimension-preserving residual block"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, norm='switch'):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation,
                              padding_mode='reflect')
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation,
                              padding_mode='reflect')

        self.norm1 = SwitchNorm1d(out_channels) if norm == 'switch' else nn.InstanceNorm1d(out_channels)
        self.norm2 = SwitchNorm1d(out_channels) if norm == 'switch' else nn.InstanceNorm1d(out_channels)

        self.activation = nn.ReLU()
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + identity)
