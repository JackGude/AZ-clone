# model.py (with Squeeze-and-Excitation blocks)

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    This block learns to re-weigh channel features, allowing the network to
    focus on the most informative channels for a given input.
    """
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # The number of channels to squeeze down to.
        squeezed_channels = channels // reduction_ratio
        
        # --- Squeeze Operation ---
        # Global Average Pooling takes the spatial dimensions (H, W) and reduces
        # them to a single value (1, 1), summarizing the information in each channel.
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # --- Excitation Operation ---
        # A small two-layer neural network learns the non-linear relationships
        # between channels and outputs an attention score for each.
        self.excitation = nn.Sequential(
            nn.Linear(channels, squeezed_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(squeezed_channels, channels, bias=False),
            nn.Sigmoid() # Sigmoid squashes the output to a [0, 1] range for scaling.
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        b, c, _, _ = x.shape
        
        # Squeeze: (b, c, h, w) -> (b, c, 1, 1) -> (b, c)
        y = self.squeeze(x).view(b, c)
        
        # Excite: (b, c) -> (b, c)
        y = self.excitation(y).view(b, c, 1, 1) # Reshape back for broadcasting
        
        # Scale: The original input `x` is multiplied by the learned attention
        # scores `y`, effectively re-weighing each channel.
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    The standard Residual Block, now enhanced with an SEBlock.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        # ADDED: The Squeeze-and-Excitation block.
        self.se    = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # ADDED: Apply the SE block to the output before the residual connection.
        out = self.se(out)
        
        out += residual
        return F.relu(out)


class AlphaZeroNet(nn.Module):
    """
    The main AlphaZero network architecture, now using SE-ResidualBlocks.
    """
    def __init__(self, in_channels=119, channels=256, n_res_blocks=19, n_moves=4672):
        super().__init__()
        # Initial convolution block
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in   = nn.BatchNorm2d(channels)

        # The tower of residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(n_res_blocks)]
        )

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 8 * 8, n_moves)

        # --- Value Head ---
        self.value_conv  = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2   = nn.Linear(256, 1)

    def forward(self, x):
        # Initial block
        out = F.relu(self.bn_in(self.conv_in(x)))
        
        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1) # Flatten
        logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1) # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return logits, v