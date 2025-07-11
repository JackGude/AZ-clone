# model.py

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
            nn.GELU(),
            nn.Linear(squeezed_channels, channels, bias=False),
            nn.Sigmoid(),  # Sigmoid squashes the output to a [0, 1] range for scaling.
        )

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        b, c, _, _ = x.shape

        # Squeeze: (b, c, h, w) -> (b, c, 1, 1) -> (b, c)
        y = self.squeeze(x).view(b, c)

        # Excite: (b, c) -> (b, c)
        y = self.excitation(y).view(b, c, 1, 1)  # Reshape back for broadcasting

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
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.gelu(out)


class AlphaZeroNet(nn.Module):
    """
    The main AlphaZero network architecture, now with multiple policy heads
    and an auxiliary legality head.
    """

    def __init__(
        self,
        in_channels=119,
        channels=384,
        n_res_blocks=40,
        n_moves=4672,
        n_policy_heads=3,
    ):
        super().__init__()
        self.n_policy_heads = n_policy_heads

        # Initial convolution block
        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(channels)

        # The tower of residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(n_res_blocks)]
        )

        # --- Multi-Headed Policy ---
        # Create n separate policy heads
        self.policy_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, 2, kernel_size=1),
                    nn.BatchNorm2d(2),
                    nn.GELU(),
                    nn.Flatten(),
                    nn.Linear(2 * 8 * 8, n_moves),
                )
                for _ in range(self.n_policy_heads)
            ]
        )
        
        # --- Gating Network for MoE Policy ---
        self.policy_gating_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),  # Reduce channels to 2
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 32),  # Map to hidden size
            nn.GELU(),
            nn.Linear(32, n_policy_heads)  # Output logits for each policy head
        )

        # --- Attentive Value Head ---
        # This new convolutional layer learns the attention "heat map".
        self.value_attention_conv = nn.Conv2d(
            in_channels=channels, out_channels=1, kernel_size=1
        )

        # The value head is now simpler. It takes the attended features
        # and maps them down to a single score.
        self.value_head = nn.Sequential(
            nn.Linear(channels, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        # --- Auxiliary Legality Head ---
        self.legality_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, 256),
            nn.GELU(),
            nn.Linear(256, n_moves),
        )

    def forward(self, x):
        # Initial block and residual tower
        out = F.gelu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # --- Policy Heads with MoE Gating ---
        # Get a list of logit tensors from each policy head
        policy_logits_list = [head(out) for head in self.policy_heads]
        
        # Get gating weights from the gating network
        gating_logits = self.policy_gating_head(out)  # shape: (batch_size, n_policy_heads)
        gating_weights = F.softmax(gating_logits, dim=-1)  # shape: (batch_size, n_policy_heads)
        
        # Stack policy logits and apply gating weights
        stacked_policy_logits = torch.stack(policy_logits_list, dim=1)  # shape: (batch_size, n_policy_heads, n_moves)
        gating_weights = gating_weights.unsqueeze(-1)  # shape: (batch_size, n_policy_heads, 1)
        aggregated_policy_logits = (stacked_policy_logits * gating_weights).sum(dim=1)  # shape: (batch_size, n_moves)

        # --- Attentive Value Head ---
        # 1. Create the attention "heat map"
        # Output shape: (batch, 1, 8, 8)
        attention_logits = self.value_attention_conv(out)

        # Flatten for softmax, then reshape back to an 8x8 map
        # This turns the logits into a probability distribution over the 64 squares.
        attention_map = F.softmax(
            attention_logits.view(out.size(0), -1), dim=1
        ).view_as(attention_logits)

        # 2. Apply the attention
        # Multiply the ResNet features by the attention map. This re-weighs the features
        # on each square, amplifying the important ones.
        attended_features = out * attention_map

        # 3. Sum the features across the board to get a single feature vector
        # This changes the shape from (batch, channels, 8, 8) to (batch, channels)
        v = attended_features.sum(dim=[-1, -2])

        # 4. Pass the final feature vector through the value head's linear layers
        value = self.value_head(v)

        # --- Legality Head ---
        legality_logits = self.legality_head(out)

        # Calculate load balancing loss to encourage using all experts
        mean_gating_weights = gating_weights.mean(dim=0)
        importance_per_expert = mean_gating_weights
        load_balancing_loss = torch.sum(importance_per_expert * mean_gating_weights)

        # Return all necessary outputs
        return aggregated_policy_logits, value, legality_logits, load_balancing_loss
