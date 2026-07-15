import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SemanticHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 8
    ):
        super().__init__()
        
        self.reliability_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=1, padding=0)
        )

        self.embedding_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=1, padding=0)
        )

    def forward(self, feature_maps):
        if not isinstance(feature_maps, dict):
            raise TypeError("feature_maps must be a dict of multi-scale feature maps")
        if len(feature_maps) == 0:
            raise ValueError("feature_maps cannot be empty")

        reliability_maps = {}
        embedding_maps = {}
        for key, featmap in feature_maps.items():
            reliability_maps[key] = self.reliability_mlp(featmap).squeeze(1)
            embedding_maps[key] = self.embedding_mlp(featmap)

        return reliability_maps, embedding_maps

