import torch
import torch.nn as nn

class StructureHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_bins: int = 64,
        min_depth: float = 1.0,
        max_depth: float = 100.0,
    ):
        super().__init__()
        if min_depth <= 0:
            raise ValueError("min_depth must be > 0")
        if max_depth <= min_depth:
            raise ValueError("max_depth must be > min_depth")

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.num_bins = int(num_bins)

        # Uniform bin centers in log-depth space.
        log_bins = torch.linspace(
            torch.log(torch.tensor(self.min_depth)),
            torch.log(torch.tensor(self.max_depth)),
            steps=self.num_bins,
        )
        self.register_buffer("depth_bins", torch.exp(log_bins).float(), persistent=True)

        self.depth_cls_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels//2, self.num_bins, kernel_size=1, padding=0),
        )

        self.log_var_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels//2, 1, kernel_size=1, padding=0),
        )

    def forward(self, feature_maps):
        if not isinstance(feature_maps, dict):
            raise TypeError("feature_maps must be a dict of multi-scale feature maps")
        if len(feature_maps) == 0:
            raise ValueError("feature_maps cannot be empty")

        depth_maps = {}
        log_var_maps = {}
        for key, featmap in feature_maps.items():
            depth_logits = self.depth_cls_mlp(featmap)
            depth_probs = torch.softmax(depth_logits, dim=1)
            depth_maps[key] = torch.sum(
                depth_probs * self.depth_bins.view(1, -1, 1, 1),
                dim=1,
            )
            log_var_maps[key] = self.log_var_mlp(featmap).squeeze(1)

        return depth_maps, log_var_maps
