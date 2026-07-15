import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeSampler(nn.Module):
    def __init__(self, num_nodes_per_scale, left_scales=[1, 2, 4, 8]):
        """
        Args:
            num_nodes_per_scale (int): Number of keypoints to sample per scale.
            left_scales (list): The downsampling factors corresponding to the left image maps.
        """
        super().__init__()

        self.num_nodes = num_nodes_per_scale
        self.left_scales = left_scales

    def simple_nms(self, weights, nms_radius, mask_val='-inf'):
        """Fast Non-maximum suppression to remove nearby points based on reliability weights."""
        assert(nms_radius >= 0)

        def max_pool(x):
            return F.max_pool2d(
                x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

        mask_val_float = {'zero': 0.0, '-inf': float('-inf')}[mask_val]
        mask_tensor = torch.ones_like(weights) * mask_val_float

        max_mask = weights == max_pool(weights)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_weights = torch.where(supp_mask, mask_tensor, weights)
            new_max_mask = supp_weights == max_pool(supp_weights)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, weights, mask_tensor)

    def mask_borders(self, weights, border, mask_val='-inf'):
        """Masks out weights that are too close to the border."""
        b, c, h, w = weights.shape
        mask = torch.zeros_like(weights)

        if h > 2 * border and w > 2 * border:
            mask[:, :, border:h - border, border:w - border] = 1
        else:
            mask[:, :, :, :] = 1

        mask_val_float = {'zero': 0.0, '-inf': float('-inf')}[mask_val]
        mask_tensor = torch.ones_like(weights) * mask_val_float

        weights = torch.where(mask > 0, weights, mask_tensor)
        return weights

    def forward(self, reliability_maps, embedding_maps, depth_maps):
        """
        Args:
            reliability_maps (dict): Reliability maps from SemanticHead.
            embedding_maps (dict): Embedding maps from SemanticHead.
            depth_maps (dict): Depth maps from StructureHead.

        Returns:
            all_norm_xy: [B, Total_Nodes, 2]
            all_reliability: [B, Total_Nodes]
            all_features: [B, Total_Nodes, C]
            all_depths: [B, Total_Nodes]
        """
        list_norm_xy = []
        list_reliability = []
        list_features = []
        list_depths = []

        for scale in self.left_scales:
            key = f'scale_{scale}'

            reliability_map = reliability_maps[key]
            embedding_map = embedding_maps[key]
            depth_map = depth_maps[key]

            B, C, H, W = embedding_map.shape

            border_size = max(1, H // 64)
            nms_radius = max(1, H // 64)

            reliability_map = reliability_map.unsqueeze(1)
            reliability_map = self.mask_borders(reliability_map, border=border_size, mask_val='-inf')
            reliability_map = self.simple_nms(reliability_map, nms_radius=nms_radius, mask_val='-inf')

            flat_reliability = reliability_map.flatten(2)
            topk_reliability, topk_indices = torch.topk(flat_reliability, self.num_nodes, dim=2)

            y = topk_indices // W
            x = topk_indices % W

            norm_x = (x.float() / (W - 1)) * 2 - 1
            norm_y = (y.float() / (H - 1)) * 2 - 1
            norm_xy = torch.stack([norm_x, norm_y], dim=-1)

            grid = norm_xy

            sampled_feats = F.grid_sample(embedding_map, grid, mode='bilinear', align_corners=False)
            sampled_feats = sampled_feats.squeeze(2).permute(0, 2, 1)

            sampled_depths = F.grid_sample(depth_map.unsqueeze(1), grid, mode='bilinear', align_corners=False)
            sampled_depths = sampled_depths.squeeze(2).squeeze(1)

            list_norm_xy.append(norm_xy.squeeze(1))
            list_reliability.append(topk_reliability.squeeze(1))
            list_features.append(sampled_feats)
            list_depths.append(sampled_depths)

        all_norm_xy = torch.cat(list_norm_xy, dim=1)
        all_reliability = torch.cat(list_reliability, dim=1)
        all_features = torch.cat(list_features, dim=1)
        all_depths = torch.cat(list_depths, dim=1)
        return all_norm_xy, all_reliability, all_features, all_depths
