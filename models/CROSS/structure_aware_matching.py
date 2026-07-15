import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StructureAwareMatching(nn.Module):

    def __init__(self, scales):
        super().__init__()

        self.scales = scales

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _sample_scale_similarity_and_reliability(self,
                scale: int,
                node_reliability_weights: torch.Tensor,
                norm_node_embeddings: torch.Tensor,
                sat_reliability_maps: dict[str, torch.Tensor],
                sat_embedding_maps: dict[str, torch.Tensor],
                grid: torch.Tensor,
                B: int,
                N: int,
                K: int):

        key = f'scale_{scale}'

        sat_reliability_map = sat_reliability_maps[key]  # (B, H, W)
        sampled_reliability_weights = F.grid_sample(
            sat_reliability_map.unsqueeze(1),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).view(B, N, K)  # (B, N, K)

        sat_embedding_map = sat_embedding_maps[key]  # (B, C, H, W)
        sampled_embeddings = F.grid_sample(
            sat_embedding_map,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        ).view(B, sat_embedding_map.shape[1], N, K).permute(0, 2, 3, 1)  # (B, N, K, C)

        norm_sat_embeddings = F.normalize(sampled_embeddings, p=2, dim=-1)  # (B, N, K, C)
        sim = torch.einsum('bnc, bnkc -> bnk', norm_node_embeddings, norm_sat_embeddings)  # (B, N, K)
        pair_reliability_weights = node_reliability_weights.unsqueeze(-1) + sampled_reliability_weights  # (B, N, K)

        return sim, pair_reliability_weights

    def score_hypotheses_at_scale(self,
                scale: int,
                node_reliability_weights: torch.Tensor,
                node_embeddings: torch.Tensor,
                sat_reliability_maps: dict[str, torch.Tensor],
                sat_embedding_maps: dict[str, torch.Tensor],
                candidate_node_norm_coords: torch.Tensor):
        
        B, N, _ = node_embeddings.shape
        _, K, _, _ = candidate_node_norm_coords.shape

        norm_node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)  # (B, N, D)
        grid = candidate_node_norm_coords.permute(0, 2, 1, 3).reshape(B, N * K, 1, 2)
        
        sim, pair_reliability_weights = self._sample_scale_similarity_and_reliability(
            scale=scale,
            node_reliability_weights=node_reliability_weights,
            norm_node_embeddings=norm_node_embeddings,
            sat_reliability_maps=sat_reliability_maps,
            sat_embedding_maps=sat_embedding_maps,
            grid=grid,
            B=B,
            N=N,
            K=K
        )

        pair_reliability = F.softmax(pair_reliability_weights, dim=1) # (B, N, K)
        hypothesis_logits = torch.sum(sim * pair_reliability, dim=1)  # (B, K)
        hypothesis_logits = hypothesis_logits * self.logit_scale.exp()
        return hypothesis_logits

    def score_hypotheses_combined(self,
                node_reliability_weights: torch.Tensor,
                node_embeddings: torch.Tensor,
                sat_reliability_maps: dict[str, torch.Tensor],
                sat_embedding_maps: dict[str, torch.Tensor],
                candidate_node_norm_coords: torch.Tensor):

        B, N, _ = node_embeddings.shape
        _, K, _, _ = candidate_node_norm_coords.shape

        norm_node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)  # (B, N, D)
        grid = candidate_node_norm_coords.permute(0, 2, 1, 3).reshape(B, N * K, 1, 2)

        total_sim = []
        total_reliability = []
        for scale in self.scales:
            sim, pair_reliability_weights = self._sample_scale_similarity_and_reliability(
                scale=scale,
                node_reliability_weights=node_reliability_weights,
                norm_node_embeddings=norm_node_embeddings,
                sat_reliability_maps=sat_reliability_maps,
                sat_embedding_maps=sat_embedding_maps,
                grid=grid,
                B=B,
                N=N,
                K=K
            )
            total_sim.append(sim)
            total_reliability.append(pair_reliability_weights)

        total_sim = torch.stack(total_sim, dim=-1)  # (B, N, K, S)
        total_reliability = torch.stack(total_reliability, dim=-1)  # (B, N, K, S)

        total_reliability = total_reliability.permute(0, 2, 1, 3).reshape(B, K, N * len(self.scales))
        total_reliability = F.softmax(total_reliability, dim=-1)
        total_sim = total_sim.permute(0, 2, 1, 3).reshape(B, K, N * len(self.scales))

        hypothesis_logits = torch.sum(total_sim * total_reliability, dim=-1)  # (B, K)
        hypothesis_logits = hypothesis_logits * self.logit_scale.exp()

        return hypothesis_logits