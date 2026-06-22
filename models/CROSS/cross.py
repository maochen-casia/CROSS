import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
if code_dir not in sys.path:
    sys.path.append(code_dir)
sys.path.append(os.path.dirname(code_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional
import numpy as np

from DINOv3.dinov3_encoder import DINOv3
from DINOv2.dinov2_encoder import DINOv2
from dpt import DPT
from structure_head import StructureHead
from semantic_head import SemanticHead
from node_sampler import NodeSampler
from structure_aware_matching import StructureAwareMatching
from alignment_3d_grounded import align_3d_grounded_panorama, align_3d_grounded_pinhole
from transform import matrix_to_euler_zyx, replace_yaw_keep_roll_pitch

class CROSS(nn.Module):

    @staticmethod
    def _build_frozen_dino_encoder(dino_model_name, device):
        if dino_model_name.startswith('dinov3_'):
            return DINOv3(model_name=dino_model_name, device=device, freeze=True)
        if dino_model_name.startswith('dinov2_'):
            return DINOv2(model_name=dino_model_name, device=device, freeze=True)
        raise ValueError(
            f"Unsupported frozen backbone {dino_model_name}. "
            "Expected a dinov2_* or dinov3_* model name."
        )

    def __init__(self, dino_model_name, num_nodes_per_scale, hid_dim, embed_dim,
                 depth_num_bins, depth_min_m, depth_max_m,
                 train_xy_search_steps, train_yaw_search_steps,
                 eval_xy_search_steps, eval_yaw_search_steps, 
                 supervise, device):
        
        super().__init__()

        if supervise not in ['full', 'weak_2dof', 'weak_3dof']:
            raise ValueError(
                f"Unsupported supervise mode {supervise!r}. "
                "Expected one of: 'full', 'weak_2dof', 'weak_3dof'."
            )
        self.dino = self._build_frozen_dino_encoder(dino_model_name, device)
        self.dino_scale = self.dino.scale

        self.train_xy_search_steps = train_xy_search_steps
        self.train_yaw_search_steps = train_yaw_search_steps
        self.eval_xy_search_steps = eval_xy_search_steps
        self.eval_yaw_search_steps = eval_yaw_search_steps
        self.supervise = supervise

        self.left_scales = [1, 2, 4, 8]
        self.sat_scales = [1, 2, 4, 8]

        self.dpt = DPT(in_channels=self.dino.embed_dim, features=hid_dim, final_out_channels=hid_dim,
                       out_channels=[hid_dim//2, hid_dim, hid_dim*2, hid_dim*2], out_scales=[1,2,4,8])
        
        self.structure_head = StructureHead(
            in_channels=hid_dim,
            num_bins=depth_num_bins,
            min_depth=depth_min_m,
            max_depth=depth_max_m,
        )

        self.semantic_head = SemanticHead(
            in_channels=hid_dim,
            out_channels=embed_dim
        )

        self.node_sampler = NodeSampler(num_nodes_per_scale=num_nodes_per_scale,
                                        left_scales=self.left_scales)

        self.structure_aware_matching = StructureAwareMatching(scales=self.sat_scales)

        self.device = device
        self.to(device)
    
    def train_params(self):

        params = [{'params': [p for p in self.parameters() if p.requires_grad], 'lr_scale': 1}]
        
        return params
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Overrides the default state_dict() method to exclude the parameters
        of the frozen 'dino' submodule.
        """
        original_state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        filtered_state_dict = OrderedDict()

        for key, value in original_state_dict.items():
            if not key.startswith(prefix + 'dino.'):
                filtered_state_dict[key] = value

        return filtered_state_dict

    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the default load_state_dict() method to gracefully handle
        the missing 'dino' parameters by always using strict=False internally.
        """
        super().load_state_dict(state_dict, strict=False)
    
    def _forward_images(self, left_image, sat_image):
        """
            Semantic and structure encoding based on input images.
        """

        B, _, H1, W1 = left_image.shape
        B, _, H2, W2 = sat_image.shape

        # left image semantic encoding
        left_intermediate_features = self.dino.get_intermediate_layers(left_image)
        left_multi_scale_featmaps = self.dpt(left_intermediate_features,
                                                    patch_size=(H1//self.dino_scale, W1//self.dino_scale),
                                                    out_size=(H1,W1))
        left_reliability_maps, left_embedding_maps = self.semantic_head(left_multi_scale_featmaps)
        
        # left image structure encoding
        left_depth_maps, left_depth_variance_maps = self.structure_head(left_multi_scale_featmaps)

        # sat image semantic encoding
        sat_intermediate_features = self.dino.get_intermediate_layers(sat_image)
        sat_multi_scale_featmaps = self.dpt(sat_intermediate_features,
                                                    patch_size=(H2//self.dino_scale, W2//self.dino_scale),
                                                    out_size=(H2,W2))
        sat_reliability_maps, sat_embedding_maps = self.semantic_head(sat_multi_scale_featmaps)

        # node sampling
        node_norm_coords, node_reliability_weights, node_embeddings, node_depths = self.node_sampler(
            left_reliability_maps,
            left_embedding_maps,
            left_depth_maps
        )

        node_coords = (node_norm_coords + 1) / 2 * torch.tensor([W1 - 1, H1 - 1], device=self.device)

        pred = {
            'node_coords': node_coords,
            'node_reliability_weights': node_reliability_weights,
            'node_embeddings': node_embeddings,
            'node_depths': node_depths,
            'sat_reliability_maps': sat_reliability_maps,
            'sat_embedding_maps': sat_embedding_maps,
            'left_depth_maps': left_depth_maps,
            'left_depth_variance_maps': left_depth_variance_maps,
        }

        return pred

    def _expand_image_pred(self, image_pred):
        """
            Expand image predictions from batch size B to B*B for 3-dof weak supervision.
            Left-related predictions use left_i (repeat_interleave), and satellite-related
            predictions use sat_j (repeat).
        """

        B = image_pred['node_coords'].shape[0]
        pair_left_idx = torch.arange(B, device=self.device).repeat_interleave(B)
        pair_sat_idx = torch.arange(B, device=self.device).repeat(B)

        def _expand_value(value, idx):
            if torch.is_tensor(value):
                return value[idx]
            if isinstance(value, dict):
                return {k: _expand_value(v, idx) for k, v in value.items()}
            raise TypeError(f"Unsupported image_pred value type: {type(value)}")

        expanded_pred = {}
        for key, value in image_pred.items():
            if key.startswith('sat_'):
                expanded_pred[key] = _expand_value(value, pair_sat_idx)
            else:
                expanded_pred[key] = _expand_value(value, pair_left_idx)

        return expanded_pred

    def _expand_hypotheses_around_centers(self,
                                          t_center: torch.Tensor,
                                          R_center: torch.Tensor,
                                          node_coords: torch.Tensor,
                                          node_depths: torch.Tensor,
                                          K_left: Optional[torch.Tensor],
                                          left_image_hw: torch.Tensor,
                                          sat_affine_A: torch.Tensor,
                                          sat_affine_b: torch.Tensor,
                                          search_xy_radius: torch.Tensor,
                                          search_yaw_radius: torch.Tensor,
                                          xy_search_steps: int,
                                          yaw_search_steps: int):

        assert xy_search_steps >= 1 and (xy_search_steps % 2) == 1
        assert yaw_search_steps >= 1 and (yaw_search_steps % 2) == 1

        B, N, _ = node_coords.shape

        # Build a 2D xy offset grid and pair each xy offset with each yaw offset.
        xy_step_range = torch.linspace(-1.0, 1.0, steps=xy_search_steps, device=self.device)
        yaw_step_range = torch.linspace(-1.0, 1.0, steps=yaw_search_steps, device=self.device)
        grid_y, grid_x = torch.meshgrid(xy_step_range, xy_step_range, indexing='ij')
        offsets_norm_xy = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 1, 2)  # (1, Sxy, 1, 2)

        offsets_world_xy = offsets_norm_xy * search_xy_radius.view(B, 1, 1, 1)
        offsets_world_xy = offsets_world_xy.expand(B, -1, yaw_search_steps, -1)
        offsets_world = F.pad(offsets_world_xy, (0, 1), "constant", 0.0)  # (B, Sxy, Syaw, 3)

        yaw_offsets = yaw_step_range.view(1, 1, yaw_search_steps) * search_yaw_radius.view(B, 1, 1)
        center_yaw = matrix_to_euler_zyx(R_center)[2].unsqueeze(1)  # (B, 1)
        candidate_yaw = center_yaw.unsqueeze(2) + yaw_offsets  # (B, 1, Syaw)
        candidate_yaw = candidate_yaw.expand(-1, xy_search_steps*xy_search_steps, -1)  # (B, Sxy, Syaw)

        candidate_t = t_center.view(B, 1, 1, 3) + offsets_world
        candidate_t = candidate_t.reshape(B, -1, 3)  # (B, K, 3)
        candidate_yaw = candidate_yaw.reshape(B, -1)  # (B, K)

        K = candidate_t.shape[1]

        R_center_expand = R_center.unsqueeze(1).expand(B, K, 3, 3)
        candidate_R = replace_yaw_keep_roll_pitch(
            R_center_expand.reshape(B * K, 3, 3),
            candidate_yaw.reshape(B * K)
        ).reshape(B, K, 3, 3)

        node_coords_expand = node_coords.unsqueeze(1).expand(B, K, N, 2).reshape(B * K, N, 2)
        node_depths_expand = node_depths.unsqueeze(1).expand(B, K, N).reshape(B * K, N)
        R_left2world_expand = candidate_R.reshape(B * K, 3, 3)
        candidate_t_expand = candidate_t.reshape(B * K, 3)
        sat_affine_A_expand = sat_affine_A.unsqueeze(1).expand(B, K, 2, 2).reshape(B * K, 2, 2)
        sat_affine_b_expand = sat_affine_b.unsqueeze(1).expand(B, K, 2).reshape(B * K, 2)

        # panorama case
        if K_left is None:
            left_image_hw_expand = left_image_hw.unsqueeze(1).expand(B, K, 2).reshape(B * K, 2)
            candidate_node_coords = align_3d_grounded_panorama(
                node_coords=node_coords_expand,
                node_depths=node_depths_expand,
                left_image_hw=left_image_hw_expand,
                R_left2world=R_left2world_expand,
                t_left2world=candidate_t_expand,
                sat_affine_A=sat_affine_A_expand,
                sat_affine_b=sat_affine_b_expand,
            ).reshape(B, K, N, 2)
        else:
            assert K_left is not None, "K_left is required for pinhole projection."
            K_left_expand = K_left.unsqueeze(1).expand(B, K, 3, 3).reshape(B * K, 3, 3)
            candidate_node_coords = align_3d_grounded_pinhole(
                node_coords=node_coords_expand,
                node_depths=node_depths_expand,
                K_left=K_left_expand,
                R_left2world=R_left2world_expand,
                t_left2world=candidate_t_expand,
                sat_affine_A=sat_affine_A_expand,
                sat_affine_b=sat_affine_b_expand,
            ).reshape(B, K, N, 2)

        pred = {
            'candidate_t': candidate_t,
            'candidate_R': candidate_R,
            'candidate_yaw': candidate_yaw,
            'candidate_node_coords': candidate_node_coords,
            'xy_search_steps': xy_search_steps,
            'yaw_search_steps': yaw_search_steps
        }
        return pred
    
    def _prepare_hypothesis_expansion_args(self, data, image_pred):

        """
            Prepare arguments for hypothesis expansion for different modes.
            Full supervision: expand hypotheses around the ground-truth 3-DoF pose.
            2-DoF weak supervision: use ground-truth rotation/yaw, but search over
                translation hypotheses within each positive image pair.
            3-DoF weak supervision: expand hypotheses around the initial pose and
                create all in-batch left/satellite pairs for contrastive learning.
            Inference: expand hypotheses around the initial pose.
        """

        K_left = data.get('K_left', None)
        K_left = K_left.to(self.device) if K_left is not None else None

        B = image_pred['node_coords'].shape[0]
        H1, W1 = data['left_image'].shape[2:]
        left_image_hw = torch.tensor([H1, W1], dtype=torch.float32, device=self.device).unsqueeze(0).expand(B, -1)

        sat_affine_A = data['sat_affine_A'].to(self.device)
        sat_affine_b = data['sat_affine_b'].to(self.device)

        max_init_offset = data['max_init_offset'].to(self.device)
        max_init_yaw = data['max_init_yaw'].to(self.device)

        node_coords = image_pred['node_coords']
        node_depths = image_pred['node_depths']

        if self.training and self.supervise == 'full':
            t_center = data['t_left2world_gt'].to(self.device)
            R_center = data['R_left2world_gt'].to(self.device)
        elif self.training and self.supervise == 'weak_2dof':
            t_center = data['t_left2world_init'].to(self.device)
            R_center = data['R_left2world_gt'].to(self.device)
        else:
            t_center = data['t_left2world_init'].to(self.device)
            R_center = data['R_left2world_init'].to(self.device)
        
        xy_search_steps = self.train_xy_search_steps if self.training else self.eval_xy_search_steps
        yaw_search_steps = self.train_yaw_search_steps if self.training else self.eval_yaw_search_steps

        # If initialization is exact, skip searching along that axis.
        if torch.max(torch.abs(max_init_offset)).item() <= 1e-8:
            xy_search_steps = 1
        if torch.max(torch.abs(max_init_yaw)).item() <= 1e-8:
            yaw_search_steps = 1

        if self.training and self.supervise == 'weak_3dof':
            pair_left_idx = torch.arange(B, device=self.device).repeat_interleave(B)
            pair_sat_idx = torch.arange(B, device=self.device).repeat(B)

            t_center = t_center[pair_sat_idx]
            R_center = R_center[pair_sat_idx]

            node_coords = node_coords[pair_left_idx]
            node_depths = node_depths[pair_left_idx]
            left_image_hw = left_image_hw[pair_left_idx]
            if K_left is not None:
                K_left = K_left[pair_left_idx]

            sat_affine_A = sat_affine_A[pair_sat_idx]
            sat_affine_b = sat_affine_b[pair_sat_idx]
            max_init_offset = max_init_offset[pair_sat_idx]
            max_init_yaw = max_init_yaw[pair_sat_idx]

        args = {
            't_center': t_center,
            'R_center': R_center,
            'node_coords': node_coords,
            'node_depths': node_depths,
            'K_left': K_left,
            'left_image_hw': left_image_hw,
            'sat_affine_A': sat_affine_A,
            'sat_affine_b': sat_affine_b,
            'search_xy_radius': max_init_offset,
            'search_yaw_radius': max_init_yaw,
            'xy_search_steps': xy_search_steps,
            'yaw_search_steps': yaw_search_steps
        }
        return args

    def _score_hypotheses(self, image_pred, hypothesis_pred, scale=None):

        node_reliability_weights = image_pred['node_reliability_weights']
        node_embeddings = image_pred['node_embeddings']
        sat_reliability_maps = image_pred['sat_reliability_maps']
        sat_embedding_maps = image_pred['sat_embedding_maps']
        candidate_node_coords = hypothesis_pred['candidate_node_coords']
        H2, W2 = sat_embedding_maps['scale_1'].shape[2:]
        candidate_node_norm_coords = (candidate_node_coords / torch.tensor([W2 - 1, H2 - 1], device=self.device)) * 2 - 1

        if scale is not None:
            hypothesis_logits = self.structure_aware_matching.score_hypotheses_at_scale(
                scale=scale,
                node_reliability_weights=node_reliability_weights,
                node_embeddings=node_embeddings,
                sat_reliability_maps=sat_reliability_maps,
                sat_embedding_maps=sat_embedding_maps,
                candidate_node_norm_coords=candidate_node_norm_coords,
            )
        else:
            hypothesis_logits = self.structure_aware_matching.score_hypotheses_combined(
                node_reliability_weights=node_reliability_weights,
                node_embeddings=node_embeddings,
                sat_reliability_maps=sat_reliability_maps,
                sat_embedding_maps=sat_embedding_maps,
                candidate_node_norm_coords=candidate_node_norm_coords,
            )

        best_idx = torch.argmax(hypothesis_logits, dim=1) # (B,)
        candidate_t = hypothesis_pred['candidate_t']
        candidate_R = hypothesis_pred['candidate_R']
        best_t = candidate_t[torch.arange(candidate_t.shape[0]), best_idx] # (B, 3)
        best_R = candidate_R[torch.arange(candidate_R.shape[0]), best_idx] # (B, 3, 3)

        pred = {
            'hypothesis_logits': hypothesis_logits,
            't_left2world': best_t,
            'R_left2world': best_R,
            'xy_search_steps': hypothesis_pred['xy_search_steps'],
            'yaw_search_steps': hypothesis_pred['yaw_search_steps']
        }

        return pred

    def _prepare_weak_3dof_training_output(self, image_pred, multi_scale_score_preds):

        # max-pool over hypotheses to get pair logits for contrastive learning
        multi_scale_pair_logits = {}
        for scale, score_pred in multi_scale_score_preds.items():
            hypothesis_logits = score_pred['hypothesis_logits']
            pair_logits = torch.max(hypothesis_logits, dim=1).values
            B = int(hypothesis_logits.shape[0] ** 0.5)
            pair_logits = pair_logits.view(B, B)
            multi_scale_pair_logits[scale] = pair_logits
        
        positive_indices = torch.arange(B, device=self.device) * (B + 1)
        pred = {
            'multi_scale_pair_logits': multi_scale_pair_logits,
            't_left2world': multi_scale_score_preds['combined']['t_left2world'][positive_indices],
            'R_left2world': multi_scale_score_preds['combined']['R_left2world'][positive_indices],
            'left_depth_maps': {k: v[positive_indices] for k, v in image_pred['left_depth_maps'].items()},
            'left_depth_variance_maps': {k: v[positive_indices] for k, v in image_pred['left_depth_variance_maps'].items()}
        }
        return pred

    def _prepare_weak_2dof_training_output(self, image_pred, multi_scale_score_preds):

        multi_scale_hypothesis_logits = {}
        for scale, score_pred in multi_scale_score_preds.items():
            hypothesis_logits = score_pred['hypothesis_logits']
            xy_search_steps = score_pred['xy_search_steps']
            yaw_search_steps = score_pred['yaw_search_steps']
            B = hypothesis_logits.shape[0]

            hypothesis_logits = hypothesis_logits.view(B, xy_search_steps * xy_search_steps, yaw_search_steps)

            yaw_logits = torch.max(hypothesis_logits, dim=1).values

            multi_scale_hypothesis_logits[scale] = yaw_logits

        pred = {
            'multi_scale_hypothesis_logits': multi_scale_hypothesis_logits,
            't_left2world': multi_scale_score_preds['combined']['t_left2world'],
            'R_left2world': multi_scale_score_preds['combined']['R_left2world'],
            'left_depth_maps': image_pred['left_depth_maps'],
            'left_depth_variance_maps': image_pred['left_depth_variance_maps']
        }
        return pred

    def _prepare_full_training_output(self, image_pred, multi_scale_score_preds):

        multi_scale_hypothesis_logits = {}
        for scale, score_pred in multi_scale_score_preds.items():
            multi_scale_hypothesis_logits[scale] = score_pred['hypothesis_logits']
        pred = {
            'multi_scale_hypothesis_logits': multi_scale_hypothesis_logits,
            't_left2world': multi_scale_score_preds['combined']['t_left2world'],
            'R_left2world': multi_scale_score_preds['combined']['R_left2world'],
            'left_depth_maps': image_pred['left_depth_maps'],
            'left_depth_variance_maps': image_pred['left_depth_variance_maps']
        }
        return pred

    def _prepare_inference_output(self, score_pred, image_pred):

        pred = {
            't_left2world': score_pred['t_left2world'],
            'R_left2world': score_pred['R_left2world'],
            'left_depth_map': image_pred['left_depth_maps']['scale_1'] 
        }
        return pred

    def forward(self, data):

        left_image = data['left_image'].to(self.device)
        sat_image = data['sat_image'].to(self.device)
        B, _, H1, W1 = left_image.shape

        image_pred = self._forward_images(left_image, sat_image)

        # generate hypotheses
        hypothesis_expand_args = self._prepare_hypothesis_expansion_args(data, image_pred)
        if self.training:
            multi_scale_hypothesis_preds = {}
            for scale in self.sat_scales:
                cur_hypothesis_expand_args = hypothesis_expand_args.copy()
                if self.supervise == 'full':
                    cur_hypothesis_expand_args['search_xy_radius'] = hypothesis_expand_args['search_xy_radius'] * scale ** 0.5
                    cur_hypothesis_expand_args['search_yaw_radius'] = hypothesis_expand_args['search_yaw_radius'] * scale ** 0.5
                    cur_hypothesis_expand_args['search_yaw_radius'] = torch.clamp(cur_hypothesis_expand_args['search_yaw_radius'], max=np.pi)
                multi_scale_hypothesis_preds[f'scale_{scale}'] = self._expand_hypotheses_around_centers(**cur_hypothesis_expand_args)
            multi_scale_hypothesis_preds['combined'] = multi_scale_hypothesis_preds['scale_1']
        else:
            hypothesis_pred = self._expand_hypotheses_around_centers(**hypothesis_expand_args)

        if self.training and self.supervise == 'weak_3dof':
            image_pred = self._expand_image_pred(image_pred)

        # score hypotheses
        if self.training:
            multi_scale_score_preds = {}
            for scale in self.sat_scales:
                multi_scale_score_preds[f'scale_{scale}'] = self._score_hypotheses(image_pred, multi_scale_hypothesis_preds[f'scale_{scale}'], scale=scale)
            multi_scale_score_preds['combined'] = self._score_hypotheses(image_pred, multi_scale_hypothesis_preds['combined'], scale=None)
        else:
            score_pred = self._score_hypotheses(image_pred, hypothesis_pred, scale=None)
        
        # prepare output
        if self.training and self.supervise == 'weak_3dof':
            pred = self._prepare_weak_3dof_training_output(image_pred, multi_scale_score_preds)
        elif self.training and self.supervise == 'weak_2dof':
            pred = self._prepare_weak_2dof_training_output(image_pred, multi_scale_score_preds)
        elif self.training and self.supervise == 'full':
            pred = self._prepare_full_training_output(image_pred, multi_scale_score_preds)
        else:
            pred = self._prepare_inference_output(score_pred, image_pred)

        return pred

    # loss functions
    def _scale_invariant_depth_loss(self, 
                                    pred_depth_map: torch.Tensor, 
                                    pred_depth_variance_map: torch.Tensor,
                                    mono_depth_map: torch.Tensor) -> torch.Tensor:
        
        valid_mask = (torch.isfinite(pred_depth_map) & torch.isfinite(mono_depth_map) & 
                    (pred_depth_map > 1e-6) & (mono_depth_map > 1e-6)).float()

        log_diff = (torch.log(pred_depth_map.clamp_min(1e-6)) - torch.log(mono_depth_map.clamp_min(1e-6)))
        
        s = torch.clamp(pred_depth_variance_map, min=-10.0, max=10.0)
        confidence = torch.exp(-s) * valid_mask
        
        weighted_log_diff_sum = (log_diff * confidence).sum(dim=[-2, -1], keepdim=True)
        confidence_sum = confidence.sum(dim=[-2, -1], keepdim=True).clamp_min(1e-8)
        weighted_mean_d = weighted_log_diff_sum / confidence_sum

        diff_sq = (log_diff - weighted_mean_d) ** 2
        si_loss_map = (confidence * diff_sq + (s * valid_mask))

        # 6. Final Reduction
        valid_count = valid_mask.sum(dim=[-2, -1]).clamp_min(1e-8)
        per_image_loss = si_loss_map.sum(dim=[-2, -1]) / valid_count
        
        has_valid = (valid_mask.sum(dim=[-2, -1]) > 0).float()

        return per_image_loss.sum() / has_valid.sum().clamp_min(1.0)
        

    def _multi_scale_scale_invariant_depth_loss(self, 
                                                left_depth_maps, 
                                                left_depth_variance_maps, 
                                                mono_depth_map):


        multi_scale_depth_losses = {}

        for key in sorted(left_depth_maps.keys()):
            pred_depth_map = left_depth_maps[key]

            mono_depth_map_scaled = F.interpolate(
                mono_depth_map.unsqueeze(1),
                size=pred_depth_map.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(1)

            pred_depth_variance_map = left_depth_variance_maps[key]

            multi_scale_depth_losses[key] = self._scale_invariant_depth_loss(
                pred_depth_map=pred_depth_map,
                pred_depth_variance_map=pred_depth_variance_map,
                mono_depth_map=mono_depth_map_scaled
            )

        return multi_scale_depth_losses
    

    def loss(self, pred, label):
        
        multi_scale_nll_losses = {}
        if self.supervise == 'weak_3dof':
            multi_scale_pair_logits = pred['multi_scale_pair_logits']
            B = multi_scale_pair_logits['scale_1'].shape[0]
            pair_labels = torch.arange(B, device=self.device)
            for scale, pair_logits in multi_scale_pair_logits.items():
                multi_scale_nll_losses[scale] = F.cross_entropy(pair_logits, pair_labels)
        else:
            multi_scale_hypothesis_logits = pred['multi_scale_hypothesis_logits']
            for scale, hypothesis_logits in multi_scale_hypothesis_logits.items():
                B, K = hypothesis_logits.shape
                # positive hypothesis always lies at the center for both fully and 2-DoF weakly supervised training
                hypothesis_labels = torch.ones(B, dtype=torch.long, device=self.device) * (K // 2)
                multi_scale_nll_losses[scale] = F.cross_entropy(hypothesis_logits, hypothesis_labels)

        nll_loss = sum(multi_scale_nll_losses.values())

        multi_scale_depth_losses = self._multi_scale_scale_invariant_depth_loss(
            left_depth_maps=pred['left_depth_maps'],
            left_depth_variance_maps=pred['left_depth_variance_maps'],
            mono_depth_map=label['left_mono_depth_map'].to(self.device)
        )

        depth_loss = sum(multi_scale_depth_losses.values())

        total_loss = nll_loss + 0.1 * depth_loss

        t_error = torch.norm(pred['t_left2world'] - label['t_left2world'].to(self.device), dim=-1).mean().item()

        loss_dict = {
            'loss': total_loss.item(),
            'nll': [round(multi_scale_nll_losses[key].item(), 2) for key in ['scale_1', 'scale_2', 'scale_4', 'scale_8', 'combined']],
            'dl': depth_loss.item(),
            't': t_error
        }

        return total_loss, loss_dict
