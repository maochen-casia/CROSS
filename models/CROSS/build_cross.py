import torch

from .cross import CROSS

def build_cross(config):
    device = torch.device(config.device)
    model = CROSS(dino_model_name=config.dino_model_name,
                  num_nodes_per_scale=config.num_nodes_per_scale,
                  hid_dim=config.hid_dim,
                  embed_dim=config.embed_dim,
                  depth_num_bins=config.depth_num_bins,
                  depth_min_m=config.depth_min_m,
                  depth_max_m=config.depth_max_m,
                  train_xy_search_steps=config.train_xy_search_steps,
                  train_yaw_search_steps=config.train_yaw_search_steps,
                  eval_xy_search_steps=config.eval_xy_search_steps,
                  eval_yaw_search_steps=config.eval_yaw_search_steps,   
                  supervise=config.supervise,
                  device=device)
    return model
