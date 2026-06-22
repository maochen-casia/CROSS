import torch
import torch.nn.functional as F


def get_inv_K(K: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(K)

@torch.jit.script
def backproject_nodes_to_world_pinhole(
    node_coords: torch.Tensor,
    node_depths: torch.Tensor,
    K_left: torch.Tensor,
    R_left2world: torch.Tensor,
    t_left2world: torch.Tensor,
) -> torch.Tensor:
    """Back-project image nodes and transform them into world coordinates."""
    uv_hom = F.pad(node_coords, (0, 1), "constant", 1.0)
    inv_K_left = torch.linalg.inv(K_left)
    rays = uv_hom @ inv_K_left.transpose(-2, -1)
    p_cam = rays * node_depths.unsqueeze(-1)
    return p_cam @ R_left2world.transpose(-2, -1) + t_left2world.unsqueeze(1)


@torch.jit.script
def backproject_nodes_to_world_panorama(
    node_coords: torch.Tensor,
    node_depths: torch.Tensor,
    left_image_hw: torch.Tensor,
    R_left2world: torch.Tensor,
    t_left2world: torch.Tensor,
) -> torch.Tensor:
    """Back-project equirectangular panorama nodes into world coordinates.

    left_image_hw stores per-sample (H, W).
    node_depths are treated as radial distances along panorama rays.

        Coordinate convention (aligned with VIGOR preprocessing/visualization):
        - world x: east
        - world y: north
        - world z: up
        - u center corresponds to yaw=0 (north), increasing u rotates clockwise
            (toward east), and increasing v looks downward.
    """
    h = left_image_hw[:, 0].view(-1, 1).clamp_min(1.0)
    w = left_image_hw[:, 1].view(-1, 1).clamp_min(1.0)

    u = node_coords[..., 0]
    v = node_coords[..., 1]

    u_norm = u / (w - 1.0).clamp_min(1.0)
    v_norm = v / (h - 1.0).clamp_min(1.0)

    yaw = (u_norm - 0.5) * (2.0 * torch.pi)
    elev = (0.5 - v_norm) * torch.pi

    cos_elev = torch.cos(elev)
    ray_x = cos_elev * torch.sin(yaw)
    ray_y = cos_elev * torch.cos(yaw)
    ray_z = torch.sin(elev)
    rays = torch.stack([ray_x, ray_y, ray_z], dim=-1)

    p_cam = rays * node_depths.unsqueeze(-1)
    return p_cam @ R_left2world.transpose(-2, -1) + t_left2world.unsqueeze(1)

@torch.jit.script
def align_3d_grounded_pinhole(
    node_coords: torch.Tensor,
    node_depths: torch.Tensor,
    K_left: torch.Tensor,
    R_left2world: torch.Tensor,
    t_left2world: torch.Tensor,
    sat_affine_A: torch.Tensor,
    sat_affine_b: torch.Tensor,
) -> torch.Tensor:

    p_world = backproject_nodes_to_world_pinhole(
        node_coords=node_coords,
        node_depths=node_depths,
        K_left=K_left,
        R_left2world=R_left2world,
        t_left2world=t_left2world,
    )
    p_xy = p_world[..., :2]
    pix = p_xy @ sat_affine_A.transpose(-2, -1) + sat_affine_b.unsqueeze(1)
    return pix


@torch.jit.script
def align_3d_grounded_panorama(
    node_coords: torch.Tensor,
    node_depths: torch.Tensor,
    left_image_hw: torch.Tensor,
    R_left2world: torch.Tensor,
    t_left2world: torch.Tensor,
    sat_affine_A: torch.Tensor,
    sat_affine_b: torch.Tensor,
) -> torch.Tensor:
    """Project panorama nodes into satellite pixels using world->sat affine mapping."""
    p_world = backproject_nodes_to_world_panorama(
        node_coords=node_coords,
        node_depths=node_depths,
        left_image_hw=left_image_hw,
        R_left2world=R_left2world,
        t_left2world=t_left2world,
    )
    p_xy = p_world[..., :2]
    pix = p_xy @ sat_affine_A.transpose(-2, -1) + sat_affine_b.unsqueeze(1)
    return pix

