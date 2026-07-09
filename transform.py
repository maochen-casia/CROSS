import torch


def euler_zyx_to_matrix(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Build batched rotation matrices from ZYX Euler angles (R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr

    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr

    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    row0 = torch.stack([r00, r01, r02], dim=-1)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def matrix_to_euler_zyx(R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract batched ZYX Euler angles (roll, pitch, yaw) from rotation matrices."""
    pitch = torch.asin(torch.clamp(-R[..., 2, 0], -1.0, 1.0))
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    return roll, pitch, yaw


def replace_yaw_keep_roll_pitch(R: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Replace yaw while keeping roll and pitch components unchanged."""
    roll, pitch, _ = matrix_to_euler_zyx(R)
    return euler_zyx_to_matrix(roll, pitch, yaw)
