import math
import os
import random
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from transform import matrix_to_euler_zyx, replace_yaw_keep_roll_pitch
from .augment import Augmentation


def _round_to_multiple(x: float, multiple: int = 16, min_value: int = 16) -> int:
    out = int(round(x / multiple) * multiple)
    return max(min_value, out)


def resolve_patch_align(dino_model_name) -> int:
    if dino_model_name.startswith("dinov2_"):
        return 14
    elif dino_model_name.startswith("dinov3_"):
        return 16
    else:
        raise ValueError(f"Unsupported DINO model name: {dino_model_name}")


def compute_resize_from_long_edge(orig_w: int, orig_h: int, long_edge: int, align: int = 16) -> Tuple[int, int, float, float]:
    scale = float(long_edge) / float(max(orig_w, orig_h))
    new_w = _round_to_multiple(orig_w * scale, multiple=align)
    new_h = _round_to_multiple(orig_h * scale, multiple=align)
    scale_x = new_w / float(orig_w)
    scale_y = new_h / float(orig_h)
    return new_h, new_w, scale_x, scale_y


class SatImageAugment:
    def __init__(self):
        pass

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        sat_affine_A: torch.Tensor,
        sat_affine_b: torch.Tensor,
        rotate: float,
        world_offset_xy: torch.Tensor,
    ):
        """Augment satellite-view image by rotation and translation"""
        w, h = image.size

        pix_shift = sat_affine_A @ world_offset_xy
        sat_affine_b = sat_affine_b - pix_shift

        image = image.transform(
            image.size,
            Image.AFFINE,
            data=[1.0, 0.0, pix_shift[0].item(), 0.0, 1.0, pix_shift[1].item()],
            resample=Image.BILINEAR,
        )

        cos_r, sin_r = math.cos(rotate), math.sin(rotate)
        R_2d = torch.tensor([[cos_r, -sin_r], [sin_r, cos_r]], dtype=torch.float32)
        center = torch.tensor([(w - 1) * 0.5, (h - 1) * 0.5], dtype=torch.float32)
        trans_params = center - R_2d @ center

        image_new = image.transform(
            image.size,
            Image.AFFINE,
            data=[
                R_2d[0, 0].item(),
                R_2d[0, 1].item(),
                trans_params[0].item(),
                R_2d[1, 0].item(),
                R_2d[1, 1].item(),
                trans_params[1].item(),
            ],
            resample=Image.BILINEAR,
        )

        R_inv = torch.linalg.inv(R_2d)
        sat_affine_A_new = R_inv @ sat_affine_A
        sat_affine_b_new = R_inv @ (sat_affine_b - trans_params)

        return image_new, sat_affine_A_new, sat_affine_b_new

def png_depth_to_tensor(file_path: str, out_h: int, out_w: int, scale: float = 100.0) -> torch.Tensor:
    with Image.open(file_path) as img:
        depth_np = np.array(img, dtype=np.float32) / scale
    depth = torch.from_numpy(depth_np)
    depth = F.interpolate(depth[None, None], (out_h, out_w), mode="bilinear", align_corners=False)
    return depth.squeeze(0).squeeze(0)

class LocDataset(Dataset):
    def __init__(
        self,
        pairs,
        left_image_size,
        sat_image_size,
        patch_align=16,
        is_train_split=False,
        aug=False,
        max_init_offset=20,
        max_init_yaw_deg=15.0,
        max_aug_offset=50,
        max_aug_rotate=180,
        seed=None,
        supervise='full',
    ):
        super().__init__()
        self.pairs = pairs
        self.left_to_tensor = ToTensor()
        self.sat_to_tensor = ToTensor()

        
        self.patch_align = int(patch_align)
        self.left_image_size = self._get_aligned_size(left_image_size)
        self.sat_image_size = self._get_aligned_size(sat_image_size)
        self.is_train_split = bool(is_train_split)

        self.aug = Augmentation(seed) if aug else None
        self.sat_aug = SatImageAugment()

        self.max_init_offset = max_init_offset
        self.max_init_yaw = max_init_yaw_deg / 180.0 * np.pi
        
        self.max_aug_offset = max_aug_offset
        self.max_aug_rotate = max_aug_rotate

        assert supervise in ('full', 'weak_3dof', 'weak_2dof'), f"Unsupported supervise mode {supervise!r}."
        self.supervise = supervise

    def __len__(self):
        return len(self.pairs)

    def deep_copy_dict(self, data: Dict):
        new_data = {}
        for key, val in data.items():
            if isinstance(val, (str, float, int)):
                new_data[key] = val
            elif isinstance(val, torch.Tensor):
                new_data[key] = val.clone()
            else:
                raise NotImplementedError(f"Unsupported data type: {type(val)}")
        return new_data

    def _load_left_image(self, path: str) -> Tuple[torch.Tensor, float, float]:
        with Image.open(path).convert("RGB") as left_image:
            orig_w, orig_h = left_image.size
            out_h, out_w, sx, sy = compute_resize_from_long_edge(orig_w, orig_h, self.left_image_size, align=self.patch_align)
            if self.aug:
                left_image = self.aug(left_image)
            left_image = left_image.resize((out_w, out_h), Image.BILINEAR)
            left_image = self.left_to_tensor(left_image)
        return left_image, sx, sy

    def _get_aligned_size(self, size: int) -> int:
        return _round_to_multiple(size, multiple=self.patch_align, min_value=self.patch_align)

    def _load_sat_image(self, path: str, pair: Dict) -> Tuple[torch.Tensor, Dict]:
        with Image.open(path).convert("RGB") as sat_image:
            orig_w, orig_h = sat_image.size

            if self.aug:
                random_rotate = (random.random() * 2 - 1) * self.max_aug_rotate / 180 * np.pi
                random_offset = (torch.rand(2) * 2 - 1) * self.max_aug_offset

                sat_image, pair["sat_affine_A"], pair["sat_affine_b"] = self.sat_aug(
                    sat_image,
                    pair["sat_affine_A"],
                    pair["sat_affine_b"],
                    random_rotate,
                    random_offset,
                )
                sat_image = self.aug(sat_image)

            sat_image_size = self.sat_image_size
            sat_image = sat_image.resize((sat_image_size, sat_image_size), Image.BILINEAR)
            sat_image = self.sat_to_tensor(sat_image)

        sat_scale_x = sat_image_size / float(orig_w)
        sat_scale_y = sat_image_size / float(orig_h)
        return sat_image, sat_scale_x, sat_scale_y

    def _load_depth(self, depth_path: str, out_h: int, out_w: int) -> torch.Tensor:
        ext = os.path.splitext(depth_path)[1].lower()
        if ext == ".png":
            return png_depth_to_tensor(depth_path, out_h=out_h, out_w=out_w, scale=100.0)
        raise ValueError(f"Unsupported depth extension: {depth_path}")

    def _is_vigor_pair(self, pair: Dict) -> bool:
        left_path = os.path.normpath(pair["left_image_path"])
        return "VIGOR" in left_path.split(os.sep)

    def _vigor_yaw_to_roll_shift(self, yaw: torch.Tensor | float, width: int) -> int:
        yaw_value = float(yaw.squeeze().item()) if isinstance(yaw, torch.Tensor) else float(yaw)
        return int(round(yaw_value / (2.0 * math.pi) * width))

    def __getitem__(self, index):
        pair = self.deep_copy_dict(self.pairs[index])
        is_vigor = self._is_vigor_pair(pair)

        left_image_path = pair["left_image_path"]
        left_mono_depth_path = pair['left_mono_depth_path']

        sat_image_path = pair["sat_image_path"]

        t_left2world = pair["t_left2world"]
        R_left2world = pair["R_left2world"]

        x_offset_ratio = pair['x_offset_ratio']
        y_offset_ratio = pair['y_offset_ratio']
        yaw_offset_ratio = pair['yaw_offset_ratio']

        x_offset = x_offset_ratio * self.max_init_offset
        y_offset = y_offset_ratio * self.max_init_offset
        t_left2world_init = t_left2world.clone()
        t_left2world_init[0:2] += torch.tensor([x_offset, y_offset])

        _, _, yaw_gt = matrix_to_euler_zyx(R_left2world.unsqueeze(0))
        yaw_noise = yaw_offset_ratio * self.max_init_yaw
        if is_vigor:
            # apply yaw noise to panorama and roll it later
            yaw_gt = yaw_gt + yaw_noise
            yaw_init = torch.zeros_like(yaw_gt)
            R_left2world = replace_yaw_keep_roll_pitch(R_left2world.unsqueeze(0), yaw_gt).squeeze(0)
            pair["R_left2world"] = R_left2world
        else:
            # apply yaw noise to initial pose
            yaw_init = yaw_gt + yaw_noise

        R_left2world_init = replace_yaw_keep_roll_pitch(R_left2world.unsqueeze(0), yaw_init).squeeze(0)

        left_image, left_sx, left_sy = self._load_left_image(left_image_path)
        _, out_h, out_w = left_image.shape

        sat_image, sat_scale_x, sat_scale_y = self._load_sat_image(sat_image_path, pair)

        left_mono_depth_map = self._load_depth(left_mono_depth_path, out_h=out_h, out_w=out_w)

        if is_vigor:
            # roll panorama
            roll_shift = self._vigor_yaw_to_roll_shift(yaw_noise, out_w)
            if roll_shift:
                left_image = torch.roll(left_image, shifts=roll_shift, dims=-1)
                left_mono_depth_map = torch.roll(left_mono_depth_map, shifts=roll_shift, dims=-1)

        pair["left_image"] = left_image
        pair["sat_image"] = sat_image
        pair["left_mono_depth_map"] = left_mono_depth_map
        pair["t_left2world_init"] = t_left2world_init
        pair["R_left2world_init"] = R_left2world_init

        if "K_left" in pair:
            pair["K_left"] = pair["K_left"].clone()
            pair["K_left"][0, :] = pair["K_left"][0, :] * left_sx
            pair["K_left"][1, :] = pair["K_left"][1, :] * left_sy
            pair["K_left"][2, :] = torch.tensor([0.0, 0.0, 1.0], dtype=pair["K_left"].dtype)

        # scale sat affine parameters
        pair["sat_affine_A"] = pair["sat_affine_A"].clone()
        pair["sat_affine_A"][0, :] = pair["sat_affine_A"][0, :] * sat_scale_x
        pair["sat_affine_A"][1, :] = pair["sat_affine_A"][1, :] * sat_scale_y
        pair["sat_affine_b"] = pair["sat_affine_b"].clone()
        pair["sat_affine_b"][0] = pair["sat_affine_b"][0] * sat_scale_x
        pair["sat_affine_b"][1] = pair["sat_affine_b"][1] * sat_scale_y

        pair["max_init_offset"] = torch.tensor(self.max_init_offset, dtype=torch.float32)
        pair["max_init_yaw"] = torch.tensor(self.max_init_yaw, dtype=torch.float32)

        return pair

    def collate_fn(self, pairs):
        collated_pairs = {}

        for key in pairs[0].keys():
            collated_data = [pair[key] for pair in pairs]
            if isinstance(collated_data[0], torch.Tensor):
                collated_pairs[key] = torch.stack(collated_data, dim=0)
            else:
                collated_pairs[key] = collated_data

        # random resolution augmentation
        if self.aug:
            cur_h, cur_w = collated_pairs["left_image"].shape[-2:]
            random_long = random.randint(16, max(16, self.left_image_size // self.patch_align)) * self.patch_align
            if random_long > max(cur_h, cur_w):
                random_long = max(cur_h, cur_w)
            aug_h, aug_w, sx, sy = compute_resize_from_long_edge(cur_w, cur_h, random_long, align=self.patch_align)

            collated_pairs["left_image"] = F.interpolate(
                collated_pairs["left_image"],
                size=(aug_h, aug_w),
                mode="bilinear",
                align_corners=False,
            )

            collated_pairs["left_mono_depth_map"] = F.interpolate(
                collated_pairs["left_mono_depth_map"].unsqueeze(1),
                size=(aug_h, aug_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            # re-scale K_left
            if "K_left" in collated_pairs:
                collated_pairs["K_left"] = collated_pairs["K_left"].clone()
                collated_pairs["K_left"][..., 0, :] = collated_pairs["K_left"][..., 0, :] * sx
                collated_pairs["K_left"][..., 1, :] = collated_pairs["K_left"][..., 1, :] * sy
                collated_pairs["K_left"][..., 2, :] = torch.tensor([0.0, 0.0, 1.0], device=collated_pairs["K_left"].device)

            aug_sat_size = random.randint(16, max(16, self.sat_image_size // self.patch_align)) * self.patch_align
            if aug_sat_size > self.sat_image_size:
                aug_sat_size = self.sat_image_size
            aug_sat_size = self._get_aligned_size(aug_sat_size)
            collated_pairs["sat_image"] = F.interpolate(
                collated_pairs["sat_image"],
                size=(aug_sat_size, aug_sat_size),
                mode="bilinear",
                align_corners=False,
            )

            sat_sx = aug_sat_size / float(self.sat_image_size)
            sat_sy = sat_sx
            
            collated_pairs["sat_affine_A"] = collated_pairs["sat_affine_A"].clone()
            collated_pairs["sat_affine_A"][..., 0, :] = collated_pairs["sat_affine_A"][..., 0, :] * sat_sx
            collated_pairs["sat_affine_A"][..., 1, :] = collated_pairs["sat_affine_A"][..., 1, :] * sat_sy
            collated_pairs["sat_affine_b"] = collated_pairs["sat_affine_b"].clone()
            collated_pairs["sat_affine_b"][..., 0] = collated_pairs["sat_affine_b"][..., 0] * sat_sx
            collated_pairs["sat_affine_b"][..., 1] = collated_pairs["sat_affine_b"][..., 1] * sat_sy

        input_data = {
            'left_image': collated_pairs["left_image"],
            'sat_image': collated_pairs["sat_image"],
            'R_left2world_init': collated_pairs["R_left2world_init"],
            't_left2world_init': collated_pairs["t_left2world_init"],
            'sat_affine_A': collated_pairs["sat_affine_A"],
            'sat_affine_b': collated_pairs["sat_affine_b"],
            # R_left2world_gt provided during fully and 2-DoF weakly supervised training
            'R_left2world_gt': collated_pairs["R_left2world"] if (self.is_train_split and self.supervise != 'weak_3dof') else None,
            # t_left2world_gt provided during fully supervised training
            't_left2world_gt': collated_pairs["t_left2world"] if (self.is_train_split and self.supervise == 'full') else None,
            'max_init_offset': collated_pairs["max_init_offset"],
            'max_init_yaw': collated_pairs["max_init_yaw"],
        }

        if "K_left" in collated_pairs:
            input_data['K_left'] = collated_pairs["K_left"]

        label_data = {
            'R_left2world': collated_pairs["R_left2world"],
            't_left2world': collated_pairs["t_left2world"],
            'left_mono_depth_map': collated_pairs["left_mono_depth_map"],
        }

        data = {
            'input': input_data,
            'label': label_data,
        }

        return data


def read_data(data_dir, dataset_name, keys=("train", "val", "test"), split_type=None):
    dataset_dir = os.path.join(data_dir, dataset_name)
    if dataset_name.upper() == "VIGOR":
        vigor_split = split_type or "same_area"
        if vigor_split not in ("same_area", "cross_area"):
            raise ValueError(f"Unsupported VIGOR split_type: {vigor_split}")
        dataset_dir = os.path.join(dataset_dir, vigor_split)

    data_dict = {}
    for key in keys:
        data_file = os.path.join(dataset_dir, f"{key}_data.pth")
        data = torch.load(data_file, weights_only=False, map_location="cpu")
        for i in range(len(data)):
            data[i]["left_image_path"] = os.path.join(data_dir, data[i]["left_image_path"])
            data[i]["sat_image_path"] = os.path.join(data_dir, data[i]["sat_image_path"])
            if "left_depth_path" in data[i] and data[i]["left_depth_path"]:
                data[i]["left_bino_depth_path"] = os.path.join(data_dir, data[i]["left_depth_path"])
            if "left_bino_depth_path" in data[i] and data[i]["left_bino_depth_path"]:
                data[i]["left_bino_depth_path"] = os.path.join(data_dir, data[i]["left_bino_depth_path"])
            if "left_mono_depth_path" in data[i] and data[i]["left_mono_depth_path"]:
                data[i]["left_mono_depth_path"] = os.path.join(data_dir, data[i]["left_mono_depth_path"])

        data_dict[key] = data

    return data_dict
