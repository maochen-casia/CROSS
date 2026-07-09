import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import random


def _parse_split_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    # test split lines may include extra shift/theta numbers.
    return line.split()[0]


def _read_split_file(file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    entries: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            rel = _parse_split_line(line)
            if rel:
                entries.append(rel)
    return entries


def _parse_p_rect_02(calib_file: Path) -> torch.Tensor:
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("P_rect_02"):
                vals = [float(x) for x in line.split(":", 1)[1].strip().split()]
                if len(vals) != 12:
                    raise ValueError(f"Unexpected P_rect_02 size in {calib_file}")
                P = torch.tensor(vals, dtype=torch.float32).reshape(3, 4)
                return P[:, :3]
    raise ValueError(f"P_rect_02 not found in {calib_file}")


def _parse_kitti_calib(calib_file: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    with open(calib_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if not parts:
                continue
            try:
                arr = np.array([float(x) for x in parts], dtype=np.float64)
            except ValueError:
                continue
            if arr.size:
                data[key] = arr
    return data


def _build_transform(R_flat: np.ndarray, t_flat: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_flat.reshape(3, 3)
    T[:3, 3] = t_flat.reshape(3)
    return T


def _build_imu_to_rectcam2_transform(
    calib_cam_to_cam: Dict[str, np.ndarray],
    calib_imu_to_velo: Dict[str, np.ndarray],
    calib_velo_to_cam: Dict[str, np.ndarray],
) -> np.ndarray:
    t_imu_velo = _build_transform(calib_imu_to_velo["R"], calib_imu_to_velo["T"])
    t_velo_cam0 = _build_transform(calib_velo_to_cam["R"], calib_velo_to_cam["T"])
    t_cam0_cam2 = _build_transform(calib_cam_to_cam["R_02"], calib_cam_to_cam["T_02"])

    t_rect_02 = np.eye(4, dtype=np.float64)
    t_rect_02[:3, :3] = calib_cam_to_cam["R_rect_02"].reshape(3, 3)

    return t_rect_02 @ t_cam0_cam2 @ t_velo_cam0 @ t_imu_velo


def _load_oxts_pose(oxts_file: Path) -> Tuple[float, float, float, float, float, float]:
    with open(oxts_file, "r", encoding="utf-8") as f:
        fields = f.readline().strip().split()
    if len(fields) < 6:
        raise ValueError(f"Invalid oxts line in {oxts_file}")
    lat = float(fields[0])
    lon = float(fields[1])
    alt = float(fields[2])
    roll = float(fields[3])
    pitch = float(fields[4])
    yaw = float(fields[5])
    return lat, lon, alt, roll, pitch, yaw


def _euler_zyx_to_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = torch.tensor([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=torch.float32)
    return R


def _latlon_to_mercator(lat: float, lon: float, scale: float) -> Tuple[float, float]:
    r = 6378137.0
    x = scale * lon * math.pi * r / 180.0
    y = scale * r * math.log(math.tan(math.pi * (90.0 + lat) / 360.0))
    return x, y


def _meter_per_pixel(lat: float, zoom: int = 18, sat_scale: float = 1.0) -> float:
    mpp = 156543.03392 * math.cos(math.radians(lat)) / (2**zoom)
    mpp /= 2.0
    mpp /= sat_scale
    return mpp


def _build_record(
    kitti_root: Path,
    rel_file: str,
    calib_cache: Dict[str, torch.Tensor],
    extrinsic_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    drive_ref: Dict[str, Tuple[float, float, float]],
) -> Dict:
    date_dir, drive_dir, file_name = rel_file.split("/")
    frame_stem = Path(file_name).stem

    drive_rel = f"{date_dir}/{drive_dir}"

    left_image_abs = kitti_root / "raw_data" / drive_rel / "image_02" / "data" / file_name
    sat_image_abs = kitti_root / "satmap" / drive_rel / file_name
    oxts_abs = kitti_root / "raw_data" / drive_rel / "oxts" / "data" / f"{frame_stem}.txt"
    calib_abs = kitti_root / "raw_data" / date_dir / "calib_cam_to_cam.txt"

    if not left_image_abs.exists():
        raise FileNotFoundError(f"Missing left image: {left_image_abs}")
    if not sat_image_abs.exists():
        raise FileNotFoundError(f"Missing sat image: {sat_image_abs}")
    if not oxts_abs.exists():
        raise FileNotFoundError(f"Missing oxts: {oxts_abs}")

    if date_dir not in calib_cache:
        calib_cache[date_dir] = _parse_p_rect_02(calib_abs)
    K_left = calib_cache[date_dir].clone()

    if date_dir not in extrinsic_cache:
        calib_root = kitti_root / "raw_data" / date_dir
        calib_cam_to_cam = _parse_kitti_calib(calib_root / "calib_cam_to_cam.txt")
        calib_imu_to_velo = _parse_kitti_calib(calib_root / "calib_imu_to_velo.txt")
        calib_velo_to_cam = _parse_kitti_calib(calib_root / "calib_velo_to_cam.txt")

        t_imu_to_cam2_rect = _build_imu_to_rectcam2_transform(
            calib_cam_to_cam=calib_cam_to_cam,
            calib_imu_to_velo=calib_imu_to_velo,
            calib_velo_to_cam=calib_velo_to_cam,
        )
        t_cam2rect_to_imu = np.linalg.inv(t_imu_to_cam2_rect)
        R_cam2imu = torch.tensor(t_cam2rect_to_imu[:3, :3], dtype=torch.float32)
        t_cam2imu = torch.tensor(t_cam2rect_to_imu[:3, 3], dtype=torch.float32)
        extrinsic_cache[date_dir] = (R_cam2imu, t_cam2imu)

    R_cam2imu, t_cam2imu = extrinsic_cache[date_dir]

    lat, lon, alt, roll, pitch, yaw = _load_oxts_pose(oxts_abs)

    if drive_rel not in drive_ref:
        scale = math.cos(math.radians(lat))
        x0, y0 = _latlon_to_mercator(lat, lon, scale)
        drive_ref[drive_rel] = (scale, x0, y0)

    scale, x0, y0 = drive_ref[drive_rel]
    x, y = _latlon_to_mercator(lat, lon, scale)
    x -= x0
    y -= y0

    # Local metric world frame for KITTI: x=east, y=north, z=up.
    t_imu2world = torch.tensor([x, y, alt], dtype=torch.float32)
    R_imu2world = _euler_zyx_to_matrix(roll, pitch, yaw)

    # True left camera pose from IMU pose + fixed calibration extrinsics.
    R_left2world = R_imu2world @ R_cam2imu
    t_left2world = R_imu2world @ t_cam2imu + t_imu2world

    # Satellite mapping as affine world->pixel (north-up map, centered on current frame GPS).
    sat_size = 1280
    mpp = _meter_per_pixel(lat=lat, zoom=18, sat_scale=1.0)
    sat_affine_A = torch.tensor([[1.0 / mpp, 0.0], [0.0, -1.0 / mpp]], dtype=torch.float32)
    sat_affine_b = torch.tensor([
        (sat_size - 1.0) * 0.5 - x / mpp,
        (sat_size - 1.0) * 0.5 + y / mpp,
    ], dtype=torch.float32)

    data = {
        "left_image_path": str(Path("KITTI") / "raw_data" / drive_rel / "image_02" / "data" / file_name),
        "left_mono_depth_path": str(Path("KITTI") / "raw_data" / drive_rel / "mono_depth" / f"{frame_stem}.png"),
        "sat_image_path": str(Path("KITTI") / "satmap" / drive_rel / file_name),
        "K_left": K_left,
        "R_left2world": R_left2world,
        "t_left2world": t_left2world,
        "K_sat": torch.eye(3, dtype=torch.float32),
        "R_sat2world": torch.eye(3, dtype=torch.float32),
        "t_sat2world": torch.zeros(3, dtype=torch.float32),
        "sat_affine_A": sat_affine_A,
        "sat_affine_b": sat_affine_b,
        "x_offset_ratio": random.uniform(-1, 1),
        "y_offset_ratio": random.uniform(-1, 1),
        'yaw_offset_ratio': random.uniform(-1, 1)
    }
    return data


def build_split(kitti_root: Path, rel_files: List[str]) -> List[Dict]:
    calib_cache: Dict[str, torch.Tensor] = {}
    extrinsic_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    drive_ref: Dict[str, Tuple[float, float, float]] = {}
    records: List[Dict] = []

    for rel_file in rel_files:
        try:
            rec = _build_record(kitti_root, rel_file, calib_cache, extrinsic_cache, drive_ref)
            records.append(rec)
        except FileNotFoundError as e:
            print(f"[WARN] {e}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KITTI train/val/test metadata for WARG.")
    parser.add_argument("--train_split", type=str, default="train_files.txt")
    parser.add_argument("--val_split", type=str, default="test1_files.txt")
    parser.add_argument("--test_split", type=str, default="test2_files.txt")
    args = parser.parse_args()

    kitti_root = 'your/local/path/to/KITTI'
    output_dir = Path(kitti_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_to_file = {
        "train": f'./{args.train_split}',
        "val": f'./{args.val_split}',
        "test": f'./{args.test_split}',
    }

    for split_name, file_path in split_to_file.items():
        rel_files = _read_split_file(file_path)
        print(f"Building {split_name} with {len(rel_files)} entries from {file_path}")
        records = build_split(kitti_root=kitti_root, rel_files=rel_files)
        out_file = output_dir / f"{split_name}_data.pth"
        torch.save(records, out_file)
        print(f"Saved {len(records)} records to {out_file}")


if __name__ == "__main__":
    main()
