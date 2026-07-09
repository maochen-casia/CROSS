import argparse
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from tqdm import tqdm


CITY_MPP_AT_640: Dict[str, float] = {
    "NewYork": 0.113248,
    "Seattle": 0.100817,
    "SanFrancisco": 0.118141,
    "Chicago": 0.111262,
}


def _city_list(split_type: str, train: bool) -> Sequence[str]:
    if split_type == "same_area":
        return ["NewYork", "Seattle", "SanFrancisco", "Chicago"]
    if split_type == "cross_area":
        return ["NewYork", "Seattle"] if train else ["SanFrancisco", "Chicago"]
    raise ValueError(f"Unsupported split_type: {split_type}")


def _split_filename(split_type: str, train: bool) -> str:
    if split_type == "same_area":
        return "same_area_balanced_train__corrected.txt" if train else "same_area_balanced_test__corrected.txt"
    if split_type == "cross_area":
        return "pano_label_balanced__corrected.txt"
    raise ValueError(f"Unsupported split_type: {split_type}")


def _read_split_entries(split_file: Path) -> Iterable[Tuple[str, str, float, float]]:
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            pano_name = parts[0]
            sat_name = parts[1]
            row_offset = float(parts[2])
            col_offset = float(parts[3])
            yield pano_name, sat_name, row_offset, col_offset


def _build_record(vigor_root: Path, city: str, pano_name: str, sat_name: str, row_offset: float, col_offset: float) -> Dict:
    left_image_abs = vigor_root / city / "panorama" / pano_name
    sat_image_abs = vigor_root / city / "satellite" / sat_name

    if not left_image_abs.exists():
        raise FileNotFoundError(f"Missing panorama image: {left_image_abs}")
    if not sat_image_abs.exists():
        raise FileNotFoundError(f"Missing satellite image: {sat_image_abs}")

    with torch.no_grad():
        from PIL import Image

        with Image.open(left_image_abs) as left_img:
            left_w, left_h = left_img.size
        with Image.open(sat_image_abs) as sat_img:
            sat_w, sat_h = sat_img.size

    mpp = CITY_MPP_AT_640[city] * (640.0 / float(sat_w))

    # Corrected split convention:
    # cam_x_px = center_x - col_offset, cam_y_px = center_y + row_offset.
    # We set world origin at satellite-image center with x=east, y=north.
    t_x = -col_offset * mpp
    t_y = -row_offset * mpp

    sat_affine_A = torch.tensor([[1.0 / mpp, 0.0], [0.0, -1.0 / mpp]], dtype=torch.float32)
    sat_affine_b = torch.tensor([(sat_w - 1.0) * 0.5, (sat_h - 1.0) * 0.5], dtype=torch.float32)

    data = {
        "left_image_path": str(Path("VIGOR") / city / "panorama" / pano_name),
        "sat_image_path": str(Path("VIGOR") / city / "satellite" / sat_name),
        'left_mono_depth_path': str(Path("VIGOR") / city / "mono_depth" / pano_name.replace(".jpg", ".png")),
        "R_left2world": torch.eye(3, dtype=torch.float32),
        "t_left2world": torch.tensor([t_x, t_y, 0.0], dtype=torch.float32),
        "sat_affine_A": sat_affine_A,
        "sat_affine_b": sat_affine_b,
        "x_offset_ratio": random.uniform(-1.0, 1.0),
        "y_offset_ratio": random.uniform(-1.0, 1.0),
        "yaw_offset_ratio": random.uniform(-1.0, 1.0),
    }
    return data


def _build_split(vigor_root: Path, split_root: Path, split_type: str, train: bool) -> List[Dict]:
    records: List[Dict] = []

    split_tag = "train" if train else "eval"
    split_name = _split_filename(split_type=split_type, train=train)
    city_list = _city_list(split_type=split_type, train=train)
    for city in tqdm(city_list, desc=f"Cities ({split_tag})", leave=False):
        split_file = split_root / city / split_name
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        entries = list(_read_split_entries(split_file))
        for pano_name, sat_name, row_offset, col_offset in tqdm(
            entries,
            desc=f"{split_tag}:{city}",
            leave=False,
        ):
            try:
                records.append(
                    _build_record(
                        vigor_root=vigor_root,
                        city=city,
                        pano_name=pano_name,
                        sat_name=sat_name,
                        row_offset=row_offset,
                        col_offset=col_offset,
                    )
                )
            except FileNotFoundError as e:
                print(f"[WARN] {e}")

    return records


def _split_train_val(train_records: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if val_ratio <= 0.0:
        return train_records, []

    n = len(train_records)
    val_n = max(1, int(round(n * val_ratio)))

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_ids = set(indices[:val_n])
    train_out = [r for i, r in enumerate(train_records) if i not in val_ids]
    val_out = [r for i, r in enumerate(train_records) if i in val_ids]
    return train_out, val_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VIGOR train/val/test metadata for WARG.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="If > 0, sample val from train set.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    vigor_root = 'your/local/path/to/VIGOR'

    random.seed(args.seed)

    vigor_root = Path(vigor_root)
    split_root = "./splits__corrected"
    output_root = Path(vigor_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_type_to_dir = {
        "same_area": "same_area",
        "cross_area": "cross_area",
    }

    for split_type, split_dir_name in split_type_to_dir.items():
        output_dir = output_root / split_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        train_records = _build_split(vigor_root=vigor_root, split_root=split_root, split_type=split_type, train=True)
        eval_records = _build_split(vigor_root=vigor_root, split_root=split_root, split_type=split_type, train=False)

        train_records, sampled_val_records = _split_train_val(train_records, val_ratio=args.val_ratio, seed=args.seed)
        val_records = sampled_val_records if sampled_val_records else eval_records
        test_records = eval_records

        split_to_records = {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        }

        for split_name, records in split_to_records.items():
            out_file = output_dir / f"{split_name}_data.pth"
            torch.save(records, out_file)
            print(f"Saved {split_type}/{split_name}: {len(records)} samples -> {out_file}")


if __name__ == "__main__":
    main()
