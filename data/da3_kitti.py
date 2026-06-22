import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pathlib import Path
import sys

import numpy as np
import torch


DEFAULT_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def _import_depth_anything3():
    """Import DepthAnything3, with a local path fallback for ../Depth-Anything-3/src."""
    try:
        from depth_anything_3.api import DepthAnything3

        return DepthAnything3
    except ModuleNotFoundError:
        local_src = Path(__file__).resolve().parents[2] / "Depth-Anything-3" / "src"
        if local_src.exists():
            sys.path.insert(0, str(local_src))
            from depth_anything_3.api import DepthAnything3

            return DepthAnything3
        raise


def build_model(device=None, model_id: str = DEFAULT_MODEL_ID):
    """Build and return a Depth-Anything 3 model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    DepthAnything3 = _import_depth_anything3()
    model = DepthAnything3.from_pretrained(model_id)
    model = model.to(device=device)
    model.eval()
    return model


def process_images(image_paths):
    """Prepare image paths for DA3 inference."""
    if len(image_paths) == 0:
        raise ValueError("image_paths must contain at least one image path")
    return list(image_paths)


def _squeeze_leading_batch(arr, expected_rank):
    while arr.ndim > expected_rank and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _as_depth_array(depth):
    arr = np.asarray(depth)
    arr = _squeeze_leading_batch(arr, expected_rank=3)

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected depth with shape [N,H,W], got {arr.shape}")
    return arr.astype(np.float32)


def _as_intrinsics_array(intrinsics):
    arr = np.asarray(intrinsics)
    arr = _squeeze_leading_batch(arr, expected_rank=3)
    if arr.ndim != 3 or arr.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics with shape [N,3,3], got {arr.shape}")
    return arr.astype(np.float32)


def _as_extrinsics_array(extrinsics):
    arr = np.asarray(extrinsics)
    arr = _squeeze_leading_batch(arr, expected_rank=3)
    if arr.ndim != 3 or arr.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected extrinsics with shape [N,3,4] or [N,4,4], got {arr.shape}")
    return arr.astype(np.float32)


def infer(model, images, **inference_kwargs):
    """Run raw DA3 inference and return the Prediction object."""
    with torch.inference_mode():
        return model.inference(images, **inference_kwargs)


def predict(
    model,
    images,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    **inference_kwargs,
):
    """Run DA3 inference and normalize core prediction fields used in demos."""
    inference_kwargs = dict(inference_kwargs)
    inference_kwargs.setdefault("process_res", process_res)
    inference_kwargs.setdefault("process_res_method", process_res_method)

    prediction = infer(model, images, **inference_kwargs)

    depth = getattr(prediction, "depth", None)
    intrinsics = getattr(prediction, "intrinsics", None)
    extrinsics = getattr(prediction, "extrinsics", None)

    if depth is None:
        raise KeyError("DA3 prediction missing depth")
    if intrinsics is None:
        raise KeyError("DA3 prediction missing intrinsics")
    if extrinsics is None:
        raise KeyError("DA3 prediction missing extrinsics")

    return {
        "depth": _as_depth_array(depth),
        "intrinsic": _as_intrinsics_array(intrinsics),
        "extrinsic": _as_extrinsics_array(extrinsics),
        "conf": getattr(prediction, "conf", None),
        "processed_images": getattr(prediction, "processed_images", None),
    }



def collect_kitti_drives(root_dir: Path) -> List[Path]:
    drives = []
    for date_dir in sorted(root_dir.glob("2011_*_*")):
        if date_dir.is_dir():
            drives.extend(sorted(d for d in date_dir.glob("*_drive_*_sync") if d.is_dir()))
    return drives


def collect_frame_paths(drive_dir: Path, image_subdir: str) -> List[Path]:
    image_dir = drive_dir / image_subdir
    if not image_dir.is_dir():
        return []
    frames = sorted(image_dir.glob("*.png"))
    return frames or sorted(image_dir.glob("*.jpg"))


def depth_to_png_cm(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    depth_cm = np.zeros(depth.shape, dtype=np.uint16)
    depth_cm[valid] = np.clip(np.rint(depth[valid] * 100.0), 0, 65535).astype(np.uint16)
    return depth_cm


def save_depth_png(depth_m: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_to_png_cm(depth_m)).save(out_path)


def pending_pairs(
    frame_paths: Sequence[Path],
    output_dir: Path,
    overwrite: bool,
) -> List[Tuple[Path, Path]]:
    pairs = []
    for frame_path in frame_paths:
        out_path = output_dir / f"{frame_path.stem}.png"
        if overwrite or not out_path.exists():
            pairs.append((frame_path, out_path))
    return pairs


def export_drive_depths(
    model,
    drive_dir: Path,
    frame_paths: Sequence[Path],
    output_subdir: str,
    process_res: int,
    process_res_method: str,
    batch_size: int,
    overwrite: bool,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    pairs = pending_pairs(frame_paths, drive_dir / output_subdir, overwrite)
    written = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prediction = infer(
            model,
            [str(frame_path) for frame_path, _ in batch],
            process_res=process_res,
            process_res_method=process_res_method,
        )

        depth = np.asarray(prediction.depth, dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None]
        if depth.ndim != 3 or depth.shape[0] != len(batch):
            raise RuntimeError(f"Unexpected DA3 depth shape for {drive_dir.name}: {depth.shape}")

        for idx, (_, out_path) in enumerate(batch):
            save_depth_png(depth[idx], out_path)
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DA3 monocular depth prediction on KITTI raw drives."
    )
    parser.add_argument("--root-dir", type=str, required=True, help="Path to the KITTI raw_data directory.")
    parser.add_argument("--model-id", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--image-subdir", default="image_02/data")
    parser.add_argument("--output-subdir", default="mono_depth")
    parser.add_argument("--process-res", type=int, default=1008)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--date-name", default=None, help="Optional date folder, e.g. 2011_09_26.")
    parser.add_argument("--drive-name", default=None, help="Optional drive folder, e.g. 2011_09_26_drive_0001_sync.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--da3-log-level", default="WARN", choices=["ERROR", "WARN", "INFO", "DEBUG"])
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["DA3_LOG_LEVEL"] = args.da3_log_level

    root_dir = Path(args.root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"KITTI raw root not found: {root_dir}")

    drives = collect_kitti_drives(root_dir)
    if args.date_name:
        drives = [drive for drive in drives if drive.parent.name == args.date_name]
    if args.drive_name:
        drives = [drive for drive in drives if drive.name == args.drive_name]

    drive_frames = [
        (drive, frames)
        for drive in drives
        if (frames := collect_frame_paths(drive, args.image_subdir))
    ]
    total_frames = sum(len(frames) for _, frames in drive_frames)
    if total_frames == 0:
        print("No KITTI frames found.")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {args.model_id}")
    model = build_model(device=device, model_id=args.model_id)

    written = 0
    with tqdm(total=total_frames, desc="KITTI DA3 mono", unit="frame") as pbar:
        for drive_dir, frames in drive_frames:
            drive_written = export_drive_depths(
                model=model,
                drive_dir=drive_dir,
                frame_paths=frames,
                output_subdir=args.output_subdir,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
            written += drive_written
            pbar.update(len(frames))
            pbar.set_postfix_str(f"{drive_dir.name}: +{drive_written} png")

    print(f"Done. Wrote {written} PNG files to '{args.output_subdir}'.")


if __name__ == "__main__":
    main()
