from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from networks.models import make


def load_model(config):
    model_path = os.path.join(config["load_weights_dir"], "model.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading model weights from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(model_path, map_location=device)

    model = make(config["model"])
    if any(k.startswith("module") for k in state.keys()):
        model = nn.DataParallel(model)

    model = model.to(device)
    model_state = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items() if k in model_state}, strict=False)
    model.eval()

    print("Model loaded successfully.")
    return model, device


def infer_raw(model, device, img_rgb_u8):
    img = img_rgb_u8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor)
        if isinstance(outputs, dict) and "pred_depth" in outputs:
            if "pred_mask" in outputs:
                mask = 1 - outputs["pred_mask"]
                mask = mask > 0.5
                outputs["pred_depth"][~mask] = 1
            pred = outputs["pred_depth"][0].detach().cpu().squeeze().numpy()
        else:
            pred = outputs[0].detach().cpu().squeeze().numpy()

    return pred.astype(np.float32)


def pred_to_depth_cm_u16(pred, depth_range_m):
    # DAP inference output is treated as normalized depth in [0, 1], scaled to a metric range.
    depth_m = np.clip(pred, 0.0, 1.0) * float(depth_range_m)
    depth_cm = np.round(depth_m * 100.0)
    depth_cm = np.clip(depth_cm, 0, 65535).astype(np.uint16)
    return depth_cm


def collect_panoramas(city_panorama_dir, exts):
    exts = {e.lower() for e in exts}
    files = []
    for p in sorted(city_panorama_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def infer_city(model, device, city_dir, depth_range_m, exts, skip_existing):
    city_name = city_dir.name
    panorama_dir = city_dir / "panorama"
    depth_dir = city_dir / "mono_depth"
    if not panorama_dir.is_dir():
        print(f"[WARN] Skip {city_name}: missing panorama dir -> {panorama_dir}")
        return 0, 0

    images = collect_panoramas(panorama_dir, exts)
    if len(images) == 0:
        print(f"[WARN] Skip {city_name}: no panorama image found in {panorama_dir}")
        return 0, 0

    depth_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    for img_path in tqdm(images, desc=f"{city_name}", leave=False):
        out_path = depth_dir / (img_path.stem + ".png")
        if skip_existing and out_path.exists():
            skipped += 1
            continue

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Cannot read image: {img_path}")
            skipped += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pred = infer_raw(model, device, img_rgb)
        depth_u16 = pred_to_depth_cm_u16(pred, depth_range_m=depth_range_m)

        ok = cv2.imwrite(str(out_path), depth_u16)
        if not ok:
            print(f"[WARN] Failed to save depth png: {out_path}")
            skipped += 1
            continue

        processed += 1

    return processed, skipped


def main():
    parser = argparse.ArgumentParser(description="Estimate VIGOR panorama depth with DAP and save 16-bit cm PNGs.")
    parser.add_argument("--config", default="config/infer.yaml", help="Path to DAP inference yaml config.")
    parser.add_argument("--weights-dir", default=None, help="Override load_weights_dir in config.")
    parser.add_argument("--vigor-root", required=True, help="VIGOR root directory.")
    parser.add_argument("--gpu", default="0", help="GPU id used by CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--depth-range-m", type=float, default=100.0, help="Metric depth range for normalized predictions.")
    parser.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png"], help="Panorama image extensions.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing depth PNG files.")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.weights_dir is not None:
        config["load_weights_dir"] = args.weights_dir

    model, device = load_model(config)

    vigor_root = Path(args.vigor_root)
    if not vigor_root.is_dir():
        raise FileNotFoundError(f"VIGOR root not found: {vigor_root}")

    city_dirs = sorted([p for p in vigor_root.iterdir() if p.is_dir() and (p / "panorama").is_dir()])
    if len(city_dirs) == 0:
        raise RuntimeError(f"No city folder with panorama directory found under: {vigor_root}")

    print(f"Found {len(city_dirs)} city folders under {vigor_root}")
    print(f"Depth output format: uint16 PNG in centimeters, range mapped by depth_range_m={args.depth_range_m}")

    total_processed = 0
    total_skipped = 0
    for city_dir in city_dirs:
        processed, skipped = infer_city(
            model=model,
            device=device,
            city_dir=city_dir,
            depth_range_m=args.depth_range_m,
            exts=args.exts,
            skip_existing=args.skip_existing,
        )
        total_processed += processed
        total_skipped += skipped
        print(f"{city_dir.name}: processed={processed}, skipped={skipped}")

    print("Done.")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped:   {total_skipped}")


if __name__ == "__main__":
    main()