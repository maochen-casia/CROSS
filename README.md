# More Than Where You Are: Learning Semantics, Structure, and Geometry from Cross-View Consistency

<p align="center">
  <img src="figures/CROSS.png" alt="CROSS overview" width="90%">
</p>

## Setup

Install the project dependencies used by this repository, then install the external repositories used for image encoders and monocular depth:

- [DINOv2](https://github.com/facebookresearch/dinov2)
- DINOv3
- [Depth-Anything-3](https://github.com/DepthAnything/Depth-Anything-3)
- DAP

Download the corresponding pretrained weights for DINOv2, DINOv3, Depth-Anything-3, and DAP.

Set the local paths before training:

```bash
export CROSS_DATA_ROOT=<data-root>
export DINOV2_REPO=<DINOv2-repo>
export DINOV2_VITL14_WEIGHTS=<dinov2_vitl14_pretrain.pth>
export DINOV3_REPO=<DINOv3-repo>
export DINOV3_VITL16_WEIGHTS=<dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth>
```

`CROSS_DATA_ROOT` must contain the processed dataset folders:

```text
$CROSS_DATA_ROOT/
  KITTI/
  VIGOR/
```

You can also set `data.root` directly in a config file instead of using `CROSS_DATA_ROOT`.

## Data Processing

### KITTI

Prepare the KITTI raw data and satellite maps under:

```text
$CROSS_DATA_ROOT/KITTI/
  raw_data/
  satmap/
```

Generate monocular depth for KITTI with Depth-Anything-3:

```bash
python data/da3_kitti.py \
  --root-dir "$CROSS_DATA_ROOT/KITTI/raw_data" \
  --model-id depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
  --batch-size 4
```

This writes `mono_depth/` PNG files inside each KITTI raw drive.

Then build the KITTI metadata files:

```bash
cd data/KITTI
python process_data.py --kitti_root "$CROSS_DATA_ROOT/KITTI"
cd ../..
```

This creates:

```text
$CROSS_DATA_ROOT/KITTI/train_data.pth
$CROSS_DATA_ROOT/KITTI/val_data.pth
$CROSS_DATA_ROOT/KITTI/test_data.pth
```

### VIGOR

Prepare VIGOR under:

```text
$CROSS_DATA_ROOT/VIGOR/
  NewYork/
  Seattle/
  SanFrancisco/
  Chicago/
  splits__corrected/
```

Copy the VIGOR depth inference script into the DAP repository and run it there:

```bash
cp data/infer_vigor_depth.py <DAP-repo>/test/
cd <DAP-repo>
python test/infer_vigor_depth.py \
  --config config/infer.yaml \
  --weights-dir <DAP-weights-dir> \
  --vigor-root "$CROSS_DATA_ROOT/VIGOR" \
  --gpu 0 \
  --skip-existing
```

This writes `mono_depth/` PNG files inside each VIGOR city folder.

Then build the VIGOR metadata files:

```bash
cd data/VIGOR
python process_data.py --vigor_root "$CROSS_DATA_ROOT/VIGOR"
cd ../..
```

This creates same-area and cross-area metadata:

```text
$CROSS_DATA_ROOT/VIGOR/same_area/train_data.pth
$CROSS_DATA_ROOT/VIGOR/same_area/val_data.pth
$CROSS_DATA_ROOT/VIGOR/same_area/test_data.pth
$CROSS_DATA_ROOT/VIGOR/cross_area/train_data.pth
$CROSS_DATA_ROOT/VIGOR/cross_area/val_data.pth
$CROSS_DATA_ROOT/VIGOR/cross_area/test_data.pth
```

## Training

Start training with the matching config:

```bash
python train.py --config configs/vigor_same_full_dinov2_config.yaml
python train.py --config configs/vigor_cross_full_dinov2_config.yaml
python train.py --config configs/kitti_full_dinov2_config.yaml
```

Weak-supervision and reduced-DoF configs are also provided:

```bash
python train.py --config configs/vigor_same_2dof_weak_dinov2_config.yaml
python train.py --config configs/vigor_same_3dof_weak_dinov2_config.yaml
python train.py --config configs/vigor_cross_2dof_weak_dinov2_config.yaml
python train.py --config configs/vigor_cross_3dof_weak_dinov2_config.yaml
python train.py --config configs/kitti_2dof_weak_dinov2_config.yaml
python train.py --config configs/kitti_3dof_weak_dinov2_config.yaml
```

Checkpoints and logs are saved under `checkpoints/<exp_name>/`.

## Path Configuration

The repository does not require editing hard-coded machine paths. Configure local paths with environment variables or config values:

- `CROSS_DATA_ROOT`: directory containing `KITTI/` and `VIGOR/`
- `DINOV2_REPO`: local DINOv2 repository
- `DINOV2_VITL14_WEIGHTS`: DINOv2 ViT-L/14 pretrained weights
- `DINOV3_REPO`: local DINOv3 repository
- `DINOV3_VITL16_WEIGHTS`: DINOv3 ViT-L/16 pretrained weights

If you use DINOv3 configs, set `model.dino_model_name` accordingly and make sure the DINOv3 paths above are exported.
