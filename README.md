
# Neural Compression of 3D Meshes using Sparse Implicit Representation

Paper (OpenReview): https://openreview.net/forum?id=AzsN1qdLwv

## Environment Requirements
- Python 3.8+
- PyTorch 1.10+
- Other dependencies: MinkowskwEngine, open3d, trimesh, ...

## Dataset Preparation
- Supported format: `.obj`
- Place your 3D mesh datasets under `testdata/`
- Sample datasets for quick testing (unzip first):
    - `testdata/MPEG_samples/`
    - `testdata/ShapeNet_samples/`
    - `testdata/Mixed_samples/`
    

## Training/Testing


### Parameters Explanation
- --voxel_grid_res: Resolution of voxel grid, options [128, 192, 256, 384]
- --train_path: Path to training dataset
- --test_path: Path to testing dataset
- --init_ckpt: Path to initial checkpoint for fine-tuning or testing
- --log_path: Directory to save logs and outputs
- --only_test: Set to 1 for testing mode (no training), 0 for training mode (default)



This project uses the external `tmc3` tool for coordinate compression:
- Install `tmc3`: https://github.com/MPEGGroup/mpeg-pcc-tmc13
- A prebuilt Linux binary is provided; make it executable:
```bash
chmod -R 777 ./*
```

Pretrained checkpoints:
- `ckpts/model_r1.pt`
- `ckpts/model_r2.pt`

Bitrate can be adjusted via the checkpoint (different RD weights) and `--voxel_grid_res` (e.g., 192/256/384/512).



Run testing or training:

```bash
python train.py --voxel_grid_res=256 \
    --train_path='traindata/' --test_path='testdata/MPEG_samples' \
    --init_ckpt='ckpts/model_r1.pt' \
    --log_path='output/test/' \
    --only_test=1
```


## Acknowledgements

Inspired by:
- NeCGS: https://github.com/rsy6318/NeCGS
- PCGCv2: https://github.com/NJUVISION/PCGCv2
