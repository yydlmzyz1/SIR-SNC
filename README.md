
# Neural Compression of 3D Meshes using Sparse Implicit Representation


## Environment Requirements
- Python 3.8+
- PyTorch 1.10+
- Other dependencies: MinkowskwEngine, open3d, trimesh, ...

## Dataset Preparation
- Place your 3D mesh datasets in the 'data' folder
- Supported formats: .obj


## Training/Testing


### Parameters Explanation
- --voxel_grid_res: Resolution of voxel grid, options [128, 192, 256, 384]
- --train_path: Path to training dataset
- --test_path: Path to testing dataset
- --init_ckpt: Path to initial checkpoint for fine-tuning or testing
- --log_path: Directory to save logs and outputs
- --only_test: Set to 1 for testing mode (no training), 0 for training mode (default)

```bash
chmod -R 777 ./*
```


```bash
python train.py --voxel_grid_res=256 \
    --train_path='data/' --test_path='data/' \
    --init_ckpt='ckpts/model0001.pt' \
    --log_path='output/test/' \
    --only_test=1
```

