# DB-ACENet
Official implementation of the paper "DB-ACENet: Dual-Branch Adaptive Compensation Enhanced Network for 3D Human Pose Estimation"


#Environment
The code is conducted under the following environment

* Ubuntu 18.04
* Python 3.8.17
* PyTorch 1.11.0
* CUDA 11.3

bash
> pip install -r requirements.txt 

or

> conda env create -f requirement.yml

# Dataset
Human3.6M + MPI-INF-3DHP

## Human3.6M
### Preprocessing
1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:
```text
python h36m.py  --n-frames 243
```

## MPI-INF-3DHP
### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

# Download
The intermediate feature representations of MICB branch is available [here](https://drive.google.com/drive/folders/1MhWKZSi0xiQ8OA_O2Z1kJjWA0nANn30w?usp=sharing).

Download the checkpoints of best epoch and pre-train model [here](https://drive.google.com/drive/folders/1MhWKZSi0xiQ8OA_O2Z1kJjWA0nANn30w?usp=sharing).

All the preprocessed data can be downloaded [here](https://pan.baidu.com/s/1GAJLcMHsbLmOCwmsqOtTGA?pwd=akiq).

# Training from scratch
DB-ACENet:
> python train_DBACENet.py --new-checkpoint <new_save_path>

# Training from pre-train model
DB-ACENet:
> python train_DBACENet.py --new-checkpoint <new_save_path> --checkpoint <path_of_pre_train model> --resume

# Evaluation
DB-ACENet:
> python train_DBACENet.py --checkpoint <path_of_pre_train model> --resume --eval-only

## Acknowledgement
Our code refers to the following repositories:

- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [P-STMO](https://github.com/paTRICK-swk/P-STMO)
- [MHFormer](https://github.com/Vegetebird/MHFormer)