# Normalizing Flows for Human Pose Anomaly Detection [ICCV 2023]
[![arXiv](https://img.shields.io/badge/arXiv-<2211.10946>-<COLOR>.svg)](https://arxiv.org/abs/2211.10946)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/normalizing-flows-for-human-pose-anomaly/anomaly-detection-on-shanghaitech)](https://paperswithcode.com/sota/anomaly-detection-on-shanghaitech?p=normalizing-flows-for-human-pose-anomaly)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/normalizing-flows-for-human-pose-anomaly/anomaly-detection-on-ubnormal)](https://paperswithcode.com/sota/anomaly-detection-on-ubnormal?p=normalizing-flows-for-human-pose-anomaly)



The official PyTorch implementation of the paper [**"Normalizing Flows for Human Pose Anomaly Detection"**](https://arxiv.org/abs/2211.10946).


![Framework_Overview](data/arch.png)

## Citation
If you find this useful, please cite this work as follows:
```
@InProceedings{Hirschorn_2023_ICCV,
    author    = {Hirschorn, Or and Avidan, Shai},
    title     = {Normalizing Flows for Human Pose Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13545-13554}
}
```

## Getting Started

This code was tested on `Ubuntu 20.04.4 LTS` and requires:
* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### Setup Conda Environment:
```
git clone https://github.com/orhir/STG-NF
cd STG-NF

# Conda environment setup
conda env create -f environment.yml
conda activate STG-NF
```

### Directory Structure
```
.
├── checkpoints
├── data
│   ├── ShanghaiTech
│   │   ├── gt
│   │   │   └── test_frame_mask
│   │   └── pose
│   │       ├── test
│   │       └── train
│   └── UBnormal
│       ├── gt
│       ├── pose
│       │   ├── abnormal_train
│       │   ├── test
│       │   ├── train
│       │   └── validation
│       └── videos
├── models
│   └── STG_NF
│       ├── attention.py      # Attention mechanisms (Dual, Triplet, etc.)
│       ├── stgcn.py          # ST-GCN block with optional attention
│       ├── model_pose.py      # FlowNet, FlowStep, STG_NF model
│       └── ...
├── stgnf_export_scores.py     # Per-video score exporter for ensemble fusion
└── utils

```

### Data Directory
Data folder, including extracted poses and GT, can be downloaded using the [link](https://drive.google.com/file/d/1o9h3Kh6zovW4FIHpNBGnYIRSbGCu-qPt/view?usp=sharing).

The data directory holds pose graphs and ground truth vectors for the datasets.
A path for the directory may be configured using the arguments:

    --vid_path_train
    --vid_path_test
    --pose_path_train
    --pose_path_train_abnormal
    --pose_path_test

### Custom Dataset
We provide a script for creating JSON files in the accepted format using [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose).
Please download into pretrained_models folders [fast_421_res152_256x192.pth](https://drive.google.com/open?id=1kfyedqyn8exjbbNmYq8XGd2EooQjPtF9)
It is also recommended to use the YOLOX-X detector, which can be downloaded from the AlphaPose repository.
Use the flag --video for video folder, otherwise assumes a folder of JPG/PNG images for each video.
    python gen_data.py --alphapose_dir /path/to/AlphaPoseFloder/ --dir /input/dir/ --outdir /output/dir/ [--video]

## Training/Testing
Training and Evaluating is run using:
```
python train_eval.py --dataset [ShanghaiTech|UBnormal|Avenue]
```

Evaluation of our pretrained model can be done using:

ShanghaiTech/ShanghaiTech-HR:
```
python train_eval.py --dataset [ShanghaiTech/ShanghaiTech-HR] --checkpoint checkpoints/ShanghaiTech_85_9.tar
```
Unsupervised UBnormal
```
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar 
```
Supervised UBnormal
```
python train_eval.py --dataset UBnormal --seg_len 16 --R 10 --checkpoint checkpoints/UBnormal_supervised_79_2.tar
```

## Attention Mechanisms

ST-GCN blocks optionally support **learnable attention** modules inspired by
[Abnormal Human Behaviour Detection using Normalising Flows and Attention Mechanisms](https://github.com/mazhouda/Abnormal-Human-Behaviour-Detection-using-Normalising-Flows-and-Attention-Mechanisms).
Attention is applied **after** the standard GCN → TCN → ReLU pass inside each
`st_gcn` block and is controlled via four CLI flags.

### Attention Types

| Type | Streams | Pooling | Description |
|---|---|---|---|
| `none` (default) | — | — | Original STG-NF — no attention |
| `skeleton` | Skeleton | maxpool | Attends to spatial joint relationships |
| `frame` | Frame | maxpool | Attends to temporal dynamics across frames |
| `dual` | Skeleton + Frame | maxpool | DAM: averaged skeleton & frame attention + residual |
| `triplet` | Skeleton + Frame + Channel | zpool | Triplet: averaged three-stream attention + residual |

### CLI Arguments

| Flag | Default | Choices | Description |
|---|---|---|---|
| `--attention` | `none` | `none`, `skeleton`, `frame`, `dual`, `triplet` | Attention variant to use in every `st_gcn` block |
| `--n_heads` | `1` | int | Number of parallel attention heads |
| `--n_mecatt` | `1` | int | Number of sequential attention applications per `st_gcn` forward pass |
| `--n_mecatt_inside` | `1` | int | Inner iterations inside each attention module |
| `--freeze_attention` | off | flag | Initialize attention once and never update it (fixed random projection). Reproduces the reference repo's accidental behaviour. |
| `--attention_lr_mult` | `1.0` | float | Multiplier on the base LR for attention params (e.g. `0.1` = 10× slower). |
| `--attention_wd_mult` | `1.0` | float | Multiplier on the base weight decay for attention params. |
| `--attention_proj_type` | `full` | `full`, `bottleneck` | `full` = `Linear(C·T·V, C·T·V)` (reference, huge); `bottleneck` = low-rank projection. |
| `--attention_bottleneck_dim` | `64` | int | Bottleneck width (only when `--attention_proj_type bottleneck`). |

### Important: Overfitting on Small Datasets

The default `full` projection adds **~27M trainable params** (dual) or **~40M** (triplet) on top of the **1,236-param** base STG-NF. Training all those params on ShanghaiTech (~330 clips) or Avenue (16 clips) causes severe overfitting and **drops AUC below the no-attention baseline**.

This differs from the reference repo, whose attention params are *accidentally never trained* (created fresh on every forward pass), so they act as a harmless fixed random projection. Three strategies are provided to handle this:

| Strategy | Flag(s) | When to use |
|---|---|---|
| **Freeze** attention (fixed random projection) | `--freeze_attention` | Best for matching the reference repo's ~83% numbers on small datasets. Attention is initialized once and never updated. |
| **Regularize** attention (slower LR, higher WD) | `--attention_lr_mult 0.1 --attention_wd_mult 5` | When you want attention to *learn* but converge slowly. Tune the multipliers. |
| **Reduce** attention capacity (bottleneck) | `--attention_proj_type bottleneck --attention_bottleneck_dim 64` | Drops dual's params from 27M to ~2.7M (90% smaller). Lets attention actually learn without overfitting. |

These can be combined, e.g. bottleneck + low LR:
```
python train_eval.py --dataset ShanghaiTech --attention dual \
    --attention_proj_type bottleneck --attention_bottleneck_dim 64 \
    --attention_lr_mult 0.1 --attention_wd_mult 5
```

### Recommended Configurations

| Dataset | Attention | Strategy | Rationale |
|---|---|---|---|
| **Avenue** (small, 16 train clips) | `dual` | `--freeze_attention` | Motion-based anomalies; freezing prevents overfitting on tiny dataset |
| **ShanghaiTech** (~330 train clips) | `triplet` | `--attention_proj_type bottleneck --attention_bottleneck_dim 64` | Enough data to learn, but bottleneck prevents the 40M-param explosion |

### Examples

Train with frozen Dual Attention (matches reference behaviour):
```
python train_eval.py --dataset Avenue --attention dual --freeze_attention
```

Train with bottleneck Triplet Attention on ShanghaiTech:
```
python train_eval.py --dataset ShanghaiTech --attention triplet \
    --attention_proj_type bottleneck --attention_bottleneck_dim 64
```

Evaluate a Triplet Attention checkpoint on ShanghaiTech:
```
python train_eval.py --dataset ShanghaiTech --attention triplet \
    --attention_proj_type bottleneck --attention_bottleneck_dim 64 \
    --checkpoint checkpoints/ShanghaiTech_triplet.tar
```

> **Note**: The `--attention_proj_type` / `--attention_bottleneck_dim` / `--freeze_attention` flags **must match** between training and evaluation, otherwise the checkpoint will not load.

### Architecture Details

- **Implementation**: `models/STG_NF/attention.py` — attention is implemented as proper `nn.Module` subclasses (unlike the reference repo), so parameters are registered, trainable, and saved in checkpoints.
- **Integration**: `models/STG_NF/stgcn.py` — each `st_gcn` block optionally holds an attention submodule; args are threaded from `STG_NF → FlowNet → FlowStep → get_stgcn → st_gcn`.
- **Backward compatibility**: Using `--attention none` (default) produces a model with identical architecture and checkpoint keys to the original STG-NF.
- **Tensor shape**: `(N, C, T, V)` where N=batch, C=channels, T=temporal length, V=joints.

## Score Export for Ensemble Fusion

Per-video frame-level anomaly scores can be exported to a pickle file for fusion
with other models (e.g. MULDE):

```
python stgnf_export_scores.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar --output_pkl stgnf_scores.pkl
```

The exported pickle contains `scores_by_video` (dict keyed by video ID with
`frame_indices`, `anomaly_scores`, and `labels` arrays) and the model's standalone
micro AUC.

## Acknowledgments
Our code is based on code from:
- [Graph Embedded Pose Clustering for Anomaly Detection](https://github.com/amirmk89/gepc)
- [Glow](https://github.com/y0ast/Glow-PyTorch)

## License
This code is distributed under a [Creative Commons LICENSE](LICENSE).

Note that our code depends on other libraries and uses datasets that each have their own respective licenses that must also be followed.
