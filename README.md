# STG-NF — Pose Stream for the PRISM Framework

> This repository contains the **pose-based** subcomponent of
> [PRISM](https://github.com/Hadi6618/PRISM) (Pose + RGB Integration for
> Scene Monitoring), a two-stream late-fusion framework for Video Anomaly
> Detection.

STG-NF models normal human motion with **Spatio-Temporal Graph Normalizing
Flows**. Each person is represented as a graph of 17 COCO keypoints tracked
across time, embedded by a Graph Convolutional Network, and mapped to a
latent Gaussian distribution through a stack of bijective coupling layers.
At test time, the negative log-likelihood of a pose window under the learned
flow serves as the anomaly score — unusual motions (fighting, falling,
loitering) receive low likelihood.

> **Paper:** *Normalizing Flows for Human Pose Anomaly Detection*, ICCV 2023 —
> [arXiv:2211.10946](https://arxiv.org/abs/2211.10946) ·
> [original code](https://github.com/orhir/STG-NF)

---

## Role in PRISM

STG-NF is the **pose stream** of the PRISM ensemble. It detects behavioural
anomalies — fighting, falling, stealing, loitering — by modelling the
distribution of *normal human motion* and flagging frames whose kinematics
deviate.

| Component | Repository | What It Watches |
| :-- | :-- | :-- |
| **STG-NF (pose)** | **This repo** | **What people are *doing*** |
| MULDE (appearance) | [Hadi6618/MULDE](https://github.com/Hadi6618/MULDE) | What the scene *looks like* |
| Fusion pipeline | [Hadi6618/PRISM](https://github.com/Hadi6618/PRISM) | Combines both streams |

The pose extraction, training, and score export pipeline is orchestrated from
the PRISM repository's [`STG-NF.ipynb`](https://github.com/Hadi6618/PRISM/blob/main/STG-NF.ipynb)
Colab notebook, which clones this repo, runs AlphaPose to extract skeleton
keypoints, trains the normalizing flow, and exports a score pickle for the
fusion stage.

---

## Repository Layout

```
STG-NF/
├── args.py                           # CLI argument definitions and experiment directory setup
├── dataset.py                        # Dataset loaders for ShanghaiTech, Avenue, UBnormal
├── train_eval.py                     # Main entry point: training and evaluation
├── stgnf_export_scores.py            # Per-video score exporter for ensemble fusion
├── gen_data.py                       # Pose extraction script (AlphaPose + PoseFlow)
├── models/
│   ├── training.py                   # Trainer class (train/eval loop, checkpointing)
│   └── STG_NF/
│       ├── model_pose.py             # FlowNet, FlowStep, STG_NF (main model)
│       ├── stgcn.py                  # ST-GCN block with optional attention
│       ├── attention.py              # Attention modules (Dual, Triplet, bottleneck)
│       ├── modules_pose.py           # Coupling layers, actnorm, invertible 1×1 conv
│       ├── graph.py                  # Graph adjacency matrix construction
│       ├── tgcn.py                   # Temporal Graph Convolutional Network
│       └── utils.py                  # Model helpers
├── utils/
│   ├── data_utils.py                 # Pose graph transformations
│   ├── optim_init.py                 # Optimizer and scheduler factory
│   ├── pose_utils.py                 # Pose keypoint processing
│   ├── scoring_utils.py              # Frame-level scoring and AUC computation
│   └── train_utils.py                # Argument dumping, parameter counting
├── checkpoints/                      # Pretrained model weights
├── data/                             # Ground truth masks and architecture diagram
│   ├── arch.png                      # Framework overview diagram
│   ├── ShanghaiTech/gt/              # ShanghaiTech test frame masks (.npy)
│   ├── Avenue/gt/                    # Avenue test frame masks
│   └── UBnormal/gt/                  # UBnormal test frame masks
├── environment.yml                   # Conda environment specification
└── LICENSE
```

---

## Setup

### Conda Environment

```bash
git clone https://github.com/Hadi6618/STG-NF.git
cd STG-NF

conda env create -f environment.yml
conda activate STG-NF
```

Tested on Ubuntu 20.04 (Python 3.8, CUDA-capable GPU).

---

## Training & Evaluation

```bash
# Train + evaluate on ShanghaiTech (default)
python train_eval.py --dataset ShanghaiTech

# Evaluate a pretrained checkpoint
python train_eval.py --dataset ShanghaiTech --checkpoint checkpoints/ShanghaiTech_85_9.tar

# Avenue
python train_eval.py --dataset Avenue

# UBnormal (unsupervised)
python train_eval.py --dataset UBnormal --seg_len 16 --checkpoint checkpoints/UBnormal_unsupervised_71_8.tar

# UBnormal (supervised)
python train_eval.py --dataset UBnormal --seg_len 16 --R 10 --checkpoint checkpoints/UBnormal_supervised_79_2.tar
```

### Custom Pose Extraction

For a custom video dataset, poses must first be extracted using AlphaPose:

```bash
python gen_data.py \
    --alphapose_dir /path/to/AlphaPose/ \
    --dir /input/video/folder/ \
    --outdir /output/pose/folder/ \
    --video
```

Requires the FastPose-ResNet152 weights and (optionally) the YOLOX-X detector
from the [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) repository.

---

## Score Export for Ensemble Fusion

Per-video frame-level anomaly scores can be exported to a pickle file for fusion
with other models (e.g. MULDE in PRISM):

```bash
python stgnf_export_scores.py \
    --dataset ShanghaiTech \
    --checkpoint checkpoints/ShanghaiTech_85_9.tar \
    --output_pkl stgnf_scores.pkl
```

The exported pickle contains:

| Key | Type | Description |
| :-- | :-- | :-- |
| `scores_by_video` | `dict[str, dict]` | Per-video `frame_indices`, `anomaly_scores`, `labels` arrays |
| `micro_auc` | `float` | Standalone frame-level Micro AUC |

---

## Attention Mechanisms

ST-GCN blocks optionally support **learnable attention** modules inspired by
[Abnormal Human Behaviour Detection using Normalising Flows and Attention
Mechanisms](https://github.com/mazhouda/Abnormal-Human-Behaviour-Detection-using-Normalising-Flows-and-Attention-Mechanisms).
Attention is applied **after** the standard GCN → TCN → ReLU pass inside each
`st_gcn` block and is controlled via CLI flags.

### Attention Types

| Type | Streams | Pooling | Description |
| :-- | :-- | :-- | :-- |
| `none` (default) | — | — | Original STG-NF — no attention |
| `skeleton` | Skeleton | maxpool | Attends to spatial joint relationships |
| `frame` | Frame | maxpool | Attends to temporal dynamics across frames |
| `dual` | Skeleton + Frame | maxpool | DAM: averaged skeleton & frame attention + residual |
| `triplet` | Skeleton + Frame + Channel | zpool | Triplet: averaged three-stream attention + residual |

### CLI Arguments

| Flag | Default | Description |
| :-- | :-- | :-- |
| `--attention` | `none` | Attention variant (`none`, `skeleton`, `frame`, `dual`, `triplet`) |
| `--n_heads` | `1` | Number of parallel attention heads |
| `--n_mecatt` | `1` | Sequential attention applications per `st_gcn` forward pass |
| `--n_mecatt_inside` | `1` | Inner iterations inside each attention module |
| `--freeze_attention` | off | Initialize attention once and never update (fixed random projection) |
| `--attention_lr_mult` | `1.0` | LR multiplier for attention params |
| `--attention_wd_mult` | `1.0` | Weight decay multiplier for attention params |
| `--attention_proj_type` | `full` | `full` or `bottleneck` projection |
| `--attention_bottleneck_dim` | `64` | Bottleneck width (only with `bottleneck` projection) |

### Recommended Configurations

The default `full` projection adds **~27M trainable params** (dual) or **~40M**
(triplet) on top of the **1,236-param** base STG-NF, which causes severe
overfitting on small datasets. Three strategies are provided:

| Strategy | Flags | Use When |
| :-- | :-- | :-- |
| **Freeze** | `--freeze_attention` | Best for matching the reference repo's ~83% on small datasets |
| **Regularize** | `--attention_lr_mult 0.1 --attention_wd_mult 5` | Want attention to learn but converge slowly |
| **Reduce** | `--attention_proj_type bottleneck --attention_bottleneck_dim 64` | Lets attention learn without the 40M-param explosion |

```bash
# Frozen Dual Attention (matches reference behaviour)
python train_eval.py --dataset Avenue --attention dual --freeze_attention

# Bottleneck Triplet Attention on ShanghaiTech
python train_eval.py --dataset ShanghaiTech --attention triplet \
    --attention_proj_type bottleneck --attention_bottleneck_dim 64
```

> **Important:** The `--attention_proj_type`, `--attention_bottleneck_dim`, and
> `--freeze_attention` flags **must match** between training and evaluation,
> otherwise the checkpoint will not load.

---

## Results (within PRISM)

| Method | Stream | ShanghaiTech (Micro AUC) | Avenue (Micro AUC) |
| :-- | :-- | ---: | ---: |
| **STG-NF** | Pose | **84%** | **57.0%** |
| MULDE | Appearance | 79.7% | 81.4% |
| PRISM (fusion) | Both | 89.9% | 82.8% |

STG-NF is the stronger stream on ShanghaiTech (84% vs. MULDE's 79.7%) where
behavioural anomalies are dominant, but underperforms on Avenue (57%) where
most anomalies are contextual (object-based). The fusion in PRISM leverages
the complementary strengths of both streams.

---

## Citation

```bibtex
@InProceedings{Hirschorn_2023_ICCV,
    author    = {Hirschorn, Or and Avidan, Shai},
    title     = {Normalizing Flows for Human Pose Anomaly Detection},
    booktitle = {Proceedings of the IEEE/CVF International Conference on
                 Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13545-13554}
}

@misc{prism2026,
  title   = {PRISM: Pose + RGB Integration for Scene Monitoring},
  author  = {Hadi},
  year    = {2026},
  note    = {Late fusion of STG-NF and MULDE for video anomaly detection},
  url     = {https://github.com/Hadi6618/PRISM}
}
```

## Acknowledgments

This code is based on:

- [Graph Embedded Pose Clustering for Anomaly Detection](https://github.com/amirmk89/gepc)
- [Glow](https://github.com/y0ast/Glow-PyTorch)

## License

This code is distributed under a [Creative Commons LICENSE](LICENSE).

Note that our code depends on other libraries and uses datasets that each have
their own respective licenses that must also be followed.
