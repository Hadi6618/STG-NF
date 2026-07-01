# STG-NF frame-level score exporter for the STG-NF + MULDE ensemble.
# Mirrors the test-eval flow of train_eval.py but also serialises per-clip,
# per-frame anomaly scores into stgnf_scores.pkl.

import argparse
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch

from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from dataset import shanghaitech_hr_skip
from utils.scoring_utils import get_clip_score, gt_root_for_dataset, score_dataset
from utils.train_utils import init_model_params


def _build_argparser():
    parser = init_parser()
    parser.add_argument(
        "--output_pkl",
        type=str,
        required=True,
        help="Destination path for stgnf_scores.pkl",
    )
    parser.add_argument(
        "--skip_smoothing",
        action="store_true",
        help="Skip the temporal Gaussian smoothing used by score_dataset.",
    )
    return parser


def _per_clip_per_frame_scores(normality_scores, dataset, args):
    metadata = dataset.metadata
    metadata_np = np.array(metadata)
    per_frame_scores_root = gt_root_for_dataset(args.dataset)
    clip_list = sorted(
        fn for fn in os.listdir(per_frame_scores_root) if fn.endswith(".npy")
    )

    out = OrderedDict()
    for clip in clip_list:
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split(".")[0].split("_")]
        if shanghaitech_hr_skip((args.dataset == "ShanghaiTech-HR"), scene_id, clip_id):
            continue
        clip_gt, clip_score = get_clip_score(
            normality_scores,
            clip,
            metadata_np,
            metadata,
            per_frame_scores_root,
            args,
        )
        if clip_score is None:
            continue
        video_id = f"{scene_id:02d}_{clip_id:04d}"
        frame_indices = np.arange(clip_score.shape[0], dtype=np.int64)
        out[video_id] = {
            "frame_indices": frame_indices,
            "anomaly_scores": clip_score.astype(np.float32),
            "labels": np.asarray(clip_gt, dtype=np.uint8),
        }
    return out


def main():
    parser = _build_argparser()
    args = parser.parse_args()
    args, _ = init_sub_args(args)

    pretrained = vars(args).get("checkpoint", None)
    if pretrained is None:
        raise SystemExit("--checkpoint is required for score export")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if args.seed != 999:
        torch.manual_seed(args.seed)
        np.random.seed(0)

    dataset, loader = get_dataset_and_loader(
        args,
        trans_list=trans_list,
        only_test=True,
    )
    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    trainer = Trainer(
        args,
        model,
        loader["train"],
        loader["test"],
        optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
        scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs),
    )
    trainer.load_checkpoint(pretrained)
    normality_scores = trainer.test()

    if args.skip_smoothing:
        scores_by_video = _per_clip_per_frame_scores(normality_scores, dataset["test"], args)
        micro_auc = None
    else:
        auc, scores_np = score_dataset(normality_scores, dataset["test"].metadata, args=args)
        micro_auc = float(auc)
        scores_by_video = _per_clip_per_frame_scores(normality_scores, dataset["test"], args)
        finite = np.isfinite(scores_np)
        finite_max = scores_np[finite].max() if finite.any() else 0.0
        finite_min = scores_np[finite].min() if finite.any() else 0.0
        offset = 0
        for video_id, payload in scores_by_video.items():
            n = payload["frame_indices"].shape[0]
            smoothed = scores_np[offset : offset + n].astype(np.float32).copy()
            non_finite = ~np.isfinite(smoothed)
            if non_finite.any():
                first_non_finite = smoothed[non_finite][0]
                smoothed[non_finite] = finite_max if first_non_finite > 0 else finite_min
            payload["anomaly_scores"] = smoothed
            offset += n

    output = {
        "scores_by_video": scores_by_video,
        "micro_auc": micro_auc,
        "num_videos": len(scores_by_video),
    }
    os.makedirs(os.path.dirname(args.output_pkl) or ".", exist_ok=True)
    with open(args.output_pkl, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved STG-NF frame scores for {len(scores_by_video)} videos to {args.output_pkl}")
    if micro_auc is not None:
        print(f"Single-model Micro AUC (smoothed): {micro_auc * 100:.4f}%")


if __name__ == "__main__":
    main()
