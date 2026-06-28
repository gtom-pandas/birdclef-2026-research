"""
BirdCLEF inference/blending entrypoint for portfolio-grade reproducibility.

The Kaggle notebooks used Perch v2, SED models, ecological priors and temporal
heads. This script covers the final lightweight stage: load precomputed test
features, average fold checkpoints, optionally fuse priors, and write a Kaggle
submission-style CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class BirdSequenceHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.temporal = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(0.0), nn.Linear(hidden_dim * 2, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.temporal(x)
        weights = torch.softmax(self.attention(sequence), dim=1)
        return self.head((sequence * weights).sum(dim=1))


def load_checkpoints(paths: list[Path], device: torch.device) -> list[nn.Module]:
    models = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device)
        model = BirdSequenceHead(
            checkpoint["input_dim"],
            checkpoint["hidden_dim"],
            checkpoint["n_classes"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        models.append(model)
    return models


def apply_rank_blend(probs: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return probs
    ranks = np.argsort(np.argsort(probs, axis=1), axis=1).astype("float32")
    ranks = ranks / np.maximum(1, probs.shape[1] - 1)
    return (1.0 - weight) * probs + weight * ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="NPZ with X and row_id arrays.")
    parser.add_argument("--classes", required=True, help="Text file with one target class per line.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Fold checkpoint path. Can be repeated.")
    parser.add_argument("--priors", default=None, help="Optional CSV with row_id plus class prior columns.")
    parser.add_argument("--prior-weight", type=float, default=0.08)
    parser.add_argument("--rank-weight", type=float, default=0.05)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    payload = np.load(args.features, allow_pickle=True)
    x = payload["X"].astype("float32")
    row_ids = payload["row_id"].astype(str)
    classes = [line.strip() for line in Path(args.classes).read_text(encoding="utf-8").splitlines() if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_checkpoints([Path(p) for p in args.checkpoint], device)

    with torch.no_grad():
        tensor_x = torch.from_numpy(x).to(device)
        fold_probs = [torch.sigmoid(model(tensor_x)).detach().cpu().numpy() for model in models]
    probs = np.mean(fold_probs, axis=0)

    if args.priors:
        priors = pd.read_csv(args.priors).set_index("row_id").reindex(row_ids)
        prior_values = priors[classes].fillna(0.0).to_numpy(dtype="float32")
        probs = (1.0 - args.prior_weight) * probs + args.prior_weight * prior_values

    probs = apply_rank_blend(probs, args.rank_weight)
    probs = np.clip(probs, 0.0, 1.0)

    submission = pd.DataFrame(probs, columns=classes)
    submission.insert(0, "row_id", row_ids)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with shape {submission.shape}")


if __name__ == "__main__":
    main()
