"""
Train a compact BirdCLEF sequence head on precomputed acoustic features.

This is the portfolio/research version of the training entrypoint used to
document the competition solution. It assumes the expensive feature extraction
step has already produced an NPZ file with:

  X: float32 array shaped [n_soundscapes, n_windows, n_features]
  y: float32 multi-label targets shaped [n_soundscapes, n_classes]

In the Kaggle notebooks, X was built from Perch v2 logits/embeddings, ecological
priors, PCA features and optional SED side-channel scores. This script trains the
lightweight temporal correction head that can sit on top of those features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    features: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    hidden_dim: int
    folds: int
    seed: int


class BirdSequenceHead(nn.Module):
    """Small temporal head for 12-window BirdCLEF soundscape features."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int) -> None:
        super().__init__()
        self.temporal = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.temporal(x)
        weights = torch.softmax(self.attention(sequence), dim=1)
        pooled = (sequence * weights).sum(dim=1)
        return self.head(pooled)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    x = payload["X"].astype("float32")
    y = payload["y"].astype("float32")
    if "groups" in payload:
        groups = payload["groups"]
    else:
        groups = np.arange(len(x))
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape [n, windows, features], got {x.shape}")
    if y.ndim != 2:
        raise ValueError(f"Expected y with shape [n, classes], got {y.shape}")
    return x, y, groups


def average_precision_proxy(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    """Fast validation proxy used for local model selection."""
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    truth = y_true.detach().cpu().numpy()
    order = np.argsort(-probs, axis=1)
    scores = []
    for row, labels in zip(order, truth):
        positives = set(np.flatnonzero(labels > 0.5).tolist())
        if not positives:
            continue
        hits = 0
        precision_sum = 0.0
        for rank, cls_idx in enumerate(row, start=1):
            if cls_idx in positives:
                hits += 1
                precision_sum += hits / rank
            if hits == len(positives):
                break
        scores.append(precision_sum / max(1, len(positives)))
    return float(np.mean(scores)) if scores else 0.0


def run_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    cfg: TrainConfig,
    fold: int,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdSequenceHead(x.shape[-1], cfg.hidden_dim, y.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx]))
    valid_x = torch.from_numpy(x[valid_idx]).to(device)
    valid_y = torch.from_numpy(y[valid_idx]).to(device)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    best_score = -1.0
    best_loss = float("inf")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            valid_logits = model(valid_x)
            valid_loss = float(criterion(valid_logits, valid_y).detach().cpu())
            valid_score = average_precision_proxy(valid_y, valid_logits)

        if valid_score > best_score:
            best_score = valid_score
            best_loss = valid_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": x.shape[-1],
                    "hidden_dim": cfg.hidden_dim,
                    "n_classes": y.shape[-1],
                    "fold": fold,
                    "config": asdict(cfg),
                },
                output_dir / f"bird_sequence_head_fold{fold}.pt",
            )
        print(
            json.dumps(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "valid_loss": valid_loss,
                    "valid_map_proxy": valid_score,
                }
            )
        )

    return {"fold": fold, "best_valid_map_proxy": best_score, "best_valid_loss": best_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="NPZ with X, y and optional groups arrays.")
    parser.add_argument("--output-dir", default="artifacts/birdclef_sequence_head")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(**vars(args))
    set_seed(cfg.seed)
    x, y, groups = load_npz(Path(cfg.features))

    splitter = GroupKFold(n_splits=cfg.folds)
    metrics = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        metrics.append(run_fold(x, y, train_idx, valid_idx, cfg, fold))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
