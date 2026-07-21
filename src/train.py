"""Train the two lightweight sequence heads used inside EoS.8 Model_51.

This module is a cleaned extraction of the executed ``Model_51`` cell from the
author's private Kaggle notebook ``ttgrcgrc/birdclef-2026-eos-8``.  It does not
train Perch v2 or the distilled EfficientNet SED model: those were attached
pretrained dependencies.  It trains LightProtoSSM and the subsequent
ResidualSSM on cached Perch embeddings and 234-class window scores.

Expected NPZ arrays
-------------------
embeddings: [n_files * 12, 1536]
perch_logits: [n_files * 12, n_classes]
labels: [n_files * 12, n_classes]
site_ids: [n_files]
hour_ids: [n_files]

The notebook trained these heads during submission inference.  Saving them here
makes that historically executed training stage inspectable and reusable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_WINDOWS = 12
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class SelectiveSSM(nn.Module):
    """Compact selective state-space block from the Model_51 notebook."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        a = a.unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, width = x.shape
        x_ssm, _gate = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :steps].transpose(1, 2)
        x_conv = F.silu(x_conv)
        delta = F.softplus(self.dt_proj(x_conv))
        a = -torch.exp(self.A_log)
        b = self.B_proj(x_conv)
        c = self.C_proj(x_conv)
        state = torch.zeros(batch_size, width, self.d_state, device=x.device)
        outputs = []
        for step in range(steps):
            discrete_a = torch.exp(a[None] * delta[:, step, :, None])
            discrete_b = delta[:, step, :, None] * b[:, step, None, :]
            state = state * discrete_a + x[:, step, :, None] * discrete_b
            outputs.append((state * c[:, step, None, :]).sum(-1))
        return torch.stack(outputs, dim=1) + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    """Bidirectional SSM with metadata injection and class prototypes."""

    def __init__(
        self,
        d_input: int = 1536,
        d_model: int = 128,
        d_state: int = 16,
        n_classes: int = 234,
        n_windows: int = N_WINDOWS,
        dropout: float = 0.15,
        n_sites: int = 20,
        meta_dim: int = 16,
        use_cross_attn: bool = True,
        cross_attn_heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.ssm_fwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop = nn.Dropout(dropout)
        if use_cross_attn:
            self.cross_attn = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        d_model, cross_attn_heads, dropout=dropout, batch_first=True
                    )
                    for _ in range(2)
                ]
            )
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            projected = self.input_proj(embeddings)
            for class_index in range(self.n_classes):
                mask = labels[:, class_index] > 0.5
                if mask.sum() > 0:
                    mean = projected[mask].mean(0)
                    self.prototypes.data[class_index] = F.normalize(mean, dim=0)

    def forward(
        self,
        embeddings: torch.Tensor,
        perch_logits: torch.Tensor | None = None,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, steps, _ = embeddings.shape
        hidden = self.input_proj(embeddings) + self.pos_enc[:, :steps, :]
        if site_ids is not None and hours is not None:
            metadata = torch.cat([self.site_emb(site_ids), self.hour_emb(hours)], dim=-1)
            hidden = hidden + self.meta_proj(metadata)[:, None, :]
        for index, (forward, backward, merge, norm) in enumerate(
            zip(self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)
        ):
            residual = hidden
            hidden_fwd = forward(hidden)
            hidden_bwd = backward(hidden.flip(1)).flip(1)
            hidden = self.drop(merge(torch.cat([hidden_fwd, hidden_bwd], dim=-1)))
            hidden = norm(hidden + residual)
            if self.use_cross_attn:
                attended, _ = self.cross_attn[index](hidden, hidden, hidden)
                hidden = self.cross_norm[index](hidden + attended)
        similarities = torch.matmul(
            F.normalize(hidden, dim=-1), F.normalize(self.prototypes, dim=-1).T
        )
        similarities = (
            similarities * F.softplus(self.proto_temp) + self.class_bias[None, None, :]
        )
        if perch_logits is None:
            return similarities
        alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
        return alpha * similarities + (1.0 - alpha) * perch_logits


class ResidualSSM(nn.Module):
    """Second-pass model that predicts corrections to first-pass logits."""

    def __init__(
        self,
        d_input: int = 1536,
        d_scores: int = 234,
        d_model: int = 64,
        d_state: int = 8,
        n_classes: int = 234,
        n_windows: int = N_WINDOWS,
        dropout: float = 0.1,
        n_sites: int = 20,
        meta_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.ssm_fwd = SelectiveSSM(d_model, d_state)
        self.ssm_bwd = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm = nn.LayerNorm(d_model)
        self.ssm_drop = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, n_classes)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        first_pass: torch.Tensor,
        site_ids: torch.Tensor | None = None,
        hours: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, steps, _ = embeddings.shape
        hidden = self.input_proj(torch.cat([embeddings, first_pass], dim=-1))
        hidden = hidden + self.pos_enc[:, :steps, :]
        if site_ids is not None and hours is not None:
            metadata = torch.cat(
                [
                    self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings - 1)),
                    self.hour_emb(hours.clamp(0, 23)),
                ],
                dim=-1,
            )
            hidden = hidden + self.meta_proj(metadata).unsqueeze(1)
        residual = hidden
        hidden_fwd = self.ssm_fwd(hidden)
        hidden_bwd = self.ssm_bwd(hidden.flip(1)).flip(1)
        hidden = self.ssm_drop(self.ssm_merge(torch.cat([hidden_fwd, hidden_bwd], dim=-1)))
        hidden = self.ssm_norm(hidden + residual)
        return self.output_head(hidden)


def train_proto(
    embeddings: np.ndarray,
    perch_logits: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
    hour_ids: np.ndarray,
    epochs: int = 40,
    patience: int = 8,
    learning_rate: float = 1e-3,
) -> LightProtoSSM:
    """Execute the Model_51 full-batch LightProtoSSM training schedule."""
    n_classes = labels.shape[1]
    n_files = len(embeddings) // N_WINDOWS
    emb = torch.tensor(embeddings.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32)
    scores = torch.tensor(perch_logits.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32)
    truth = torch.tensor(labels.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32)
    sites = torch.tensor(site_ids, dtype=torch.long)
    hours = torch.tensor(hour_ids, dtype=torch.long)
    model = LightProtoSSM(n_classes=n_classes, n_sites=max(20, int(sites.max()) + 1))
    model.init_prototypes(
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    positive_count = truth.sum(dim=(0, 1))
    total = truth.shape[0] * truth.shape[1]
    positive_weight = ((total - positive_count) / (positive_count + 1)).clamp(max=25.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    best_loss = float("inf")
    best_state = None
    wait = 0
    for _epoch in range(epochs):
        model.train()
        output = model(emb, scores, site_ids=sites, hours=hours)
        loss = F.binary_cross_entropy_with_logits(
            output, truth, pos_weight=positive_weight[None, None, :]
        ) + 0.15 * F.mse_loss(output, scores)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("ProtoSSM training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def train_residual(
    embeddings: np.ndarray,
    first_pass_logits: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
    hour_ids: np.ndarray,
    epochs: int = 30,
    patience: int = 8,
    learning_rate: float = 1e-3,
) -> ResidualSSM:
    """Execute the Model_51 residual-correction training stage."""
    n_classes = labels.shape[1]
    n_files = len(embeddings) // N_WINDOWS
    emb = torch.tensor(embeddings.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32)
    first = torch.tensor(
        first_pass_logits.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32
    )
    truth = labels.reshape(n_files, N_WINDOWS, -1).astype(np.float32)
    first_probability = 1.0 / (1.0 + np.exp(-np.clip(first_pass_logits, -30, 30)))
    residual = torch.tensor(
        truth - first_probability.reshape(n_files, N_WINDOWS, -1), dtype=torch.float32
    )
    sites = torch.tensor(site_ids, dtype=torch.long)
    hours = torch.tensor(hour_ids, dtype=torch.long)
    validation_size = max(1, int(n_files * 0.15))
    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(n_files, generator=generator).numpy()
    valid_index, train_index = permutation[:validation_size], permutation[validation_size:]
    model = ResidualSSM(
        d_scores=n_classes,
        n_classes=n_classes,
        n_sites=max(20, int(sites.max()) + 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=1,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    best_loss = float("inf")
    best_state = None
    wait = 0
    for _epoch in range(epochs):
        model.train()
        correction = model(
            emb[train_index],
            first[train_index],
            site_ids=sites[train_index],
            hours=hours[train_index],
        )
        loss = F.mse_loss(correction, residual[train_index])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            valid_correction = model(
                emb[valid_index],
                first[valid_index],
                site_ids=sites[valid_index],
                hours=hours[valid_index],
            )
            valid_loss = F.mse_loss(valid_correction, residual[valid_index]).item()
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("ResidualSSM training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def validate_payload(payload: np.lib.npyio.NpzFile) -> None:
    required = {
        "embeddings",
        "perch_logits",
        "labels",
        "site_ids",
        "hour_ids",
        "first_pass_logits",
    }
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"Missing NPZ arrays: {sorted(missing)}")
    rows = len(payload["embeddings"])
    if rows % N_WINDOWS:
        raise ValueError(f"Expected a multiple of {N_WINDOWS} windows, received {rows}")
    if payload["perch_logits"].shape != payload["labels"].shape:
        raise ValueError("perch_logits and labels must have the same shape")
    if len(payload["site_ids"]) != rows // N_WINDOWS:
        raise ValueError("site_ids must contain one value per soundscape")
    if len(payload["hour_ids"]) != rows // N_WINDOWS:
        raise ValueError("hour_ids must contain one value per soundscape")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--proto-epochs", type=int, default=40)
    parser.add_argument("--residual-epochs", type=int, default=30)
    args = parser.parse_args()
    seed_everything()
    payload = np.load(args.features, allow_pickle=False)
    validate_payload(payload)
    proto = train_proto(
        payload["embeddings"],
        payload["perch_logits"],
        payload["labels"],
        payload["site_ids"],
        payload["hour_ids"],
        epochs=args.proto_epochs,
    )
    # The historical notebook builds first-pass logits from ProtoSSM plus
    # prior/MLP-adjusted scores. A faithful caller must export that array.
    residual = train_residual(
        payload["embeddings"],
        payload["first_pass_logits"],
        payload["labels"],
        payload["site_ids"],
        payload["hour_ids"],
        epochs=args.residual_epochs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(proto.state_dict(), args.output_dir / "light_protossm.pt")
    torch.save(residual.state_dict(), args.output_dir / "residual_ssm.pt")
    manifest = {
        "source": "ttgrcgrc/birdclef-2026-eos-8, Model_51",
        "windows_per_file": N_WINDOWS,
        "proto_epochs": args.proto_epochs,
        "residual_epochs": args.residual_epochs,
        "seed": SEED,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
