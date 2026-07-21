from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import LightProtoSSM, ResidualSSM


def test_sequence_heads_preserve_window_predictions() -> None:
    embeddings = torch.randn(2, 12, 16)
    logits = torch.randn(2, 12, 5)
    sites = torch.tensor([1, 2])
    hours = torch.tensor([6, 18])
    proto = LightProtoSSM(
        d_input=16,
        d_model=8,
        d_state=4,
        n_classes=5,
        n_sites=4,
        meta_dim=4,
        cross_attn_heads=2,
    )
    proto_output = proto(embeddings, logits, site_ids=sites, hours=hours)
    assert proto_output.shape == (2, 12, 5)

    residual = ResidualSSM(
        d_input=16,
        d_scores=5,
        d_model=8,
        d_state=4,
        n_classes=5,
        n_sites=4,
        meta_dim=4,
    )
    residual_output = residual(embeddings, logits, site_ids=sites, hours=hours)
    assert residual_output.shape == (2, 12, 5)
