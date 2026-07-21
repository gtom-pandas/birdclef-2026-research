from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infer import taxonomy_smoothing, weighted_component_blend


def test_eos8_component_weights() -> None:
    index = pd.Index(["file_5", "file_10"], name="row_id")
    model_22 = pd.DataFrame({"a": [0.0, 1.0], "b": [0.4, 0.6]}, index=index)
    model_51 = pd.DataFrame({"a": [1.0, 0.0], "b": [0.8, 0.2]}, index=index)
    result = weighted_component_blend(model_22, model_51)
    expected = 0.03 * model_22 + 0.97 * model_51
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy())


def test_taxonomy_smoothing_matches_genus_then_class_order() -> None:
    predictions = pd.DataFrame(
        {"a": [1.0], "b": [0.0], "c": [0.5]},
        index=pd.Index(["file_5"], name="row_id"),
    )
    taxonomy = pd.DataFrame(
        {
            "primary_label": ["a", "b", "c"],
            "scientific_name": ["Genus one", "Genus two", "Other three"],
            "class_name": ["Aves", "Aves", "Aves"],
        }
    )
    result = taxonomy_smoothing(predictions, taxonomy, genus_alpha=0.15, class_alpha=0.05)
    after_genus = np.array([[0.925, 0.075, 0.5]], dtype=np.float32)
    expected = 0.95 * after_genus + 0.05 * after_genus.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)


def test_component_alignment_uses_model_22_order() -> None:
    model_22 = pd.DataFrame(
        {"a": [0.1, 0.2]}, index=pd.Index(["x", "y"], name="row_id")
    )
    model_51 = pd.DataFrame(
        {"a": [0.8, 0.7]}, index=pd.Index(["y", "x"], name="row_id")
    )
    result = weighted_component_blend(model_22, model_51)
    np.testing.assert_allclose(result["a"].to_numpy(), [0.682, 0.782])
