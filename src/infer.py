"""Reproduce the final, score-defining EoS.8 ensemble stage.

The best private submission (0.94216) combined the complete Model_22 and
Model_51 prediction tables with weights 0.03 and 0.97, then applied genus and
taxonomic-class smoothing with alpha values 0.15 and 0.05.  This script
reproduces that final stage exactly and validates the Kaggle submission schema.

The expensive component inference remains preserved in the canonical Kaggle
notebook under ``notebooks/birdclef-2026-eos-8.ipynb``.  Model_22 and Model_51
are attributed third-party-derived pipelines; see ``SOURCES.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_22_WEIGHT = 0.03
MODEL_51_WEIGHT = 0.97
GENUS_ALPHA = 0.15
CLASS_ALPHA = 0.05


def read_submission(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "row_id" not in frame.columns:
        raise ValueError(f"row_id column missing in {path}")
    if not frame["row_id"].is_unique:
        raise ValueError(f"duplicate row_id values in {path}")
    probability_columns = [column for column in frame.columns if column != "row_id"]
    if not probability_columns:
        raise ValueError(f"no probability columns in {path}")
    values = frame[probability_columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"NaN or infinite probability in {path}")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"probability outside [0, 1] in {path}")
    frame["row_id"] = frame["row_id"].astype(str)
    return frame.set_index("row_id")


def weighted_component_blend(
    model_22: pd.DataFrame,
    model_51: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the active EoS.8 top-level 3/97 direct blend."""
    if list(model_22.columns) != list(model_51.columns):
        raise ValueError("Model_22 and Model_51 probability columns differ")
    missing = model_22.index.difference(model_51.index)
    extra = model_51.index.difference(model_22.index)
    if len(missing) or len(extra):
        raise ValueError(
            f"row_id mismatch: missing_from_model_51={len(missing)}, "
            f"extra_in_model_51={len(extra)}"
        )
    aligned_51 = model_51.loc[model_22.index, model_22.columns]
    return MODEL_22_WEIGHT * model_22 + MODEL_51_WEIGHT * aligned_51


def taxonomy_smoothing(
    predictions: pd.DataFrame,
    taxonomy: pd.DataFrame,
    genus_alpha: float = GENUS_ALPHA,
    class_alpha: float = CLASS_ALPHA,
) -> pd.DataFrame:
    """Apply the exact v221 genus-then-class smoothing used by EoS.8."""
    required = {"primary_label", "scientific_name", "class_name"}
    missing = required.difference(taxonomy.columns)
    if missing:
        raise ValueError(f"taxonomy is missing columns: {sorted(missing)}")
    taxonomy = taxonomy.copy()
    taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
    label_to_genus = {
        row.primary_label: str(row.scientific_name).split(" ")[0]
        for row in taxonomy.itertuples(index=False)
    }
    label_to_class = {
        row.primary_label: str(row.class_name)
        for row in taxonomy.itertuples(index=False)
    }
    genus_groups: dict[str, list[str]] = {}
    class_groups: dict[str, list[str]] = {}
    for column in predictions.columns:
        genus_groups.setdefault(label_to_genus.get(column, column), []).append(column)
        class_name = label_to_class.get(column, "")
        if class_name:
            class_groups.setdefault(class_name, []).append(column)
    probabilities = predictions.to_numpy(dtype=np.float32, copy=True)
    column_positions = {column: index for index, column in enumerate(predictions.columns)}
    for members in genus_groups.values():
        if len(members) <= 1:
            continue
        positions = [column_positions[member] for member in members]
        mean = probabilities[:, positions].mean(axis=1, keepdims=True)
        probabilities[:, positions] = (
            (1.0 - genus_alpha) * probabilities[:, positions] + genus_alpha * mean
        )
    for members in class_groups.values():
        if len(members) <= 1:
            continue
        positions = [column_positions[member] for member in members]
        mean = probabilities[:, positions].mean(axis=1, keepdims=True)
        probabilities[:, positions] = (
            (1.0 - class_alpha) * probabilities[:, positions] + class_alpha * mean
        )
    return pd.DataFrame(
        probabilities, index=predictions.index, columns=predictions.columns
    )


def align_to_sample(predictions: pd.DataFrame, sample_path: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    sample["row_id"] = sample["row_id"].astype(str)
    target_columns = sample.columns[1:].tolist()
    if set(target_columns) != set(predictions.columns):
        raise ValueError("prediction columns do not match sample_submission.csv")
    if set(sample["row_id"]) != set(predictions.index):
        raise ValueError("prediction row_id values do not match sample_submission.csv")
    output = predictions.loc[sample["row_id"], target_columns].reset_index()
    output.columns = sample.columns
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-22", required=True, type=Path)
    parser.add_argument("--model-51", required=True, type=Path)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--sample-submission", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()
    model_22 = read_submission(args.model_22)
    model_51 = read_submission(args.model_51)
    direct = weighted_component_blend(model_22, model_51)
    smoothed = taxonomy_smoothing(direct, pd.read_csv(args.taxonomy))
    if args.sample_submission:
        output = align_to_sample(smoothed, args.sample_submission)
    else:
        output = smoothed.reset_index()
    values = output.iloc[:, 1:].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("final EoS.8 probabilities failed validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(
        f"Wrote {args.output}: rows={len(output)}, classes={output.shape[1]-1}, "
        f"min={values.min():.6f}, max={values.max():.6f}"
    )


if __name__ == "__main__":
    main()
