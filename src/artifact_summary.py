"""Summarize the public BirdCLEF 2026 research metadata.

The repository intentionally does not track the raw Kaggle notebook exports.
This script summarizes the curated CSV files that replace those artifacts in the
public repo: submission history, leaderboard row and notebook-theme inventory.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEYWORDS = [
    "Perch",
    "ProtoSSM",
    "ResidualSSM",
    "EfficientNet",
    "SED",
    "pseudo",
    "rank",
    "blend",
    "ensemble",
    "ONNX",
    "PCEN",
    "gated",
    "OOF",
    "PCA",
    "MLP",
    "GroupKFold",
    "TTA",
    "prior",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_submissions(root: Path) -> None:
    rows = read_csv(root / "data" / "submissions.csv")
    scored = [
        row
        for row in rows
        if row.get("publicScore") not in ("", None)
    ]
    best_public = max(scored, key=lambda row: float(row["publicScore"]))
    best_private = max(
        [row for row in scored if row.get("privateScore") not in ("", None)],
        key=lambda row: float(row["privateScore"]),
    )
    print("Submission summary")
    print(f"- total rows: {len(rows)}")
    print(
        f"- best public: {best_public['publicScore']} "
        f"({best_public['description']})"
    )
    print(
        f"- best private: {best_private['privateScore']} "
        f"({best_private['description']})"
    )


def summarize_leaderboard(root: Path) -> None:
    rows = read_csv(root / "data" / "leaderboard_summary.csv")
    if not rows:
        return
    row = rows[0]
    print("\nLeaderboard summary")
    print(
        f"- rank: {row['Rank']}, score: {row['Score']}, "
        f"submissions: {row['SubmissionCount']}"
    )


def summarize_notebooks(root: Path) -> None:
    inventory = read_csv(root / "data" / "notebook_inventory.csv")
    keyword_counts = {keyword: 0 for keyword in KEYWORDS}
    for row in inventory:
        text = row.get("keywords", "")
        for keyword in KEYWORDS:
            if keyword.lower() in text.lower():
                keyword_counts[keyword] += 1

    print("\nNotebook summary")
    print(f"- notebooks: {len(inventory)}")
    for keyword, count in sorted(
        keyword_counts.items(), key=lambda item: item[1], reverse=True
    )[:10]:
        print(f"- {keyword}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    summarize_submissions(root)
    summarize_leaderboard(root)
    summarize_notebooks(root)


if __name__ == "__main__":
    main()
