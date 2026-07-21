"""Create evidence-backed figures for the BirdCLEF+ 2026 technical report."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

train = pd.read_csv(ROOT / "data" / "competition_metadata" / "train.csv")
counts = train["primary_label"].value_counts().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(6.7, 3.15))
ax.plot(range(1, len(counts) + 1), counts.values, color="#176B87", linewidth=1.6)
ax.fill_between(range(1, len(counts) + 1), counts.values, color="#64CCC5", alpha=0.25)
ax.set_yscale("log")
ax.set_xlabel("Observed primary classes, ordered by frequency")
ax.set_ylabel("Training recordings (log scale)")
ax.set_title("Long-tailed distribution of primary labels")
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(OUT / "class_frequency.pdf", bbox_inches="tight")
fig.savefig(OUT / "class_frequency.png", dpi=220, bbox_inches="tight")
plt.close(fig)

taxonomy = pd.read_csv(ROOT / "data" / "competition_metadata" / "taxonomy.csv")
tax_counts = taxonomy["class_name"].value_counts().sort_values()
fig, ax = plt.subplots(figsize=(6.7, 2.8))
colors = ["#DAFFFB", "#64CCC5", "#176B87", "#04364A", "#FF9E44"]
bars = ax.barh(tax_counts.index, tax_counts.values, color=colors[: len(tax_counts)])
ax.bar_label(bars, padding=3)
ax.set_xlabel("Target classes")
ax.set_title("Taxonomic composition of the 234 competition targets")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "taxonomy.pdf", bbox_inches="tight")
fig.savefig(OUT / "taxonomy.png", dpi=220, bbox_inches="tight")
plt.close(fig)

subs = pd.read_csv(ROOT / "data" / "submissions.csv")
subs["date"] = pd.to_datetime(subs["date"])
scored = subs.dropna(subset=["publicScore", "privateScore"]).sort_values("date")
fig, ax = plt.subplots(figsize=(6.7, 3.2))
ax.plot(scored["date"], scored["publicScore"], label="Public", color="#176B87", linewidth=1.5)
ax.plot(scored["date"], scored["privateScore"], label="Private", color="#E56B6F", linewidth=1.5)
ax.scatter(scored["date"], scored["publicScore"], color="#176B87", s=11)
ax.scatter(scored["date"], scored["privateScore"], color="#E56B6F", s=11)
ax.axhline(0.5, color="0.55", linestyle="--", linewidth=0.8, label="Chance baseline")
ax.set_ylabel("Macro ROC-AUC")
ax.set_xlabel("Submission date (2026)")
ax.set_title("Official Kaggle score progression (scored submissions)")
ax.legend(frameon=False, ncol=3, loc="lower right")
ax.grid(alpha=0.22)
fig.autofmt_xdate(rotation=25)
fig.tight_layout()
fig.savefig(OUT / "score_progression.pdf", bbox_inches="tight")
fig.savefig(OUT / "score_progression.png", dpi=220, bbox_inches="tight")
plt.close(fig)

summary = {
    "train_rows": len(train),
    "observed_primary_classes": train["primary_label"].nunique(),
    "target_classes": len(taxonomy),
    "secondary_label_nonempty_rows": int((train["secondary_labels"] != "[]").sum()),
    "recordings_min": int(counts.min()),
    "recordings_median": float(counts.median()),
    "recordings_max": int(counts.max()),
}
print(summary)
