"""Create evidence-backed figures for the BirdCLEF+ 2026 technical report."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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


def box(ax, xy, wh, text, color, fontsize=8):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.015",
        linewidth=1.0, edgecolor="#04364A", facecolor=color
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return (x, y, w, h)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=1.0, color="#355C64"))


fig, ax = plt.subplots(figsize=(11.2, 5.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

box(ax, (0.015, 0.39), (0.105, 0.22), "1-min audio\n32 kHz\n12 × 5 s", "#DAFFFB")
box(ax, (0.155, 0.67), (0.125, 0.18), "Perch v2\nlogits + 1536-D\nembeddings", "#B8E8E5")
box(ax, (0.155, 0.15), (0.125, 0.18), "log-mel\nSED inputs", "#FFE1C2")
box(ax, (0.325, 0.70), (0.15, 0.13), "mapping + priors\n+ MLP probes", "#B8E8E5")
box(ax, (0.325, 0.49), (0.15, 0.13), "LightProtoSSM\ncontext + prototypes", "#B8E8E5")
box(ax, (0.325, 0.28), (0.15, 0.13), "ResidualSSM\nerror correction", "#B8E8E5")
box(ax, (0.325, 0.07), (0.15, 0.13), "5-fold distilled\nEfficientNet SED", "#FFE1C2")
box(ax, (0.525, 0.28), (0.15, 0.30), "Model_51\nrank fusion 0.60/0.40\nconditional gates\nfile + temporal rules", "#92D9D2")
box(ax, (0.525, 0.69), (0.15, 0.13), "Model_22\npublic reproduction", "#D7CCF3")
box(ax, (0.725, 0.43), (0.115, 0.16), "Direct blend\n3% / 97%", "#FFF0B5")
box(ax, (0.875, 0.43), (0.11, 0.16), "Genus 0.15\nthen class 0.05", "#FFD19A")

arrow(ax, (0.12, 0.53), (0.155, 0.76))
arrow(ax, (0.12, 0.47), (0.155, 0.24))
arrow(ax, (0.28, 0.76), (0.325, 0.765))
arrow(ax, (0.28, 0.74), (0.325, 0.555))
arrow(ax, (0.40, 0.49), (0.40, 0.41))
arrow(ax, (0.28, 0.24), (0.325, 0.135))
arrow(ax, (0.475, 0.765), (0.525, 0.52))
arrow(ax, (0.475, 0.555), (0.525, 0.47))
arrow(ax, (0.475, 0.345), (0.525, 0.40))
arrow(ax, (0.475, 0.135), (0.525, 0.33))
arrow(ax, (0.675, 0.755), (0.725, 0.54))
arrow(ax, (0.675, 0.43), (0.725, 0.49))
arrow(ax, (0.84, 0.51), (0.875, 0.51))

ax.text(0.5, 0.96, "Reconstructed inference architecture of the private-best submission",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#04364A")
ax.text(0.59, 0.23, "internal public ensemble", ha="center", fontsize=7, color="#355C64")
fig.tight_layout()
fig.savefig(OUT / "architecture.pdf", bbox_inches="tight")
fig.savefig(OUT / "architecture.png", dpi=220, bbox_inches="tight")
plt.close(fig)
