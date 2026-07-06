"""
03_deg_analysis.py
==================
Differential expression analysis (COPD vs. CTRL) on the batch-corrected
GSE47460 matrix.

Pipeline
--------
  1. Per-gene Mann-Whitney U test (COPD vs. CTRL)
  2. Benjamini-Hochberg FDR correction -> q-value (statsmodels)
  3. log2 fold change = mean(COPD) - mean(CTRL)   [data is already log2]
  4. Significant DEG: q < 0.05 AND |log2FC| > 1.0
  5. Volcano plot (top-20 genes labeled)
  6. Heatmap of top-50 DEGs, samples ordered by disease label
  7. Save results/deg_results.csv

Inputs : data/expr_corrected.csv, data/clinical.csv
Run    : python src/03_deg_analysis.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "figures")
RES_DIR = os.path.join(HERE, "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

Q_THRESH = 0.05
LFC_THRESH = 1.0

COLORS = {"COPD": "#ff7f0e", "CTRL": "#1f77b4"}  # up = COPD-high (orange), down = CTRL-high (blue)
NS_COLOR = "#bbbbbb"

DEG_OUT = os.path.join(RES_DIR, "deg_results.csv")
VOLCANO_OUT = os.path.join(FIG_DIR, "volcano_plot.png")
HEATMAP_OUT = os.path.join(FIG_DIR, "heatmap_deg.png")


# --------------------------------------------------------------------------- #
# DEG computation
# --------------------------------------------------------------------------- #
def compute_deg(expr: pd.DataFrame, disease: pd.Series) -> pd.DataFrame:
    """Vectorized Mann-Whitney U + BH-FDR + log2FC per gene."""
    copd_cols = disease.index[disease == "COPD"]
    ctrl_cols = disease.index[disease == "CTRL"]
    print(f"      COPD samples={len(copd_cols)}  CTRL samples={len(ctrl_cols)}")

    x = expr[copd_cols].values   # genes x COPD
    y = expr[ctrl_cols].values   # genes x CTRL

    # axis=1 vectorizes the test across all genes at once
    stat, pval = mannwhitneyu(x, y, axis=1, alternative="two-sided")
    qval = multipletests(pval, method="fdr_bh")[1]
    log2fc = x.mean(axis=1) - y.mean(axis=1)   # already log2 scale

    deg = pd.DataFrame({
        "gene": expr.index,
        "log2FC": log2fc,
        "U_stat": stat,
        "pvalue": pval,
        "qvalue": qval,
    }).set_index("gene")
    deg["neg_log10_q"] = -np.log10(deg["qvalue"].clip(lower=np.finfo(float).tiny))
    deg["significant"] = (deg["qvalue"] < Q_THRESH) & (deg["log2FC"].abs() > LFC_THRESH)
    deg["direction"] = np.where(deg["significant"],
                                np.where(deg["log2FC"] > 0, "up_COPD", "down_COPD"),
                                "ns")
    deg = deg.sort_values("qvalue")
    n_sig = int(deg["significant"].sum())
    print(f"      significant DEGs (q<{Q_THRESH}, |log2FC|>{LFC_THRESH}): {n_sig}")
    print(f"        up in COPD  : {int((deg.direction=='up_COPD').sum())}")
    print(f"        down in COPD: {int((deg.direction=='down_COPD').sum())}")
    return deg


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_volcano(deg: pd.DataFrame) -> None:
    print("      drawing volcano plot...")
    fig, ax = plt.subplots(figsize=(9, 7))
    up = deg[deg.direction == "up_COPD"]
    dn = deg[deg.direction == "down_COPD"]
    ns = deg[deg.direction == "ns"]
    ax.scatter(ns.log2FC, ns.neg_log10_q, s=6, c=NS_COLOR, alpha=0.4, label="Not significant")
    ax.scatter(up.log2FC, up.neg_log10_q, s=10, c=COLORS["COPD"], alpha=0.7,
               label=f"Up in COPD (n={len(up)})")
    ax.scatter(dn.log2FC, dn.neg_log10_q, s=10, c=COLORS["CTRL"], alpha=0.7,
               label=f"Down in COPD (n={len(dn)})")

    ax.axhline(-np.log10(Q_THRESH), ls="--", c="grey", lw=1)
    ax.axvline(LFC_THRESH, ls="--", c="grey", lw=1)
    ax.axvline(-LFC_THRESH, ls="--", c="grey", lw=1)

    # label the top-20 most significant genes among the DEGs that pass BOTH
    # thresholds (colored points on the wings) — these are the biomarker
    # candidates and avoid a cluster of unlabelable small-fold-change genes.
    labelset = deg[deg.significant].head(20)
    if labelset.empty:                       # fallback if nothing passes both cuts
        labelset = deg.head(20)
    for gene, row in labelset.iterrows():
        ax.annotate(gene, (row.log2FC, row.neg_log10_q), fontsize=7, fontweight="bold",
                    xytext=(3, 3), textcoords="offset points")

    ax.set_title("Volcano plot: COPD vs. CTRL differential expression\n"
                 f"(threshold: q < {Q_THRESH}, |log2FC| > {LFC_THRESH})")
    ax.set_xlabel("log2 Fold Change  (COPD − CTRL)")
    ax.set_ylabel("−log10(q-value)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(VOLCANO_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {VOLCANO_OUT}")


def plot_heatmap(deg: pd.DataFrame, expr: pd.DataFrame, disease: pd.Series) -> None:
    print("      drawing heatmap of top-50 DEGs...")
    sig = deg[deg.significant]
    top = (sig if len(sig) >= 50 else deg).head(50).index

    # order samples by disease label (COPD block, then CTRL block)
    order = list(disease.index[disease == "COPD"]) + list(disease.index[disease == "CTRL"])
    sub = expr.loc[top, order]

    # z-score per gene for visualization
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, 1), axis=0)
    col_colors = disease.loc[order].map(COLORS)

    g = sns.clustermap(
        z, row_cluster=True, col_cluster=False,
        col_colors=col_colors.values, cmap="RdBu_r", center=0,
        vmin=-3, vmax=3, xticklabels=False, yticklabels=True,
        figsize=(12, 11), cbar_kws={"label": "z-score (per gene)"},
    )
    g.ax_heatmap.set_xlabel("Samples (left: COPD | right: CTRL)")
    g.ax_heatmap.set_ylabel("Top-50 DEGs")
    g.ax_heatmap.tick_params(axis="y", labelsize=7)
    # legend for the disease strip
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ["COPD", "CTRL"]]
    g.ax_col_dendrogram.legend(handles, ["COPD", "CTRL"], title="Group",
                               loc="center", ncol=2, frameon=False)
    g.figure.suptitle("Top-50 differentially expressed genes (COPD vs. CTRL)", y=1.01)
    g.savefig(HEATMAP_OUT, dpi=150, bbox_inches="tight")
    plt.close(g.figure)
    print(f"      wrote {HEATMAP_OUT}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    expr = pd.read_csv(os.path.join(DATA_DIR, "expr_corrected.csv"), index_col=0)
    clin = pd.read_csv(os.path.join(DATA_DIR, "clinical.csv"), index_col=0)
    disease = clin.loc[expr.columns, "disease_state"]
    print(f"loaded: expr_corrected {expr.shape}, clinical {clin.shape}\n")

    print("[1-4/7] Differential expression (Mann-Whitney U + BH-FDR + log2FC)...")
    deg = compute_deg(expr, disease)

    print("[5/7] Volcano plot...")
    plot_volcano(deg)

    print("[6/7] Heatmap...")
    plot_heatmap(deg, expr, disease)

    print("[7/7] Saving DEG table...")
    deg.to_csv(DEG_OUT)
    print(f"      wrote {DEG_OUT}  ({deg.shape[0]} genes)")

    print("\nTop 10 DEGs by q-value:")
    print(deg.head(10)[["log2FC", "qvalue", "direction"]].to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
