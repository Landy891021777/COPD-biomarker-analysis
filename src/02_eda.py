"""
02_eda.py
=========
Exploratory data analysis for the GSE47460 COPD cohort. Produces five figures
in figures/:

  1. sample_distribution.png  — COPD vs CTRL counts, sub-grouped by GOLD stage
  2. fev1_boxplot.png         — FEV1 by group + Mann-Whitney U p-value
  3. expression_violin.png    — per-sample expression violins (log2 sanity check)
  4. biomarker_boxplot.png    — known COPD biomarkers, COPD vs CTRL
  5. umap_disease.png         — UMAP of all genes, colored by disease label

Inputs (produced by src/01_data_loading.py):
  data/clinical.csv, data/expr_raw.csv, data/expr_corrected.csv

Run:  python src/02_eda.py
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# COPD = orange, CTRL = blue (used consistently across all figures)
COLORS = {"COPD": "#ff7f0e", "CTRL": "#1f77b4"}
GROUP_ORDER = ["COPD", "CTRL"]

GOLD_ORDER = ["0-At Risk", "1-Mild COPD", "2-Moderate COPD",
              "3-Severe COPD", "4-Very Severe COPD", "Unknown"]

BIOMARKERS = ["MMP9", "IL6", "SFTPC", "SFTPB", "HIF1A"]

RANDOM_STATE = 42
sns.set_style("whitegrid")


def _fmt_p(p: float) -> str:
    return f"p = {p:.2e}" if p < 1e-3 else f"p = {p:.3f}"


def _savefig(fig, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {path}")


# --------------------------------------------------------------------------- #
# Figure builders
# --------------------------------------------------------------------------- #
def fig1_sample_distribution(clin: pd.DataFrame) -> None:
    print("[1/5] Sample distribution by disease + GOLD stage...")
    df = clin.copy()
    df["GOLD_stage"] = df["GOLD_stage"].fillna("Unknown")
    order = [g for g in GOLD_ORDER if g in set(df["GOLD_stage"])]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.countplot(data=df, x="GOLD_stage", hue="disease_state",
                  order=order, hue_order=GROUP_ORDER, palette=COLORS, ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    ax.set_title("GSE47460 sample distribution: COPD vs. CTRL by GOLD stage")
    ax.set_xlabel("GOLD stage")
    ax.set_ylabel("Number of samples")
    ax.legend(title="Group")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _savefig(fig, "sample_distribution.png")


def fig2_fev1_boxplot(clin: pd.DataFrame) -> None:
    print("[2/5] FEV1 boxplot with Mann-Whitney U test...")
    df = clin.dropna(subset=["FEV1"])
    copd = df.loc[df.disease_state == "COPD", "FEV1"]
    ctrl = df.loc[df.disease_state == "CTRL", "FEV1"]
    _, p = mannwhitneyu(copd, ctrl, alternative="two-sided")

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=df, x="disease_state", y="FEV1", order=GROUP_ORDER,
                hue="disease_state", palette=COLORS, legend=False,
                width=0.55, fliersize=2, ax=ax)
    sns.stripplot(data=df, x="disease_state", y="FEV1", order=GROUP_ORDER,
                  color="black", alpha=0.25, size=2.5, ax=ax)

    y = df["FEV1"].max() * 1.05
    ax.plot([0, 0, 1, 1], [y, y * 1.02, y * 1.02, y], lw=1.2, c="black")
    ax.text(0.5, y * 1.03, f"Mann-Whitney U  {_fmt_p(p)}",
            ha="center", va="bottom", fontsize=10)
    ax.set_title("Lung function (FEV1 % predicted): COPD vs. CTRL")
    ax.set_xlabel("Group")
    ax.set_ylabel("FEV1 (% predicted)")
    _savefig(fig, "fev1_boxplot.png")


def fig3_expression_violin(expr_raw: pd.DataFrame) -> None:
    print("[3/5] Per-sample expression violins (log2 sanity check)...")
    rng = np.random.default_rng(RANDOM_STATE)
    cols = rng.choice(expr_raw.columns, size=5, replace=False)
    long = (expr_raw[cols]
            .melt(var_name="sample", value_name="expression"))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.violinplot(data=long, x="sample", y="expression",
                   inner="quartile", color="#8fbfe0", ax=ax)
    ax.set_title("Per-sample expression distribution (5 random samples)\n"
                 "Aligned, roughly symmetric distributions confirm log2 normalization")
    ax.set_xlabel("Sample (GSM ID)")
    ax.set_ylabel("Expression (log2)")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    _savefig(fig, "expression_violin.png")


def fig4_biomarker_boxplot(expr: pd.DataFrame, clin: pd.DataFrame) -> None:
    print("[4/5] Known COPD biomarker boxplots...")
    disease = clin.loc[expr.columns, "disease_state"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    for ax, gene in zip(axes, BIOMARKERS):
        if gene not in expr.index:
            ax.text(0.5, 0.5, f"{gene}\nnot in shared\nplatform genes",
                    ha="center", va="center", fontsize=11, color="grey")
            ax.set_title(gene)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        vals = expr.loc[gene]
        df = pd.DataFrame({"expression": vals.values,
                           "disease_state": disease.values})
        sns.boxplot(data=df, x="disease_state", y="expression", order=GROUP_ORDER,
                    hue="disease_state", palette=COLORS, legend=False,
                    width=0.55, fliersize=2, ax=ax)
        copd = df.loc[df.disease_state == "COPD", "expression"]
        ctrl = df.loc[df.disease_state == "CTRL", "expression"]
        _, p = mannwhitneyu(copd, ctrl, alternative="two-sided")
        ax.set_title(f"{gene}   ({_fmt_p(p)})")
        ax.set_xlabel("")
        ax.set_ylabel("Expression (log2, batch-corrected)")

    # hide any unused panel(s)
    for ax in axes[len(BIOMARKERS):]:
        ax.set_visible(False)

    fig.suptitle("Known COPD biomarker expression: COPD vs. CTRL", y=1.0, fontsize=13)
    _savefig(fig, "biomarker_boxplot.png")


def fig5_umap(expr: pd.DataFrame, clin: pd.DataFrame) -> None:
    print("[5/5] UMAP embedding colored by disease...")
    disease = clin.loc[expr.columns, "disease_state"]
    X = StandardScaler().fit_transform(expr.T.values)          # samples x genes
    # PCA pre-reduction denoises and speeds up UMAP on ~15k genes
    X50 = PCA(n_components=50, random_state=RANDOM_STATE).fit_transform(X)
    emb = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                    n_neighbors=15, min_dist=0.3).fit_transform(X50)

    fig, ax = plt.subplots(figsize=(7, 6))
    for grp in GROUP_ORDER:
        m = (disease.values == grp)
        ax.scatter(emb[m, 0], emb[m, 1], s=16, alpha=0.75,
                   c=COLORS[grp], label=f"{grp} (n={int(m.sum())})")
    ax.set_title("UMAP of GSE47460 lung expression (all genes), by disease label")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(title="Group")
    _savefig(fig, "umap_disease.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    clin = pd.read_csv(os.path.join(DATA_DIR, "clinical.csv"), index_col=0)
    expr_raw = pd.read_csv(os.path.join(DATA_DIR, "expr_raw.csv"), index_col=0)
    expr_corr = pd.read_csv(os.path.join(DATA_DIR, "expr_corrected.csv"), index_col=0)
    print(f"loaded: clinical {clin.shape}, expr_raw {expr_raw.shape}, "
          f"expr_corrected {expr_corr.shape}\n")

    fig1_sample_distribution(clin)
    fig2_fev1_boxplot(clin)
    fig3_expression_violin(expr_raw)          # raw values for normalization check
    fig4_biomarker_boxplot(expr_corr, clin)   # batch-corrected for biology
    fig5_umap(expr_corr, clin)                # batch-corrected so platform ≠ driver

    print("\nDone. 5 figures written to figures/.")


if __name__ == "__main__":
    main()
