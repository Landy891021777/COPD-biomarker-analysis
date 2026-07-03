"""
01_data_loading.py
===================
Load GSE47460 (Lung Genomics Research Consortium) lung tissue gene-expression
data, merge its two microarray platforms on shared genes, extract clinical
metadata, and keep the COPD vs. control cohort (excluding ILD).

Outputs
-------
data/expr_raw.csv   genes  x samples   (raw expression, common genes only)
data/clinical.csv   samples x clinical features
                    columns: disease_state, GOLD_stage, FEV1, age, sex, platform

Run
---
    python src/01_data_loading.py
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import requests
import GEOparse

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pycombat import Combat

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
GEO_ID = "GSE47460"
PLATFORMS = ["GPL6480", "GPL14550"]  # Agilent 4x44K and 8x60K

# GEOparse defaults to FTP, whose size check is unreliable against NCBI.
# We download the SOFT family file over HTTPS ourselves and load it locally.
SOFT_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE47nnn/GSE47460/soft/GSE47460_family.soft.gz"
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SOFT_PATH = os.path.join(DATA_DIR, "GSE47460_family.soft.gz")
EXPR_OUT = os.path.join(DATA_DIR, "expr_raw.csv")
CLIN_OUT = os.path.join(DATA_DIR, "clinical.csv")
CORR_OUT = os.path.join(DATA_DIR, "expr_corrected.csv")
PCA_BEFORE = os.path.join(FIG_DIR, "pca_before_batch.png")
PCA_AFTER = os.path.join(FIG_DIR, "pca_after_batch.png")
PCA_COMPARE = os.path.join(FIG_DIR, "pca_batch_comparison.png")


def download_soft(url: str, dest: str) -> str:
    """Download the SOFT family file over HTTPS (skips if already complete)."""
    head = requests.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers.get("Content-Length", 0))

    if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
        print(f"      cached: {dest} ({expected/1e6:.1f} MB)")
        return dest

    print(f"      downloading {url}")
    print(f"      -> {dest} ({expected/1e6:.1f} MB)")
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    size = os.path.getsize(tmp)
    if expected and size != expected:
        os.remove(tmp)
        raise IOError(f"size mismatch: got {size}, expected {expected}")
    os.replace(tmp, dest)
    print(f"      downloaded {size/1e6:.1f} MB")
    return dest


# --------------------------------------------------------------------------- #
# Metadata parsing helpers
# --------------------------------------------------------------------------- #
def _characteristics(gsm) -> list[str]:
    """Flatten all characteristics_ch* entries of a GSM into a lowercase list."""
    items: list[str] = []
    for key, vals in gsm.metadata.items():
        if key.startswith("characteristics"):
            items.extend(str(v) for v in vals)
    return items


def _find(chars: list[str], *keywords: str) -> str | None:
    """Return the value part ('key: value') of the first characteristic whose
    key contains ALL given keywords (case-insensitive)."""
    for entry in chars:
        left, _, value = entry.partition(":")
        left_l = left.lower()
        if all(k.lower() in left_l for k in keywords):
            return value.strip()
    return None


def _num(text: str | None) -> float:
    """Extract the first number from a string, else NaN."""
    if not text:
        return np.nan
    m = re.search(r"-?\d+\.?\d*", text)
    return float(m.group()) if m else np.nan


def _norm_disease(raw: str | None) -> str | None:
    """Normalise disease_state to COPD / CTRL / ILD / OTHER."""
    if not raw:
        return None
    r = raw.lower()
    if "copd" in r or "chronic obstructive" in r:
        return "COPD"
    if "control" in r or "ctrl" in r or "normal" in r or "no lung disease" in r:
        return "CTRL"
    if "ild" in r or "interstitial" in r or "ipf" in r or "fibros" in r:
        return "ILD"
    return "OTHER"


def _norm_sex(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.lower()
    if r.startswith("m") or "male" in r and "female" not in r:
        return "M"
    if r.startswith("f") or "female" in r:
        return "F"
    return None


# --------------------------------------------------------------------------- #
# Expression matrix helpers
# --------------------------------------------------------------------------- #
def _gene_symbol_col(gpl_table: pd.DataFrame) -> str:
    """Locate the gene-symbol column in a GPL annotation table."""
    candidates = ["GENE_SYMBOL", "Gene Symbol", "GeneSymbol", "gene_symbol",
                  "Symbol", "GENE_NAME", "ILMN_Gene"]
    for c in candidates:
        if c in gpl_table.columns:
            return c
    # fall back: any column containing 'symbol'
    for c in gpl_table.columns:
        if "symbol" in c.lower():
            return c
    raise KeyError(f"No gene-symbol column found in GPL table: {list(gpl_table.columns)}")


def build_platform_matrix(gse, gpl_name: str) -> pd.DataFrame:
    """Build a genes x samples expression matrix for one platform.

    Probe-level values are collapsed to gene symbols by mean.
    """
    gpl = gse.gpls[gpl_name]
    sym_col = _gene_symbol_col(gpl.table)
    probe2gene = (
        gpl.table.set_index("ID")[sym_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    probe2gene = probe2gene[probe2gene.ne("") & probe2gene.ne("---")]

    # samples belonging to this platform
    sample_series: dict[str, pd.Series] = {}
    for gsm_name, gsm in gse.gsms.items():
        if gsm.metadata.get("platform_id", [None])[0] != gpl_name:
            continue
        tbl = gsm.table
        if "ID_REF" not in tbl.columns or "VALUE" not in tbl.columns:
            continue
        s = pd.to_numeric(
            tbl.set_index("ID_REF")["VALUE"], errors="coerce"
        )
        sample_series[gsm_name] = s

    if not sample_series:
        raise RuntimeError(f"No samples found for platform {gpl_name}")

    probe_mat = pd.DataFrame(sample_series)          # probes x samples
    probe_mat.index = probe_mat.index.astype(str)

    # map probes -> gene symbol, collapse duplicates by mean
    genes = probe2gene.reindex(probe_mat.index)
    probe_mat = probe_mat.loc[genes.notna().values]
    probe_mat["__gene__"] = genes.dropna().values
    gene_mat = probe_mat.groupby("__gene__").mean()  # genes x samples

    print(f"  [{gpl_name}] samples={gene_mat.shape[1]:>4}  genes={gene_mat.shape[0]:>6}")
    return gene_mat


# --------------------------------------------------------------------------- #
# Batch correction (ComBat) + PCA visualization
# --------------------------------------------------------------------------- #
_PLATFORM_COLORS = {"GPL6480": "#d62728", "GPL14550": "#1f77b4"}


def _plot_pca_on_ax(ax, expr_gxs: pd.DataFrame, batch: pd.Series, title: str) -> None:
    """PCA of samples (rows) on a genes x samples matrix, colored by batch."""
    X = StandardScaler().fit_transform(expr_gxs.T.values)  # samples x genes
    pcs = PCA(n_components=2, random_state=0).fit(X)
    coords = pcs.transform(X)
    var = pcs.explained_variance_ratio_ * 100
    for b in batch.unique():
        m = (batch.values == b)
        ax.scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.7,
                   label=f"{b} (n={int(m.sum())})",
                   color=_PLATFORM_COLORS.get(b))
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)


def run_combat(expr_gxs: pd.DataFrame, batch: pd.Series,
               bio: pd.Series | None = None) -> pd.DataFrame:
    """ComBat batch correction. Returns a genes x samples DataFrame.

    expr_gxs : genes x samples (columns aligned to `batch`/`bio` index)
    batch    : platform label per sample (the batch variable to remove)
    bio      : optional biological label per sample to PRESERVE (e.g. disease)
    """
    samples = expr_gxs.columns
    b = batch.loc[samples].values

    # Drop genes with zero variance within any batch (ComBat is undefined there).
    keep = pd.Series(True, index=expr_gxs.index)
    for lvl in np.unique(b):
        sub = expr_gxs.loc[:, samples[b == lvl]]
        keep &= sub.var(axis=1) > 1e-8
    dropped = int((~keep).sum())
    expr_gxs = expr_gxs.loc[keep]
    print(f"      dropped {dropped} genes with zero within-batch variance; "
          f"{expr_gxs.shape[0]} genes remain")

    Y = expr_gxs.T.values.astype(float)  # samples x genes (Combat convention)
    X = None
    if bio is not None:
        # single binary column marking the biological group to protect
        codes = pd.factorize(bio.loc[samples])[0].astype(float)
        X = codes.reshape(-1, 1)
        print(f"      protecting biological covariate: "
              f"{dict(bio.loc[samples].value_counts())}")

    corrected = Combat().fit_transform(Y, b, X=X)  # samples x genes
    if np.isnan(corrected).any():
        raise RuntimeError("ComBat produced NaNs — check batch/variance filtering")

    return pd.DataFrame(corrected.T, index=expr_gxs.index, columns=samples)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"[1/10] Fetching {GEO_ID} SOFT file over HTTPS (first run only)...")
    soft = download_soft(SOFT_URL, SOFT_PATH)
    gse = GEOparse.get_GEO(filepath=soft, silent=True)
    print(f"      platforms available: {list(gse.gpls.keys())}")
    print(f"      total samples:       {len(gse.gsms)}")

    print("[2/10] Building & merging per-platform expression matrices...")
    mats = []
    for gpl_name in PLATFORMS:
        if gpl_name not in gse.gpls:
            print(f"  WARNING: {gpl_name} not present in {GEO_ID}, skipping.")
            continue
        mats.append(build_platform_matrix(gse, gpl_name))

    common_genes = mats[0].index
    for m in mats[1:]:
        common_genes = common_genes.intersection(m.index)
    expr_df = pd.concat([m.loc[common_genes] for m in mats], axis=1)
    expr_df = expr_df[~expr_df.index.duplicated(keep="first")]
    print(f"      merged expr_df: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples "
          f"(common genes = {len(common_genes)})")

    print("[3/10] Extracting clinical metadata...")
    rows = []
    for gsm_name, gsm in gse.gsms.items():
        chars = _characteristics(gsm)
        disease = _norm_disease(
            _find(chars, "disease") or _find(chars, "diagnosis")
        )
        rows.append({
            "sample": gsm_name,
            "disease_state": disease,
            "GOLD_stage": _find(chars, "gold"),
            "FEV1": _num(_find(chars, "fev1")),
            "age": _num(_find(chars, "age")),
            "sex": _norm_sex(_find(chars, "sex") or _find(chars, "gender")),
            "platform": gsm.metadata.get("platform_id", [None])[0],
        })
    clinical_df = pd.DataFrame(rows).set_index("sample")
    print(f"      clinical_df (all): {clinical_df.shape}")
    print("      disease_state counts:")
    print(clinical_df["disease_state"].value_counts(dropna=False).to_string())

    print("[4/10] Filtering to COPD + CTRL (excluding ILD/OTHER)...")
    keep = clinical_df["disease_state"].isin(["COPD", "CTRL"])
    clinical_df = clinical_df[keep]
    # align expression to the same samples that survive filtering AND exist in expr
    shared = clinical_df.index.intersection(expr_df.columns)
    clinical_df = clinical_df.loc[shared]
    expr_df = expr_df[shared]
    print(f"      kept samples: {len(shared)}")
    print(f"      COPD={int((clinical_df['disease_state']=='COPD').sum())}  "
          f"CTRL={int((clinical_df['disease_state']=='CTRL').sum())}")
    print(f"      expr_df:     {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
    print(f"      clinical_df: {clinical_df.shape[0]} samples x {clinical_df.shape[1]} features")

    print("[5/10] Saving raw matrix + clinical to CSV...")
    expr_df.to_csv(EXPR_OUT)
    clinical_df.to_csv(CLIN_OUT)
    print(f"      wrote {EXPR_OUT}")
    print(f"      wrote {CLIN_OUT}")

    # ----- Batch correction (GPL6480 vs GPL14550) ----- #
    batch = clinical_df["platform"]
    bio = clinical_df["disease_state"]

    print("[6/10] PCA before correction (colored by platform)...")
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_pca_on_ax(ax, expr_df, batch, "PCA before ComBat (by platform)")
    fig.tight_layout()
    fig.savefig(PCA_BEFORE, dpi=150)
    plt.close(fig)
    print(f"      wrote {PCA_BEFORE}")

    print("[7/10] Running ComBat (batch=platform, preserving disease_state)...")
    expr_corr = run_combat(expr_df, batch, bio=bio)
    print(f"      corrected matrix: {expr_corr.shape[0]} genes x {expr_corr.shape[1]} samples")

    print("[8/10] PCA after correction (colored by platform)...")
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_pca_on_ax(ax, expr_corr, batch.loc[expr_corr.columns],
                    "PCA after ComBat (by platform)")
    fig.tight_layout()
    fig.savefig(PCA_AFTER, dpi=150)
    plt.close(fig)
    print(f"      wrote {PCA_AFTER}")

    print("[9/10] Side-by-side comparison figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_pca_on_ax(axes[0], expr_df, batch, "Before ComBat")
    _plot_pca_on_ax(axes[1], expr_corr, batch.loc[expr_corr.columns], "After ComBat")
    fig.suptitle("GSE47460 platform batch effect: before vs. after ComBat", y=1.02)
    fig.tight_layout()
    fig.savefig(PCA_COMPARE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {PCA_COMPARE}")

    print("[10/10] Saving corrected matrix + done.")
    expr_corr.to_csv(CORR_OUT)
    print(f"      wrote {CORR_OUT}")
    print("      Final shapes:")
    print(f"        expr_raw.csv       : {expr_df.shape}    (genes x samples)")
    print(f"        expr_corrected.csv : {expr_corr.shape}    (genes x samples)")
    print(f"        clinical.csv       : {clinical_df.shape}      (samples x features)")


if __name__ == "__main__":
    main()
