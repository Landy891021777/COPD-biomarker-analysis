"""
05_external_validation.py
=========================
External validation of the Day-2 biomarker panel + trained model on an
independent cohort: GSE151052 (GPL17556, Affymetrix; 77 COPD + 40 Control = 117).

The training data (GSE47460) is Agilent; GSE151052 is Affymetrix and annotates
probes by Entrez Gene ID, so the panel (gene symbols) is matched cross-platform
via a symbol->Entrez map built from the GSE47460 Agilent annotation.

Steps
-----
  1. Download GSE151052 (HTTPS SOFT, cached).
  2. Keep the training biomarker-panel genes; record the missing fraction.
  3. Predict with the Day-2 best model (loaded from models/, NOT retrained).
  4. Metrics on the external cohort: ROC-AUC, F1, Accuracy.
  5. ROC curve annotated with internal-test AUC vs external AUC.
  6. If the AUC gap > 0.1, print concrete improvement suggestions.
  7. Save results/external_validation.csv

Design note (cross-study): absolute intensities differ across platforms, so each
panel gene is z-scored WITHIN the external cohort (standard for cross-study model
transfer). Missing genes are imputed as 0 (the z-score mean) = neutral.

Run: python src/05_external_validation.py
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import GEOparse
import requests
import joblib
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score, roc_curve)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
FIG_DIR = os.path.join(HERE, "..", "figures")
RES_DIR = os.path.join(HERE, "..", "results")
MODEL_DIR = os.path.join(HERE, "..", "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

EXT_ID = "GSE151052"
EXT_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/"
           "GSE151nnn/GSE151052/soft/GSE151052_family.soft.gz")
EXT_SOFT = os.path.join(DATA_DIR, "GSE151052_family.soft.gz")
TRAIN_SOFT = os.path.join(DATA_DIR, "GSE47460_family.soft.gz")

BUNDLE_PATH = os.path.join(MODEL_DIR, "best_model_bundle.joblib")
SYM2ENT_CACHE = os.path.join(MODEL_DIR, "symbol2entrez.csv")
ROC_EXT_OUT = os.path.join(FIG_DIR, "roc_external.png")
EXT_OUT = os.path.join(RES_DIR, "external_validation.csv")

AUC_GAP_ALERT = 0.10


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def download_soft(url: str, dest: str) -> str:
    head = requests.head(url, timeout=60, allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers.get("Content-Length", 0))
    if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
        print(f"      cached: {dest} ({expected/1e6:.1f} MB)")
        return dest
    print(f"      downloading {url} ({expected/1e6:.1f} MB)")
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    os.replace(tmp, dest)
    return dest


def symbol_to_entrez() -> dict:
    """Map gene symbol -> Entrez ID using the GSE47460 Agilent annotation
    (columns GENE_SYMBOL + GENE). Cached to CSV to avoid re-parsing 78 MB."""
    if os.path.exists(SYM2ENT_CACHE):
        df = pd.read_csv(SYM2ENT_CACHE, dtype=str)
        return dict(zip(df["symbol"], df["entrez"]))
    print("      building symbol->Entrez map from GSE47460 annotation...")
    gse = GEOparse.get_GEO(filepath=TRAIN_SOFT, silent=True)
    mapping: dict[str, str] = {}
    for gpl in gse.gpls.values():
        t = gpl.table
        if "GENE" in t.columns and "GENE_SYMBOL" in t.columns:
            sub = t[["GENE_SYMBOL", "GENE"]].dropna()
            for sym, ent in zip(sub["GENE_SYMBOL"].astype(str), sub["GENE"]):
                try:
                    mapping[sym.strip()] = str(int(float(ent)))
                except (ValueError, TypeError):
                    continue
    pd.DataFrame({"symbol": list(mapping), "entrez": list(mapping.values())}) \
        .to_csv(SYM2ENT_CACHE, index=False)
    return mapping


def load_external_expr(gse) -> pd.DataFrame:
    """Entrez-indexed expression matrix (genes x samples) for GSE151052.
    Probe ID_REF is '<entrez>_at'; duplicates collapsed by mean."""
    series = {}
    for gsm_name, gsm in gse.gsms.items():
        tbl = gsm.table
        if "ID_REF" not in tbl.columns or "VALUE" not in tbl.columns:
            continue
        ent = tbl["ID_REF"].astype(str).str.split("_").str[0]
        s = pd.to_numeric(tbl["VALUE"], errors="coerce")
        s.index = ent
        series[gsm_name] = s
    mat = pd.DataFrame(series)
    mat = mat[mat.index.str.fullmatch(r"\d+")]          # keep numeric Entrez only
    mat = mat.groupby(mat.index).mean()
    return mat


def extract_labels(gse) -> pd.Series:
    """COPD=1 / Control=0 from the 'disease state' characteristic."""
    rows = {}
    for gsm_name, gsm in gse.gsms.items():
        val = None
        for k, vs in gsm.metadata.items():
            if k.startswith("characteristics"):
                for v in vs:
                    if "disease" in str(v).lower() or "diagnosis" in str(v).lower():
                        val = str(v).lower()
        if val is None:
            continue
        if "copd" in val:
            rows[gsm_name] = 1
        elif "control" in val or "ctrl" in val or "normal" in val:
            rows[gsm_name] = 0
    return pd.Series(rows, name="label")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print("[1/7] Loading Day-2 model bundle...")
    bundle = joblib.load(BUNDLE_PATH)
    model, panel_genes = bundle["model"], bundle["genes"]
    internal_auc = bundle["test_auc"]
    print(f"      model={bundle['name']}  panel={len(panel_genes)} genes  "
          f"internal test AUC={internal_auc:.3f}")

    print(f"[1/7] Downloading {EXT_ID}...")
    download_soft(EXT_URL, EXT_SOFT)
    gse = GEOparse.get_GEO(filepath=EXT_SOFT, silent=True)
    print(f"      platform(s)={list(gse.gpls.keys())}  samples={len(gse.gsms)}")

    ext_expr = load_external_expr(gse)              # entrez x samples
    labels = extract_labels(gse)
    shared = labels.index.intersection(ext_expr.columns)
    labels = labels.loc[shared]
    ext_expr = ext_expr[shared]
    print(f"      external cohort: {len(shared)} samples "
          f"(COPD={int((labels==1).sum())}, CTRL={int((labels==0).sum())})")

    print("[2/7] Matching biomarker panel across platforms (via Entrez ID)...")
    sym2ent = symbol_to_entrez()
    panel_entrez = {g: sym2ent.get(g) for g in panel_genes}
    present, missing = [], []
    for g in panel_genes:
        ent = panel_entrez[g]
        if ent is not None and ent in ext_expr.index:
            present.append(g)
        else:
            missing.append(g)
    miss_frac = len(missing) / len(panel_genes)
    print(f"      panel genes present: {len(present)}/{len(panel_genes)}  "
          f"(missing {len(missing)} = {miss_frac:.1%})")
    if missing:
        print(f"      missing: {', '.join(missing)}")

    # build feature matrix samples x panel (train gene order); z-score within cohort
    feat = pd.DataFrame(0.0, index=shared, columns=panel_genes)
    for g in present:
        row = ext_expr.loc[panel_entrez[g]]
        z = (row - row.mean()) / (row.std(ddof=0) or 1.0)
        feat[g] = z.values
    X_ext = feat.values
    y_ext = labels.values

    print("[3/7] Predicting with Day-2 model (no retraining)...")
    proba = model.predict_proba(X_ext)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("[4/7] External metrics...")
    ext_auc = roc_auc_score(y_ext, proba)
    metrics = {
        "ROC_AUC": ext_auc,
        "F1": f1_score(y_ext, pred),
        "Precision": precision_score(y_ext, pred),
        "Recall": recall_score(y_ext, pred),
        "Accuracy": accuracy_score(y_ext, pred),
    }
    for k, v in metrics.items():
        print(f"      {k:10s} = {v:.3f}")
    gap = internal_auc - ext_auc
    print(f"      internal test AUC={internal_auc:.3f}  external AUC={ext_auc:.3f}  "
          f"gap={gap:+.3f}")

    print("[5/7] ROC figure (internal vs external)...")
    fpr, tpr, _ = roc_curve(y_ext, proba)
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot(fpr, tpr, lw=2.5, color="#d62728",
            label=f"External {EXT_ID} (AUC={ext_auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Chance")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.text(0.55, 0.18,
            f"Internal test (GSE47460): AUC = {internal_auc:.3f}\n"
            f"External   (GSE151052): AUC = {ext_auc:.3f}\n"
            f"Gap = {gap:+.3f}",
            fontsize=10, bbox=dict(boxstyle="round", fc="#fff3e0", ec="grey"))
    ax.set_title(f"External validation ROC — {bundle['name']} biomarker panel")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(ROC_EXT_OUT, dpi=150)
    plt.close(fig)
    print(f"      wrote {ROC_EXT_OUT}")

    print("[6/7] Generalization assessment...")
    suggestions = []
    if gap > AUC_GAP_ALERT:
        print(f"      [!] AUC dropped by {gap:.3f} (> {AUC_GAP_ALERT}) - likely "
              f"cross-platform batch effect. Suggestions:")
        suggestions = [
            "Cross-platform shift (Agilent->Affymetrix): apply cross-cohort "
            "normalization (ComBat / quantile) or rank-transform features.",
            "Train on rank-based or platform-robust features rather than raw "
            "intensities so the model transfers across platforms.",
            f"{len(missing)} panel genes absent on the external array - refit the "
            "panel on the intersection of both platforms' measurable genes.",
            "Pool both cohorts (multi-study training) or use domain adaptation.",
            "Report internal CV as an optimistic bound; external AUC is the "
            "honest deployment estimate.",
        ]
        for s in suggestions:
            print(f"        - {s}")
    else:
        print(f"      [OK] AUC gap {gap:+.3f} within {AUC_GAP_ALERT} - panel "
              f"generalizes reasonably to the independent cohort.")

    print("[7/7] Saving results...")
    out = pd.DataFrame({
        "metric": ["internal_test_AUC", "external_AUC", "AUC_gap",
                   "external_F1", "external_Precision", "external_Recall",
                   "external_Accuracy", "n_external_samples",
                   "n_COPD", "n_CTRL", "panel_genes", "panel_present",
                   "panel_missing", "missing_fraction"],
        "value": [round(internal_auc, 4), round(ext_auc, 4), round(gap, 4),
                  round(metrics["F1"], 4), round(metrics["Precision"], 4),
                  round(metrics["Recall"], 4), round(metrics["Accuracy"], 4),
                  len(shared), int((labels == 1).sum()), int((labels == 0).sum()),
                  len(panel_genes), len(present), len(missing),
                  round(miss_frac, 4)],
    })
    out.to_csv(EXT_OUT, index=False)
    print(f"      wrote {EXT_OUT}")
    print(f"\nDone. External AUC={ext_auc:.3f} (internal {internal_auc:.3f}, "
          f"gap {gap:+.3f}); panel {len(present)}/{len(panel_genes)} genes matched.")


if __name__ == "__main__":
    main()
