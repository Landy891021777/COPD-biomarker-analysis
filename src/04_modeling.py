"""
04_modeling.py
==============
Biomarker-panel feature selection + COPD vs. CTRL classification.

Feature selection
-----------------
  1. Candidate pool = statistically significant DEGs (q < 0.05) from
     results/deg_results.csv.
  2. LASSO (LassoCV) shrinks the panel to ~20-50 key genes.
  3. Final biomarker panel saved to results/biomarker_panel.csv.

Modeling
--------
  4. 80/20 stratified split (random_state=42).
  5. Logistic Regression (baseline)
  6. Random Forest (n_estimators=200)
  7. XGBoost
  8. Metrics: ROC-AUC, F1, Precision, Recall, Accuracy (on held-out test)
  9. Overlaid ROC curves -> figures/roc_curves.png
 10. Confusion matrix of the best model -> figures/confusion_matrix.png
 11. results/model_comparison.csv

LEAKAGE NOTE
------------
Feature selection is a subtle source of leakage. To keep the reported test
metrics honest, the LASSO panel is fit on the TRAINING split only; the test set
is never seen during selection. The q<0.05 DEG pre-filter (from 03, full cohort)
is a mild pre-reduction; a fully nested CV would recompute DEGs per fold.

Inputs : data/expr_corrected.csv, data/clinical.csv, results/deg_results.csv
Run    : python src/04_modeling.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, lasso_path, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, roc_curve,
                             confusion_matrix)
from xgboost import XGBClassifier
import shap
import joblib

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
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_BUNDLE = os.path.join(MODEL_DIR, "best_model_bundle.joblib")

RANDOM_STATE = 42
Q_THRESH = 0.05
PANEL_MIN, PANEL_MAX = 20, 50

PANEL_OUT = os.path.join(RES_DIR, "biomarker_panel.csv")
COMPARE_OUT = os.path.join(RES_DIR, "model_comparison.csv")
ROC_OUT = os.path.join(FIG_DIR, "roc_curves.png")
CM_OUT = os.path.join(FIG_DIR, "confusion_matrix.png")
SHAP_SUMMARY_OUT = os.path.join(FIG_DIR, "shap_summary.png")
SHAP_BAR_OUT = os.path.join(FIG_DIR, "shap_bar.png")

sns.set_style("whitegrid")


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def lasso_panel(X_tr: np.ndarray, y_tr: np.ndarray, genes: pd.Index) -> pd.Series:
    """LASSO feature selection on the TRAINING set only.

    Returns a Series (gene -> coefficient) of PANEL_MIN..PANEL_MAX genes.
    """
    lcv = LassoCV(cv=5, random_state=RANDOM_STATE, n_jobs=-1, max_iter=10000)
    lcv.fit(X_tr, y_tr)
    coef = pd.Series(lcv.coef_, index=genes)
    n_nz = int((coef != 0).sum())
    print(f"      LassoCV alpha={lcv.alpha_:.5f} -> {n_nz} non-zero genes")

    if PANEL_MIN <= n_nz <= PANEL_MAX:
        chosen = coef[coef != 0]
    else:
        # walk the LASSO path and pick an alpha whose non-zero count is in range
        alphas, coefs, _ = lasso_path(X_tr, y_tr, n_alphas=200)
        counts = (coefs != 0).sum(axis=0)
        in_range = np.where((counts >= PANEL_MIN) & (counts <= PANEL_MAX))[0]
        if len(in_range):
            j = in_range[np.argmin(np.abs(counts[in_range] - 35))]  # target ~35
            c = pd.Series(coefs[:, j], index=genes)
            chosen = c[c != 0]
            print(f"      path re-selection: alpha={alphas[j]:.5f} -> {len(chosen)} genes")
        else:
            chosen = coef[coef != 0]

    # hard clamp by |coef| in case the path still overshoots
    chosen = chosen.reindex(chosen.abs().sort_values(ascending=False).index)
    if len(chosen) > PANEL_MAX:
        chosen = chosen.head(PANEL_MAX)
    return chosen


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate(model, X_te, y_te) -> dict:
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_te, proba),
        "F1": f1_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "Accuracy": accuracy_score(y_te, pred),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    expr = pd.read_csv(os.path.join(DATA_DIR, "expr_corrected.csv"), index_col=0)
    clin = pd.read_csv(os.path.join(DATA_DIR, "clinical.csv"), index_col=0)
    deg = pd.read_csv(os.path.join(RES_DIR, "deg_results.csv"), index_col=0)

    disease = clin.loc[expr.columns, "disease_state"]
    y = (disease == "COPD").astype(int).values          # positive class = COPD
    print(f"loaded: expr {expr.shape}, clinical {clin.shape}, "
          f"deg {deg.shape}  (COPD=1 n={int(y.sum())}, CTRL=0 n={int((y==0).sum())})\n")

    # ---- candidate pool: significant DEGs (q < 0.05) ----
    cand = deg.index[deg["qvalue"] < Q_THRESH]
    cand = cand.intersection(expr.index)
    print(f"[1/11] Candidate significant genes (q<{Q_THRESH}): {len(cand)}")
    Xcand = expr.loc[cand].T.values                     # samples x candidate genes

    # ---- 80/20 stratified split (BEFORE selection, to avoid leakage) ----
    print("[4/11] Stratified 80/20 split (done before LASSO to avoid leakage)...")
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.20, stratify=y,
                              random_state=RANDOM_STATE)
    print(f"      train={len(tr)} (COPD={int(y[tr].sum())}/CTRL={int((y[tr]==0).sum())})  "
          f"test={len(te)} (COPD={int(y[te].sum())}/CTRL={int((y[te]==0).sum())})")

    scaler = StandardScaler().fit(Xcand[tr])            # fit on TRAIN only
    Xtr_all = scaler.transform(Xcand[tr])
    Xte_all = scaler.transform(Xcand[te])

    # ---- LASSO feature selection (train only) ----
    print("[2/11] LASSO feature selection (LassoCV) on training set...")
    panel = lasso_panel(Xtr_all, y[tr].astype(float), cand)
    print(f"[3/11] Final biomarker panel: {len(panel)} genes")
    print("       " + ", ".join(panel.index[:15].tolist())
          + (" ..." if len(panel) > 15 else ""))
    pd.DataFrame({"gene": panel.index, "lasso_coef": panel.values}) \
        .to_csv(PANEL_OUT, index=False)
    print(f"       wrote {PANEL_OUT}")

    # restrict feature matrices to the selected panel
    keep = [cand.get_loc(g) for g in panel.index]
    Xtr, Xte = Xtr_all[:, keep], Xte_all[:, keep]
    ytr, yte = y[tr], y[te]

    # ---- models ----
    pos, neg = int(ytr.sum()), int((ytr == 0).sum())
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=neg / pos, random_state=RANDOM_STATE, n_jobs=-1),
    }

    print("\n[5-8/11] Training & evaluating models (test-set metrics)...")
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    rows, roc_data = {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        m = evaluate(mdl, Xte, yte)
        m["CV_AUC_train"] = cross_val_score(mdl, Xtr, ytr, cv=cv,
                                            scoring="roc_auc").mean()
        rows[name] = m
        fpr, tpr, _ = roc_curve(yte, mdl.predict_proba(Xte)[:, 1])
        roc_data[name] = (fpr, tpr, m["ROC_AUC"])
        print(f"      {name:20s} test AUC={m['ROC_AUC']:.3f}  F1={m['F1']:.3f}  "
              f"Prec={m['Precision']:.3f}  Rec={m['Recall']:.3f}  "
              f"Acc={m['Accuracy']:.3f}  (CV-AUC={m['CV_AUC_train']:.3f})")

    comp = pd.DataFrame(rows).T[
        ["ROC_AUC", "F1", "Precision", "Recall", "Accuracy", "CV_AUC_train"]]
    comp.to_csv(COMPARE_OUT)
    print(f"\n[11/11] wrote {COMPARE_OUT}")

    best = comp["ROC_AUC"].idxmax()
    best_auc = comp.loc[best, "ROC_AUC"]
    print(f"      BEST model: {best}  (test ROC-AUC = {best_auc:.3f})")

    # persist the trained best model so external validation (05) can reuse it
    # without retraining (train -> save -> serve).
    joblib.dump({
        "model": models[best],
        "name": best,
        "genes": list(panel.index),          # 34 panel gene symbols, in order
        "test_auc": float(best_auc),
        "train_mean": scaler.mean_[keep],     # per-panel-gene standardization params
        "train_scale": scaler.scale_[keep],
    }, MODEL_BUNDLE)
    print(f"      saved best-model bundle -> {MODEL_BUNDLE}")

    # ---- ROC curves ----
    print("[9/11] ROC comparison figure...")
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Chance")
    ax.set_title("ROC curves: COPD vs. CTRL classification (test set)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(ROC_OUT, dpi=150)
    plt.close(fig)
    print(f"      wrote {ROC_OUT}")

    # ---- confusion matrix (best model) ----
    print("[10/11] Confusion matrix of best model...")
    best_pred = (models[best].predict_proba(Xte)[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(yte, best_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["CTRL", "COPD"], yticklabels=["CTRL", "COPD"], ax=ax)
    ax.set_title(f"Confusion matrix — {best} (test AUC={best_auc:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(CM_OUT, dpi=150)
    plt.close(fig)
    print(f"      wrote {CM_OUT}")

    # ---- SHAP interpretability (best model) ----
    print(f"\n[SHAP] Explaining best model ({best}) on all {len(y)} samples...")
    X_all = np.vstack([Xtr, Xte])
    X_all_df = pd.DataFrame(X_all, columns=panel.index)

    if best in ("XGBoost", "RandomForest"):
        explainer = shap.TreeExplainer(models[best])
        shap_values = explainer.shap_values(X_all_df)
    else:  # LogisticRegression -> linear explainer
        explainer = shap.LinearExplainer(models[best], Xtr)
        shap_values = explainer.shap_values(X_all_df)
    # normalise possible multi-class shapes to the positive-class (COPD) matrix
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:               # (n, features, n_classes)
        shap_values = shap_values[:, :, 1]

    shap_imp = pd.Series(np.abs(shap_values).mean(axis=0),
                         index=panel.index).sort_values(ascending=False)

    # (2) beeswarm summary — top 20 genes
    print("      SHAP summary (beeswarm, top 20)...")
    shap.summary_plot(shap_values, X_all_df, max_display=20, show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 8)
    fig.suptitle(f"SHAP summary — {best} (top 20 biomarkers)", y=1.02)
    fig.tight_layout(); fig.savefig(SHAP_SUMMARY_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {SHAP_SUMMARY_OUT}")

    # (3) global importance bar plot
    print("      SHAP global importance (bar)...")
    shap.summary_plot(shap_values, X_all_df, plot_type="bar",
                      max_display=len(panel), show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 9)
    fig.suptitle(f"SHAP global feature importance — {best}", y=1.02)
    fig.tight_layout(); fig.savefig(SHAP_BAR_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {SHAP_BAR_OUT}")

    # (4) dependence plots for the top-3 genes
    print("      SHAP dependence plots (top 3 genes)...")
    for rank, gene in enumerate(shap_imp.index[:3], 1):
        shap.dependence_plot(gene, shap_values, X_all_df, show=False)
        fig = plt.gcf(); fig.set_size_inches(6.5, 5)
        fig.suptitle(f"SHAP dependence — {gene} (#{rank})", y=1.02)
        out = os.path.join(FIG_DIR, f"shap_dependence_{gene}.png")
        fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      wrote {out}")

    # (5) enriched biomarker panel: gene, direction, SHAP importance
    print("[panel] Writing enriched biomarker panel...")
    lfc = deg.loc[panel.index, "log2FC"]
    out_panel = pd.DataFrame({
        "gene": panel.index,
        "lasso_coef": panel.values,
        "log2FC": lfc.values,
        "direction": np.where(lfc.values > 0, "up_in_COPD", "down_in_COPD"),
        "shap_importance": shap_imp.reindex(panel.index).values,
    }).sort_values("shap_importance", ascending=False)
    out_panel.to_csv(PANEL_OUT, index=False)
    print(f"      wrote {PANEL_OUT}  ({len(out_panel)} genes)")
    print("\nTop 10 biomarkers by SHAP importance:")
    print(out_panel.head(10)[["gene", "direction", "shap_importance"]].to_string(index=False))

    print(f"\nDone. Panel={len(panel)} genes | best={best} AUC={best_auc:.3f}")


if __name__ == "__main__":
    main()
