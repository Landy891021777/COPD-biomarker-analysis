"""
app.py — COPD Biomarker Dashboard (Streamlit)
=============================================
Interactive dashboard for the COPD lung gene-expression side project
(申請久浪智醫 AI 演算法工程師 / applying to JuLang Smart-Medicine).

Four pages:
  1. 專案總覽  Project Overview
  2. EDA 探索  Exploratory Data Analysis
  3. 模型結果  Model Results
  4. 病人預測 Demo  Live Prediction Demo

Run:  streamlit run app.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
BUNDLE_PATH = os.path.join(MODELS, "best_model_bundle.joblib")

COPD_COLOR = "#ff7f0e"   # orange
CTRL_COLOR = "#1f77b4"   # blue

# Fixed dataset facts (from the pipeline)
N_TRAIN, N_TRAIN_COPD, N_TRAIN_CTRL = 328, 220, 108
N_GENES = 15180
N_EXT, N_EXT_COPD, N_EXT_CTRL = 117, 77, 40

st.set_page_config(page_title="COPD Biomarker Dashboard",
                   page_icon="🫁", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_resource
def load_bundle():
    if not os.path.exists(BUNDLE_PATH):
        return None
    import joblib
    return joblib.load(BUNDLE_PATH)


def show_image(path: str, caption: str = "") -> None:
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"找不到圖檔 / figure not found: `{os.path.basename(path)}`"
                   "（請先執行對應的 pipeline 腳本）")


def metric_from(df: pd.DataFrame | None, col: str, agg="max", default=None):
    if df is None or col not in df.columns:
        return default
    return getattr(df[col], agg)()


# --------------------------------------------------------------------------- #
# Sidebar navigation
# --------------------------------------------------------------------------- #
st.sidebar.title("🫁 COPD Biomarker")
st.sidebar.caption("肺部基因表現 AI 分析 · Lung gene-expression AI")
PAGE = st.sidebar.radio(
    "頁面 / Page",
    ["① 專案總覽 Overview",
     "② EDA 探索 Exploration",
     "③ 模型結果 Model Results",
     "④ 病人預測 Demo"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Landy Huang · GSE47460 → GSE151052\n\n"
                   "資料集為公開資料；原始資料未上傳版控。")

# load shared artifacts
panel_df = load_csv(os.path.join(RES, "biomarker_panel.csv"))
comp_df = load_csv(os.path.join(RES, "model_comparison.csv"))
ext_df = load_csv(os.path.join(RES, "external_validation.csv"))
deg_df = load_csv(os.path.join(RES, "deg_results.csv"))


# --------------------------------------------------------------------------- #
# PAGE 1 — Overview
# --------------------------------------------------------------------------- #
if PAGE.startswith("①"):
    st.title("COPD 肺部基因 AI 分析　Lung Gene-Expression AI for COPD")
    st.markdown(
        "#### 應用情境　Application context\n"
        "本專案模擬**久浪智醫肺器官晶片 (lung organ-on-chip) 的 AI 分析核心**："
        "從肺組織基因表現資料建立**慢性阻塞性肺病 (COPD) 分類模型**，"
        "並找出可解釋的**關鍵診斷 biomarker**。此流程對應器官晶片上「感測 → "
        "基因/分子讀出 → AI 判讀疾病狀態」的分析管線。\n\n"
        "> *A reproducible ML pipeline that classifies COPD vs. control from lung "
        "gene expression and surfaces interpretable diagnostic biomarkers — "
        "mirroring the AI readout core of a lung organ-on-chip platform.*"
    )

    st.markdown("### 📊 資料集規模　Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("訓練樣本 Train", f"{N_TRAIN}", f"COPD {N_TRAIN_COPD} / CTRL {N_TRAIN_CTRL}")
    c2.metric("基因數 Genes", f"{N_GENES:,}", "2 平台 ComBat 校正")
    c3.metric("外部驗證 External", f"{N_EXT}", f"COPD {N_EXT_COPD} / CTRL {N_EXT_CTRL}")
    c4.metric("平台 Platforms", "Agilent → Affymetrix", "跨平台驗證")

    st.markdown("### 🎯 關鍵結果　Key results")
    best_auc = metric_from(comp_df, "ROC_AUC", "max", 0.934)
    n_deg = int(deg_df["significant"].sum()) if (deg_df is not None
              and "significant" in deg_df) else 28
    n_panel = len(panel_df) if panel_df is not None else 34
    ext_auc = None
    if ext_df is not None and {"metric", "value"} <= set(ext_df.columns):
        row = ext_df.loc[ext_df["metric"] == "external_AUC", "value"]
        ext_auc = float(row.iloc[0]) if len(row) else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("最佳模型 AUC（內部測試）", f"{best_auc:.3f}", "XGBoost")
    k2.metric("外部驗證 AUC", f"{ext_auc:.3f}" if ext_auc else "0.651",
              "跨世代泛化")
    k3.metric("顯著 DEG", f"{n_deg}", "q<0.05 & |log2FC|>1")
    k4.metric("Biomarker Panel", f"{n_panel} 基因", "LASSO 篩選")

    st.info(
        "**方法學誠信 Integrity**：內部測試 AUC 0.934 是理想條件下的表現；"
        "獨立世代 (GSE151052) 外部驗證 AUC 0.651，落差主因為 **Agilent→Affymetrix "
        "跨平台批次效應**。主動做外部驗證並誠實揭露泛化落差，是本專案刻意展現的嚴謹態度。",
        icon="🔬",
    )

    st.markdown("### 🧬 分析流程　Pipeline")
    st.markdown(
        "`載入 GSE47460` → `ComBat 批次校正` → `EDA` → "
        "`DEG (Mann-Whitney + BH-FDR)` → `LASSO 特徵選擇` → "
        "`LR / RF / XGBoost` → `SHAP 可解釋性` → `GSE151052 外部驗證`"
    )


# --------------------------------------------------------------------------- #
# PAGE 2 — EDA
# --------------------------------------------------------------------------- #
elif PAGE.startswith("②"):
    st.title("EDA 探索　Exploratory Data Analysis")
    st.caption("橘色 = COPD，藍色 = CTRL（全站一致）")

    st.subheader("① 肺功能比較　Lung function (FEV1)")
    st.markdown("COPD 病人的 FEV1（第一秒用力吐氣量 %預測值）顯著低於對照組。")
    col = st.columns([1, 1])
    with col[0]:
        show_image(os.path.join(FIG, "fev1_boxplot.png"),
                   "FEV1 % predicted：COPD vs. CTRL（Mann-Whitney U）")
    with col[1]:
        show_image(os.path.join(FIG, "biomarker_boxplot.png"),
                   "已知 COPD biomarker 表現量比較")

    st.subheader("② 降維視覺化　UMAP")
    st.markdown(
        "所有基因降到 2D。COPD 與 CTRL 呈**部分分離**——全轉錄組層級疾病訊號"
        "只佔總變異一小部分，因此需要特徵選擇 + 監督式模型（見下一頁）。")
    c = st.columns([1, 2, 1])
    with c[1]:
        show_image(os.path.join(FIG, "umap_disease.png"),
                   "UMAP of all genes, colored by disease label")


# --------------------------------------------------------------------------- #
# PAGE 3 — Model Results
# --------------------------------------------------------------------------- #
elif PAGE.startswith("③"):
    st.title("模型結果　Model Results")

    st.subheader("① 三模型比較　Model comparison")
    if comp_df is not None:
        show = comp_df.rename(columns={comp_df.columns[0]: "Model"}).set_index("Model")
        st.dataframe(show.style.format("{:.3f}").highlight_max(
            subset=["ROC_AUC"], color="#ffe0b2"), use_container_width=True)
    col = st.columns(2)
    with col[0]:
        show_image(os.path.join(FIG, "roc_curves.png"),
                   "ROC 曲線比較（測試集）")
    with col[1]:
        show_image(os.path.join(FIG, "confusion_matrix.png"),
                   "最佳模型 (XGBoost) Confusion Matrix")

    st.subheader("② 可解釋性　SHAP")
    st.markdown("SHAP 量化每個基因對 COPD 判斷的貢獻。紅=高表現、藍=低表現。")
    c = st.columns([1, 3, 1])
    with c[1]:
        show_image(os.path.join(FIG, "shap_summary.png"),
                   "SHAP summary（前 20 個 biomarker）")

    st.subheader("③ Biomarker Panel（LASSO + SHAP）")
    if panel_df is not None:
        pdf = panel_df.copy()
        if "shap_importance" in pdf.columns:
            pdf = pdf.sort_values("shap_importance", ascending=False)
        st.dataframe(pdf.reset_index(drop=True), use_container_width=True, height=380)
    else:
        st.warning("找不到 biomarker_panel.csv")


# --------------------------------------------------------------------------- #
# PAGE 4 — Live Prediction Demo
# --------------------------------------------------------------------------- #
elif PAGE.startswith("④"):
    st.title("病人預測 Demo　Live Prediction")
    st.markdown(
        "拖動下方 **5 個關鍵基因**的表現量滑桿，模型會**即時**計算此虛擬病人罹患 "
        "COPD 的機率。\n\n*Drag the 5 key-gene sliders to see the live COPD "
        "probability from the trained XGBoost model.*")

    bundle = load_bundle()
    if bundle is None:
        st.error("找不到模型檔 `models/best_model_bundle.joblib`。"
                 "請先執行 `python src/04_modeling.py` 產生模型。")
        st.stop()

    genes = list(bundle["genes"])
    means = np.asarray(bundle["train_mean"], dtype=float)
    scales = np.asarray(bundle["train_scale"], dtype=float)
    model = bundle["model"]

    # top-5 slider genes by SHAP importance (fallback to first 5 panel genes)
    if panel_df is not None and "shap_importance" in panel_df.columns:
        top5 = (panel_df.sort_values("shap_importance", ascending=False)["gene"]
                .head(5).tolist())
    else:
        top5 = genes[:5]
    dir_map = {}
    if panel_df is not None and "direction" in panel_df.columns:
        dir_map = dict(zip(panel_df["gene"], panel_df["direction"]))

    st.markdown("#### 🎚️ 調整基因表現量　Adjust gene expression (log2)")
    left, right = st.columns([3, 2])

    # start every panel gene at its cohort mean (z = 0)
    z_vector = np.zeros(len(genes))
    with left:
        for g in top5:
            i = genes.index(g)
            mu, sd = means[i], scales[i]
            lo, hi = mu - 3 * sd, mu + 3 * sd
            direction = dir_map.get(g, "")
            hint = ("↑ 高表現偏 COPD" if direction == "up_in_COPD"
                    else "↓ 低表現偏 COPD" if direction == "down_in_COPD" else "")
            val = st.slider(f"**{g}**　{hint}", float(round(lo, 2)),
                            float(round(hi, 2)), float(round(mu, 2)), step=0.05,
                            help=f"訓練集平均 {mu:.2f}（log2）。0 z-score = 族群平均。")
            z_vector[i] = (val - mu) / (sd if sd else 1.0)

    # predict
    proba = float(model.predict_proba(z_vector.reshape(1, -1))[0, 1])

    with right:
        st.markdown("#### 🩺 預測結果　Prediction")
        st.metric("COPD 機率 Probability", f"{proba*100:.1f}%")
        st.progress(proba)
        label = "COPD" if proba >= 0.5 else "CTRL (健康對照)"
        color = COPD_COLOR if proba >= 0.5 else CTRL_COLOR
        st.markdown(
            f"<div style='padding:0.6rem 1rem;border-radius:8px;background:{color};"
            f"color:white;font-weight:700;text-align:center;font-size:1.1rem'>"
            f"判定 / Prediction：{label}</div>", unsafe_allow_html=True)

        # illustrative GOLD-stage band (NOT a real GOLD predictor)
        if proba < 0.35:
            gold = "GOLD 0 – At Risk（傾向健康對照）"
        elif proba < 0.55:
            gold = "GOLD 0–1（邊界 / 輕度）"
        elif proba < 0.75:
            gold = "GOLD 1–2（輕度–中度）"
        elif proba < 0.90:
            gold = "GOLD 2–3（中度–重度）"
        else:
            gold = "GOLD 3–4（重度–極重度）"
        st.markdown(f"**對應 GOLD 分期範圍（示意）**\n\n🫁 {gold}")
        st.caption("⚠️ GOLD 分期為**示意對照**，非模型直接輸出——模型僅做 "
                   "COPD/CTRL 二元分類，實際分期需依臨床肺功能 (FEV1) 判定。")

    st.markdown("---")
    st.caption(
        f"其餘 {len(genes) - len(top5)} 個 panel 基因固定在族群平均值 (z=0)；"
        "此為說明用的簡化 demo，使用的是真實訓練好的 XGBoost 模型。")


st.sidebar.markdown("---")
st.sidebar.caption("© Landy Huang — COPD biomarker side project")
