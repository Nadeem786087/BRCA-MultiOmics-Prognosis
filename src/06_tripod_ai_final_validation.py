import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

os.makedirs("results/v3_tripod_gold", exist_ok=True)

# 1. Load Harmonized Cohorts
tcga_expr = pd.read_parquet("data/processed/harmonized_tcga_expr.parquet")
tcga_mut  = pd.read_parquet("data/processed/harmonized_tcga_mut.parquet")
tcga_clin = pd.read_csv("data/processed/harmonized_tcga_clin.csv", index_col=0)

meta_expr = pd.read_parquet("data/processed/harmonized_meta_expr.parquet")
meta_mut  = pd.read_parquet("data/processed/harmonized_meta_mut.parquet")
meta_clin = pd.read_csv("data/processed/harmonized_meta_clin.csv", index_col=0)

gse_expr  = pd.read_parquet("data/processed/harmonized_gse_expr.parquet")
gse_clin  = pd.read_csv("data/processed/harmonized_gse_clin.csv", index_col=0)

train_ids = pd.read_csv("data/processed/tcga_train_ids.csv")["sample_id"].tolist()
test_ids  = pd.read_csv("data/processed/tcga_test_ids.csv")["sample_id"].tolist()
all_tcga_ids = train_ids + test_ids

# Extract Disease-Specific Survival (DSS) in METABRIC if available
raw_meta_pat = pd.read_csv("data/raw/brca_metabric/data_clinical_patient.txt", sep="\t", comment="#").set_index("PATIENT_ID")
raw_meta_pat = raw_meta_pat.loc[meta_clin.index]
if "VITAL_STATUS" in raw_meta_pat.columns:
    meta_clin["DSS"] = (raw_meta_pat["VITAL_STATUS"].astype(str) == "Died of Disease").astype(int)
else:
    meta_clin["DSS"] = meta_clin["OS"]
print(f"METABRIC Breast-Cancer-Specific Deaths (DSS): {meta_clin['DSS'].sum()} / {len(meta_clin)} ({meta_clin['DSS'].mean()*100:.1f}%)")

# 2. Extract GISTIC2 CNA + SNV + Sample-Normalized Protein-Coding mRNA + Hallmark Pathways
CNA_DRIVERS = ["ERBB2", "MYC", "CCND1", "FGFR1", "MDM2", "CDK4", "CDK6",
               "PIK3CA", "AKT1", "PTEN", "RB1", "TP53", "CDKN2A", "BRCA1", "BRCA2"]
SNV_DRIVERS = ["TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "KMT2C", "PTEN", "AKT1", "RB1", "BRCA1", "BRCA2"]

def load_cna(path, sample_list):
    cna_raw = pd.read_csv(path, sep="\t").dropna(subset=["Hugo_Symbol"]).groupby("Hugo_Symbol").mean(numeric_only=True)
    out = pd.DataFrame(index=sample_list)
    for g in CNA_DRIVERS:
        out[f"CNA_{g}"] = cna_raw.loc[g].reindex(sample_list).fillna(0.0).values if g in cna_raw.index else 0.0
    return out.fillna(0.0)

tcga_cna = load_cna("data/raw/brca_tcga_pan_can_atlas_2018/data_cna.txt", all_tcga_ids)
meta_cna = load_cna("data/raw/brca_metabric/data_cna.txt", meta_clin.index.tolist())

def build_snv(mut_df, clin_df, sids):
    g_df = pd.DataFrame(index=sids)
    for g in SNV_DRIVERS:
        g_df[f"SNV_{g}"] = mut_df.loc[g, sids].fillna(0).values if g in mut_df.index else 0
    g_df["LOG_TMB"] = np.log1p(clin_df.loc[sids, "TMB"].fillna(0).values)
    return g_df.fillna(0.0)

tcga_snv = build_snv(tcga_mut, tcga_clin, all_tcga_ids)
meta_snv = build_snv(meta_mut, meta_clin, meta_clin.index.tolist())

def sample_zscore(df):
    return df.sub(df.mean(axis=0), axis=1).div(df.std(axis=0).replace(0, 1), axis=1).fillna(0.0)

expr_tcga = sample_zscore(tcga_expr[all_tcga_ids])
expr_tr   = expr_tcga[train_ids]
expr_te   = expr_tcga[test_ids]
expr_me   = sample_zscore(meta_expr)
expr_gs   = sample_zscore(gse_expr)

clean_genes = [g for g in expr_tr.index if not (
    g.startswith("LOC") or g.startswith("LINC") or g.startswith("MIR") or
    g.startswith("SNOR") or (g.endswith("P") and any(ch.isdigit() for ch in g))
)]

path_tr = pd.read_parquet("data/processed/ssgsea_tcga_train.parquet").fillna(0.0)
path_te = pd.read_parquet("data/processed/ssgsea_tcga_test.parquet").fillna(0.0)
path_tcga = pd.concat([path_tr, path_te], axis=1)[all_tcga_ids]
path_me = pd.read_parquet("data/processed/ssgsea_metabric.parquet").fillna(0.0)
path_gs = pd.read_parquet("data/processed/ssgsea_gse20685.parquet").fillna(0.0)
common_paths = sorted(list(set(path_tr.index) & set(path_me.index) & set(path_gs.index)))

# Univariate Cox Screening Strictly on TCGA-Train
raw_var = tcga_expr.loc[clean_genes, train_ids].var(axis=1)
hv_genes = raw_var[raw_var >= raw_var.median()].index.tolist()
y_train_df = tcga_clin.loc[train_ids, ["OS_MONTHS", "OS"]].copy()
univ_rows = []
for g in hv_genes:
    tmp = y_train_df.copy(); tmp["g"] = expr_tr.loc[g].values
    try:
        cph = CoxPHFitter().fit(tmp, duration_col="OS_MONTHS", event_col="OS")
        univ_rows.append((g, cph.summary.loc["g", "p"]))
    except Exception:
        continue
cand_genes = pd.DataFrame(univ_rows, columns=["Gene", "P"]).sort_values("P").query("P < 0.01")["Gene"].head(120).tolist()

def to_surv(df, event_col="OS"):
    return Surv.from_arrays(event=df[event_col].astype(bool), time=df["OS_MONTHS"].astype(float))

y_tr = to_surv(tcga_clin.loc[train_ids])
y_te = to_surv(tcga_clin.loc[test_ids])
y_me_os  = to_surv(meta_clin, "OS")
y_me_dss = to_surv(meta_clin, "DSS")
y_gs = to_surv(gse_clin)

def fit_tune_coxnet(Xtr_raw, y_train_surv, os_events, l1_ratio=0.75):
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), index=Xtr_raw.index, columns=Xtr_raw.columns)
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.05, max_iter=100000).fit(Xtr, y_train_surv)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_a, best_c = coxnet.alphas_[len(coxnet.alphas_)//2], -1
    for a in coxnet.alphas_:
        cs = []
        for ti, vi in cv.split(Xtr, os_events):
            m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[a]).fit(Xtr.iloc[ti], y_train_surv[ti])
            cs.append(concordance_index_censored(y_train_surv[vi]["event"], y_train_surv[vi]["time"], m.predict(Xtr.iloc[vi]))[0])
        if np.mean(cs) > best_c:
            best_c, best_a = np.mean(cs), a
    final_cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best_a], fit_baseline_model=True).fit(Xtr, y_train_surv)
    return final_cox, scaler

# ==============================================================================
# UPGRADE 1: HIERARCHICAL BLOCK-WISE LATE FUSION VS EARLY CONCATENATION
# ==============================================================================
os_tr_vec = tcga_clin.loc[train_ids, "OS"]

# Block 1: mRNA + Pathway Block
X_mrna_tr = pd.concat([expr_tr.loc[cand_genes].T, path_tr.loc[common_paths].T], axis=1)
X_mrna_te = pd.concat([expr_te.loc[cand_genes].T, path_te.loc[common_paths].T], axis=1)
X_mrna_me = pd.concat([expr_me.loc[cand_genes].T, path_me.loc[common_paths].T], axis=1)
m_mrna, sc_mrna = fit_tune_coxnet(X_mrna_tr, y_tr, os_tr_vec, l1_ratio=0.75)

# Block 2: CNA Block
m_cna, sc_cna = fit_tune_coxnet(tcga_cna.loc[train_ids], y_tr, os_tr_vec, l1_ratio=0.3)

# Block 3: SNV + TMB Block
m_snv, sc_snv = fit_tune_coxnet(tcga_snv.loc[train_ids], y_tr, os_tr_vec, l1_ratio=0.3)

# Meta-Learner Block Fusion (Late Fusion of Block Risk Scores)
Z_blk_tr = pd.DataFrame({
    "Block_mRNA_Path": m_mrna.predict(sc_mrna.transform(X_mrna_tr)),
    "Block_CNA":       m_cna.predict(sc_cna.transform(tcga_cna.loc[train_ids])),
    "Block_SNV":       m_snv.predict(sc_snv.transform(tcga_snv.loc[train_ids]))
}, index=train_ids)

Z_blk_me = pd.DataFrame({
    "Block_mRNA_Path": m_mrna.predict(sc_mrna.transform(X_mrna_me)),
    "Block_CNA":       m_cna.predict(sc_cna.transform(meta_cna)),
    "Block_SNV":       m_snv.predict(sc_snv.transform(meta_snv))
}, index=meta_clin.index)

meta_fusion, sc_meta = fit_tune_coxnet(Z_blk_tr, y_tr, os_tr_vec, l1_ratio=0.05)
print("\n=== BLOCK-WISE LATE FUSION WEIGHTS (LEARNED ON TCGA-TRAIN) ===")
for col, w in zip(Z_blk_tr.columns, meta_fusion.coef_.ravel()):
    print(f"  {col:18s}: weight = {w:.4f}")

# Early Concatenation 3-Omics Model + Exact Transparent Coefficients
X_early_tr = pd.concat([X_mrna_tr, tcga_cna.loc[train_ids], tcga_snv.loc[train_ids]], axis=1)
X_early_me = pd.concat([X_mrna_me, meta_cna, meta_snv], axis=1)
m_early, sc_early = fit_tune_coxnet(X_early_tr, y_tr, os_tr_vec, l1_ratio=0.75)

coef_df = pd.DataFrame({
    "Feature": X_early_tr.columns,
    "CoxNet_Beta": m_early.coef_.ravel(),
    "Hazard_Ratio_per_SD": np.exp(m_early.coef_.ravel())
}).query("abs(CoxNet_Beta) > 1e-5").sort_values("CoxNet_Beta", key=abs, ascending=False)
coef_df.to_csv("results/v3_tripod_gold/Table_Exact_Signature_Weights.csv", index=False)

# Clinical Covariates + Clinico-3Omics Fusion
def get_clin_df(clin_df, sids):
    c = pd.DataFrame(index=sids)
    c["Age"] = pd.to_numeric(clin_df.loc[sids, "Age"], errors="coerce").fillna(clin_df["Age"].median())
    if "Stage_Simple" in clin_df.columns:
        c["Stage"] = clin_df.loc[sids, "Stage_Simple"].map({"Stage_I": 1.0, "Stage_II": 2.0, "Stage_III_IV": 3.0, "Unknown": 2.0}).fillna(2.0)
    else:
        npi = pd.to_numeric(clin_df.loc[sids, "NPI"], errors="coerce").fillna(4.0)
        c["Stage"] = np.where(npi < 3.4, 1.0, np.where(npi <= 5.4, 2.0, 3.0))
    return c

clin_tr = get_clin_df(tcga_clin, train_ids)
clin_me = get_clin_df(meta_clin, meta_clin.index.tolist())
m_clin, sc_clin = fit_tune_coxnet(clin_tr, y_tr, os_tr_vec, l1_ratio=0.05)

X_cm_tr = clin_tr.copy(); X_cm_tr["MolRisk"] = m_early.predict(sc_early.transform(X_early_tr))
X_cm_me = clin_me.copy(); X_cm_me["MolRisk"] = m_early.predict(sc_early.transform(X_early_me))
m_cm, sc_cm = fit_tune_coxnet(X_cm_tr, y_tr, os_tr_vec, l1_ratio=0.05)

# ==============================================================================
# UPGRADE 2: INTEGRATED BRIER SCORE (IBS) & MODALITY DROPOUT STRESS TEST
# ==============================================================================
def eval_full_metrics(label, model, X_eval_scaled, y_eval_surv):
    preds = model.predict(X_eval_scaled)
    cidx = concordance_index_censored(y_eval_surv["event"], y_eval_surv["time"], preds)[0]
    vt = [12.0, 36.0, 60.0]
    km = y_eval_surv["time"] <= 120.0
    _, mauc = cumulative_dynamic_auc(y_tr, y_eval_surv[km], preds[km], vt)

    # Compute Integrated Brier Score (12 to 60 months)
    times_grid = np.linspace(12.0, 60.0, 25)
    surv_fns = model.predict_survival_function(X_eval_scaled[km])
    surv_probs = np.row_stack([fn(times_grid) for fn in surv_fns])
    ibs = integrated_brier_score(y_tr, y_eval_surv[km], surv_probs, times_grid)
    return {"Model": label, "C_Index": round(float(cidx), 4),
            "Mean_1_3_5yr_AUC": round(float(mauc), 4), "IBS_1_to_5yr": round(float(ibs), 4)}, preds

# Evaluate across OS and Breast-Cancer-Specific Survival (DSS) in METABRIC + Modality Dropout
dropout_rows = []
preds_store = {}

for lbl, mod, Xsc, yev in [
    ("Clinical-Only (METABRIC OS)", m_clin, sc_clin.transform(clin_me), y_me_os),
    ("Early-Fusion 3-Omics (METABRIC OS)", m_early, sc_early.transform(X_early_me), y_me_os),
    ("Block-Wise Late-Fusion 3-Omics (METABRIC OS)", meta_fusion, sc_meta.transform(Z_blk_me), y_me_os),
    ("Clinico-3Omics Fusion (METABRIC OS)", m_cm, sc_cm.transform(X_cm_me), y_me_os),
    ("Clinico-3Omics Fusion (METABRIC Breast-Cancer DSS)", m_cm, sc_cm.transform(X_cm_me), y_me_dss),
]:
    row, pr = eval_full_metrics(lbl, mod, Xsc, yev)
    dropout_rows.append(row)
    preds_store[lbl] = pr

# Modality Dropout Stress Tests on Early-Fusion 3-Omics (setting missing blocks to 0 after scaling)
X_me_scaled_full = pd.DataFrame(sc_early.transform(X_early_me), index=X_early_me.index, columns=X_early_me.columns)
cna_cols = [c for c in X_early_me.columns if c.startswith("CNA_")]
snv_cols = [c for c in X_early_me.columns if c.startswith("SNV_") or c == "LOG_TMB"]

for drop_lbl, zero_cols in [
    ("Dropout Test: Missing CNA Layer (mRNA+SNV only)", cna_cols),
    ("Dropout Test: Missing SNV Layer (mRNA+CNA only)", snv_cols),
    ("Dropout Test: Missing All DNA Layers (mRNA+Pathway only)", cna_cols + snv_cols),
]:
    X_drop = X_me_scaled_full.copy()
    X_drop[zero_cols] = 0.0
    row, _ = eval_full_metrics(drop_lbl, m_early, X_drop.values, y_me_os)
    dropout_rows.append(row)

tripod_df = pd.DataFrame(dropout_rows)
tripod_df.to_csv("results/v3_tripod_gold/Table_TRIPOD_Calibration_and_Dropout.csv", index=False)
print("\n=== TRIPOD+AI DISCRIMINATION, INTEGRATED BRIER SCORE & MODALITY DROPOUT (METABRIC) ===")
print(tripod_df.to_string(index=False))

# ==============================================================================
# UPGRADE 3: PAIRED BOOTSTRAP HYPOTHESIS TESTS (MATCHED INCREMENTAL VALUE)
# ==============================================================================
ONCOTYPE_GENES = [g for g in ["MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2", "MMP11", "CTSL2",
                  "ERBB2", "GRB7", "ESR1", "PGR", "BCL2", "SCUBE2", "GSTM1", "BAG1", "CD68"] if g in expr_tr.index]
m_onco, sc_onco = fit_tune_coxnet(expr_tr.loc[ONCOTYPE_GENES].T, y_tr, os_tr_vec, l1_ratio=0.2)
pr_onco_me = m_onco.predict(sc_onco.transform(expr_me.loc[ONCOTYPE_GENES].T))
pr_clin_me = preds_store["Clinical-Only (METABRIC OS)"]
pr_3om_me  = preds_store["Early-Fusion 3-Omics (METABRIC OS)"]
pr_cm_me   = preds_store["Clinico-3Omics Fusion (METABRIC OS)"]

def paired_bootstrap_delta(y_ev, score_new, score_ref, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y_ev)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_ev[idx]["event"].sum() == 0:
            continue
        c_new = concordance_index_censored(y_ev[idx]["event"], y_ev[idx]["time"], score_new[idx])[0]
        c_ref = concordance_index_censored(y_ev[idx]["event"], y_ev[idx]["time"], score_ref[idx])[0]
        deltas.append(c_new - c_ref)
    deltas = np.array(deltas)
    pval = 2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(pval)

boot_tests = []
for comp_name, s_new, s_ref, y_target in [
    ("3-Omics vs. Re-Trained Oncotype DX (METABRIC OS)", pr_3om_me, pr_onco_me, y_me_os),
    ("Clinico-3Omics vs. Re-Trained Oncotype DX (METABRIC OS)", pr_cm_me, pr_onco_me, y_me_os),
    ("Clinico-3Omics vs. Clinical-Only (METABRIC DSS)", pr_cm_me, pr_clin_me, y_me_dss),
]:
    dm, lo, hi, pv = paired_bootstrap_delta(y_target, s_new, s_ref)
    boot_tests.append({"Comparison": comp_name, "Delta_C_Index": round(dm, 4),
                       "CI_95": f"{lo:.4f} to {hi:.4f}", "Paired_Bootstrap_P": pv})

boot_df = pd.DataFrame(boot_tests)
boot_df.to_csv("results/v3_tripod_gold/Table_Paired_Bootstrap_Tests.csv", index=False)
print("\n=== PAIRED BOOTSTRAP INCREMENTAL VALUE TESTS (1,000 ITERATIONS) ===")
print(boot_df.to_string(index=False))

# ==============================================================================
# FIGURE 7: 5-YEAR CALIBRATION CURVE + MODALITY DROPOUT RESILIENCE
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

# Panel A: 5-Year Calibration Curve in METABRIC (Deciles of Predicted 5-Yr Survival)
surv_fns_cm = m_cm.predict_survival_function(sc_cm.transform(X_cm_me))
pred_5yr_surv = np.array([float(fn(60.0)) for fn in surv_fns_cm])
deciles = pd.qcut(pred_5yr_surv, q=10, duplicates="drop")

cal_x, cal_y, cal_err = [], [], []
for d_bin in deciles.categories:
    mask = (deciles == d_bin)
    cal_x.append(pred_5yr_surv[mask].mean())
    kmf = KaplanMeierFitter().fit(meta_clin.loc[mask, "OS_MONTHS"], meta_clin.loc[mask, "OS"])
    obs_5yr = float(kmf.predict(60.0))
    ci_df = kmf.confidence_interval_survival_function_
    nearest_t = ci_df.index[np.argmin(np.abs(ci_df.index - 60.0))]
    cal_y.append(obs_5yr)
    cal_err.append((obs_5yr - ci_df.loc[nearest_t].iloc[0], ci_df.loc[nearest_t].iloc[1] - obs_5yr))

cal_err = np.array(cal_err).T
axes[0].plot([0.45, 0.98], [0.45, 0.98], ls="--", color="gray", lw=1.2, label="Ideal Calibration")
axes[0].errorbar(cal_x, cal_y, yerr=cal_err, fmt="o-", color="#C0392B", ecolor="#2C3E50", capsize=3, lw=1.8, markersize=6, label="Clinico-3Omics (METABRIC)")
axes[0].set_xlabel("Predicted 5-Year Overall Survival Probability")
axes[0].set_ylabel("Observed Kaplan–Meier 5-Year Survival")
axes[0].set_title("A. 5-Year Survival Calibration (METABRIC, n = 1,865)")
axes[0].legend(frameon=False, loc="upper left")

# Panel B: Modality Dropout Stress Test Barplot
drop_plot_df = tripod_df[tripod_df["Model"].str.contains("Early-Fusion|Block-Wise|Dropout")].copy()
drop_plot_df["Short_Label"] = ["Early-Fusion\n(Full 3-Omics)", "Block-Wise\n(Late Fusion)",
                               "Missing CNA\n(mRNA+SNV)", "Missing SNV\n(mRNA+CNA)", "Missing DNA\n(mRNA Only)"]
sns.barplot(data=drop_plot_df, x="Short_Label", y="C_Index", palette=["#8E44AD", "#2980B9", "#E67E22", "#E67E22", "#95A5A6"],
            ax=axes[1], edgecolor="black", linewidth=0.7)
axes[1].set_ylim(0.50, 0.62)
axes[1].set_ylabel("External C-Index (METABRIC)")
axes[1].set_xlabel("")
axes[1].set_title("B. Multi-Omics Fusion & Missing-Modality Resilience")
for i, v in enumerate(drop_plot_df["C_Index"]):
    axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")

plt.tight_layout()
plt.savefig("results/v3_tripod_gold/Fig7_Calibration_and_ModalityDropout.png")
plt.savefig("results/v3_tripod_gold/Fig7_Calibration_and_ModalityDropout.pdf")
plt.close()

# ==============================================================================
# FIGURE 8: TUMOR IMMUNE MICROENVIRONMENT (TIME) & CHECKPOINT DECONVOLUTION
# ==============================================================================
# Compute validated single-sample immune cell & checkpoint scores in METABRIC
immune_signatures = {
    "CD8+ T-Cell Effector": ["CD8A", "CD8B", "GZMA", "GZMB", "PRF1", "IFNG"],
    "Cytolytic Activity (CYT)": ["GZMA", "PRF1"],
    "T-Cell Exhaustion / Checkpoints": ["PDCD1", "CD274", "CTLA4", "LAG3", "TIGIT", "HAVCR2"],
    "Regulatory T-Cells (Treg)": ["FOXP3", "IL2RA", "CCR8", "IKZF2"],
    "M2 Macrophage Polarization": ["CD163", "MRC1", "MSR1", "TGFB1", "IL10"],
    "Antigen Presentation (MHC-I/II)": ["HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "B2M"]
}

hi_mask = pr_3om_me >= np.median(pr_3om_me)
time_rows, time_plot = [], []
for sig_name, genes in immune_signatures.items():
    avail = [g for g in genes if g in expr_me.index]
    if not avail:
        continue
    scores_sig = expr_me.loc[avail].mean(axis=0).values
    hi_v, lo_v = scores_sig[hi_mask], scores_sig[~hi_mask]
    _, pv = mannwhitneyu(hi_v, lo_v)
    time_rows.append({"TIME_Signature": sig_name, "HighRisk_MeanZ": round(float(hi_v.mean()), 4),
                      "LowRisk_MeanZ": round(float(lo_v.mean()), 4), "P_value": pv})
    for v in hi_v:
        time_plot.append({"Signature": sig_name, "Risk": "High-Risk", "Score": v})
    for v in lo_v:
        time_plot.append({"Signature": sig_name, "Risk": "Low-Risk", "Score": v})

time_df = pd.DataFrame(time_rows)
time_df["FDR"] = multipletests(time_df["P_value"], method="fdr_bh")[1]
time_df.to_csv("results/v3_tripod_gold/Table_TIME_Immune_Deconvolution.csv", index=False)
print("\n=== TUMOR IMMUNE MICROENVIRONMENT (TIME) & CHECKPOINT PROFILING (METABRIC) ===")
print(time_df.to_string(index=False))

plt.figure(figsize=(10.5, 5.0))
sns.boxplot(data=pd.DataFrame(time_plot), x="Signature", y="Score", hue="Risk",
            palette={"Low-Risk": "#1F77B4", "High-Risk": "#D62728"}, fliersize=1.2, linewidth=0.9)
plt.xticks(rotation=22, ha="right")
plt.ylabel("Normalized Signature Activity (Mean Z-Score)")
plt.xlabel("")
plt.title("Tumor Immune Microenvironment (TIME) & Checkpoint Expression by Multi-Omics Risk")
plt.legend(frameon=False, title="")
plt.tight_layout()
plt.savefig("results/v3_tripod_gold/Fig8_TIME_Immune_Checkpoints.png")
plt.savefig("results/v3_tripod_gold/Fig8_TIME_Immune_Checkpoints.pdf")
plt.close()

print("\nAll TRIPOD+AI v3 Gold-Standard Tables and Figures (Fig7 & Fig8) Generated Successfully!")