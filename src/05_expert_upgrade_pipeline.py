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
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc, brier_score

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

os.makedirs("results/v2_expert", exist_ok=True)

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

# 2. Extract True 3rd Omics Layer: GISTIC2 Copy-Number Alterations (CNA)
print("--- Extracting GISTIC2 Copy-Number Alterations (TCGA-BRCA & METABRIC) ---")
CNA_DRIVERS = ["ERBB2", "MYC", "CCND1", "FGFR1", "MDM2", "CDK4", "CDK6",
               "PIK3CA", "AKT1", "PTEN", "RB1", "TP53", "CDKN2A", "BRCA1", "BRCA2"]

def load_cna_matrix(path, sample_list):
    cna_raw = pd.read_csv(path, sep="\t").dropna(subset=["Hugo_Symbol"])
    cna_raw = cna_raw.groupby("Hugo_Symbol").mean(numeric_only=True)
    out = pd.DataFrame(index=sample_list)
    for g in CNA_DRIVERS:
        if g in cna_raw.index:
            s = cna_raw.loc[g].reindex(sample_list).fillna(0.0)
            out[f"CNA_{g}"] = s.values
        else:
            out[f"CNA_{g}"] = 0.0
    return out.fillna(0.0)

tcga_cna = load_cna_matrix("data/raw/brca_tcga_pan_can_atlas_2018/data_cna.txt", all_tcga_ids)
meta_cna = load_cna_matrix("data/raw/brca_metabric/data_cna.txt", meta_clin.index.tolist())
print(f"Extracted {tcga_cna.shape[1]} GISTIC2 CNA driver features across TCGA-BRCA and METABRIC.")

# 3. Sample-Wise Platform-Invariant Z-Score + Curated Protein-Coding Filter
def sample_zscore(df):
    z = df.sub(df.mean(axis=0), axis=1).div(df.std(axis=0).replace(0, 1), axis=1)
    return z.fillna(0.0)

expr_tcga_all = sample_zscore(tcga_expr[all_tcga_ids])
expr_tr = expr_tcga_all[train_ids]
expr_te = expr_tcga_all[test_ids]
expr_me = sample_zscore(meta_expr)
expr_gs = sample_zscore(gse_expr)

# Filter out pseudogenes / non-coding LOC/LINC artifacts so only clean protein-coding genes enter
clean_genes = [g for g in expr_tr.index if not (
    g.startswith("LOC") or g.startswith("LINC") or g.startswith("MIR") or
    g.startswith("SNOR") or g.endswith("P") and any(ch.isdigit() for ch in g)
)]
print(f"Curated Protein-Coding Gene Universe: {len(clean_genes)} / {expr_tr.shape[0]} genes")

# Load cached Hallmark ssGSEA scores
path_tr = pd.read_parquet("data/processed/ssgsea_tcga_train.parquet").fillna(0.0)
path_te = pd.read_parquet("data/processed/ssgsea_tcga_test.parquet").fillna(0.0)
path_tcga_all = pd.concat([path_tr, path_te], axis=1)[all_tcga_ids]
path_me = pd.read_parquet("data/processed/ssgsea_metabric.parquet").fillna(0.0)
path_gs = pd.read_parquet("data/processed/ssgsea_gse20685.parquet").fillna(0.0)
common_paths = sorted(list(set(path_tr.index) & set(path_me.index) & set(path_gs.index)))

# 4. Somatic SNV + CNA Genomic Feature Matrix
SNV_DRIVERS = ["TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "KMT2C", "PTEN", "AKT1", "RB1", "BRCA1", "BRCA2"]
def build_genomic_3omics(mut_df, cna_df, clin_df, sids):
    g_df = pd.DataFrame(index=sids)
    for g in SNV_DRIVERS:
        g_df[f"SNV_{g}"] = mut_df.loc[g, sids].fillna(0).values if g in mut_df.index else 0
    g_df["LOG_TMB"] = np.log1p(clin_df.loc[sids, "TMB"].fillna(0).values)
    return pd.concat([g_df, cna_df.loc[sids]], axis=1).fillna(0.0)

gen_tcga_all = build_genomic_3omics(tcga_mut, tcga_cna, tcga_clin, all_tcga_ids)
gen_tr = gen_tcga_all.loc[train_ids]
gen_te = gen_tcga_all.loc[test_ids]
gen_me = build_genomic_3omics(meta_mut, meta_cna, meta_clin, meta_clin.index.tolist())

# 5. Clinical Covariates (Age + Stage/NPI)
def build_clin(clin_df, sids):
    c_df = pd.DataFrame(index=sids)
    c_df["Age"] = pd.to_numeric(clin_df.loc[sids, "Age"], errors="coerce").fillna(clin_df["Age"].median())
    if "Stage_Simple" in clin_df.columns:
        smap = {"Stage_I": 1.0, "Stage_II": 2.0, "Stage_III_IV": 3.0, "Unknown": 2.0}
        c_df["Stage_Proxy"] = clin_df.loc[sids, "Stage_Simple"].map(smap).fillna(2.0)
    elif "NPI" in clin_df.columns:
        # Map NPI (<3.4 -> 1, 3.4-5.4 -> 2, >5.4 -> 3) onto 1-3 clinical severity scale
        npi = pd.to_numeric(clin_df.loc[sids, "NPI"], errors="coerce").fillna(4.0)
        c_df["Stage_Proxy"] = np.where(npi < 3.4, 1.0, np.where(npi <= 5.4, 2.0, 3.0))
    else:
        c_df["Stage_Proxy"] = 2.0
    return c_df

clin_tcga_all = build_clin(tcga_clin, all_tcga_ids)
clin_tr = clin_tcga_all.loc[train_ids]
clin_te = clin_tcga_all.loc[test_ids]
clin_me = build_clin(meta_clin, meta_clin.index.tolist())
clin_gs = build_clin(gse_clin, gse_clin.index.tolist())

# 6. Univariate Cox Screening Strictly on TCGA-Train (High-Variance Protein-Coding Genes)
raw_var = tcga_expr.loc[clean_genes, train_ids].var(axis=1)
hv_genes = raw_var[raw_var >= raw_var.median()].index.tolist()

y_train_df = tcga_clin.loc[train_ids, ["OS_MONTHS", "OS"]].copy()
univ_rows = []
for g in hv_genes:
    vals = expr_tr.loc[g].values
    if np.std(vals) < 1e-5:
        continue
    tmp = y_train_df.copy()
    tmp["g"] = vals
    try:
        cph = CoxPHFitter().fit(tmp, duration_col="OS_MONTHS", event_col="OS")
        univ_rows.append((g, cph.summary.loc["g", "exp(coef)"], cph.summary.loc["g", "p"]))
    except Exception:
        continue

univ_df = pd.DataFrame(univ_rows, columns=["Gene", "HR", "P_value"]).sort_values("P_value")
univ_df["FDR"] = multipletests(univ_df["P_value"], method="fdr_bh")[1]
cand_genes = univ_df[univ_df["P_value"] < 0.01]["Gene"].head(120).tolist()
print(f"Protein-Coding Univariate Cox Genes (p < 0.01): {(univ_df['P_value'] < 0.01).sum()} (FDR < 0.05: {(univ_df['FDR'] < 0.05).sum()})")

# Assemble 3-Omics Feature Matrices
X_trans_all = pd.concat([expr_tcga_all.loc[cand_genes].T, path_tcga_all.loc[common_paths].T], axis=1)
X_trans_tr  = X_trans_all.loc[train_ids]
X_trans_te  = X_trans_all.loc[test_ids]
X_trans_me  = pd.concat([expr_me.loc[cand_genes].T, path_me.loc[common_paths].T], axis=1)
X_trans_gs  = pd.concat([expr_gs.loc[cand_genes].T, path_gs.loc[common_paths].T], axis=1)

X_3omics_all = pd.concat([X_trans_all, gen_tcga_all], axis=1)
X_3omics_tr  = X_3omics_all.loc[train_ids]
X_3omics_te  = X_3omics_all.loc[test_ids]
X_3omics_me  = pd.concat([X_trans_me, gen_me], axis=1)

def to_surv(df):
    return Surv.from_arrays(event=df["OS"].astype(bool), time=df["OS_MONTHS"].astype(float))

y_all = to_surv(tcga_clin.loc[all_tcga_ids])
y_tr  = to_surv(tcga_clin.loc[train_ids])
y_te  = to_surv(tcga_clin.loc[test_ids])
y_me  = to_surv(meta_clin)
y_gs  = to_surv(gse_clin)

def bootstrap_cindex(y_true, scores, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx]["event"].sum() == 0:
            continue
        vals.append(concordance_index_censored(y_true[idx]["event"], y_true[idx]["time"], scores[idx])[0])
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

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
            p = m.predict(Xtr.iloc[vi])
            cs.append(concordance_index_censored(y_train_surv[vi]["event"], y_train_surv[vi]["time"], p)[0])
        if np.mean(cs) > best_c:
            best_c, best_a = np.mean(cs), a
    final_cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best_a], fit_baseline_model=True).fit(Xtr, y_train_surv)
    return final_cox, scaler, best_a

# Train 3-Omics CoxNet on TCGA-Train
cox_3omics, scaler_3omics, best_alpha = fit_tune_coxnet(X_3omics_tr, y_tr, tcga_clin.loc[train_ids, "OS"], l1_ratio=0.75)
sel_3omics_feats = X_3omics_tr.columns[np.abs(cox_3omics.coef_.ravel()) > 1e-5].tolist()
print(f"\nSelected 3-Omics Signature Features ({len(sel_3omics_feats)}): {sel_3omics_feats}")

# Also compute 5-Fold Out-of-Fold CV predictions across ALL 994 TCGA-BRCA patients
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_3omics = np.zeros(len(all_tcga_ids))
oof_clin_3omics = np.zeros(len(all_tcga_ids))
os_all_vec = tcga_clin.loc[all_tcga_ids, "OS"].values

for ti, vi in skf.split(X_3omics_all, os_all_vec):
    sc_f = StandardScaler()
    X_f_tr = sc_f.fit_transform(X_3omics_all.iloc[ti])
    X_f_va = sc_f.transform(X_3omics_all.iloc[vi])
    m_f = CoxnetSurvivalAnalysis(l1_ratio=0.75, alphas=[best_alpha]).fit(X_f_tr, y_all[ti])
    oof_3omics[vi] = m_f.predict(X_f_va)

    # Clinico-3Omics fusion fold
    cm_tr = clin_tcga_all.iloc[ti].copy(); cm_tr["MolRisk"] = m_f.predict(X_f_tr)
    cm_va = clin_tcga_all.iloc[vi].copy(); cm_va["MolRisk"] = oof_3omics[vi]
    sc_cm = StandardScaler()
    m_cm = CoxnetSurvivalAnalysis(l1_ratio=0.1, alphas=[0.02]).fit(sc_cm.fit_transform(cm_tr), y_all[ti])
    oof_clin_3omics[vi] = m_cm.predict(sc_cm.transform(cm_va))

# Score Locked Holdout (TCGA-Test) & External Cohorts (METABRIC)
score_3omics_tr = cox_3omics.predict(scaler_3omics.transform(X_3omics_tr))
score_3omics_te = cox_3omics.predict(scaler_3omics.transform(X_3omics_te))
score_3omics_me = cox_3omics.predict(scaler_3omics.transform(X_3omics_me))

# Clinico-3Omics Fusion Model (trained strictly on TCGA-Train)
X_cm3_tr = clin_tr.copy(); X_cm3_tr["MolRisk"] = score_3omics_tr
X_cm3_te = clin_te.copy(); X_cm3_te["MolRisk"] = score_3omics_te
X_cm3_me = clin_me.copy(); X_cm3_me["MolRisk"] = score_3omics_me
cox_cm3, scaler_cm3, _ = fit_tune_coxnet(X_cm3_tr, y_tr, tcga_clin.loc[train_ids, "OS"], l1_ratio=0.1)

score_cm3_tr = cox_cm3.predict(scaler_cm3.transform(X_cm3_tr))
score_cm3_te = cox_cm3.predict(scaler_cm3.transform(X_cm3_te))
score_cm3_me = cox_cm3.predict(scaler_cm3.transform(X_cm3_me))

# 7. Head-to-Head Benchmark vs. Commercial Signatures (Oncotype DX, MammaPrint, PAM50)
ONCOTYPE_GENES = ["MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2", "MMP11", "CTSL2",
                  "ERBB2", "GRB7", "ESR1", "PGR", "BCL2", "SCUBE2", "GSTM1", "BAG1", "CD68"]
PAM50_GENES = ["UBE2T", "BIRC5", "NUF2", "CDC6", "CCNB1", "TYMS", "MYBL2", "CEP55", "MELK",
               "NDC80", "RRM2", "UBE2C", "CENPF", "PTTG1", "EXO1", "ORC6", "ANLN", "CCNE1",
               "CDC20", "MKI67", "KIF2C", "ACTR3B", "MYC", "EGFR", "KRT5", "PHGDH", "CDH3",
               "MIA", "KRT17", "FOXC1", "SFRP1", "KRT14", "ESR1", "SLC39A6", "BAG1", "MAPT",
               "PGR", "CXXC5", "MLPH", "BCL2", "MDM2", "NAT1", "FOXA1", "BLVRA", "MMP11",
               "GPR160", "FGFR4", "GRB7", "TMEM45B", "ERBB2"]

def eval_benchmark_gene_set(label, gene_list):
    avail = [g for g in gene_list if g in expr_tr.index and g in expr_me.index and g in expr_gs.index]
    m_sig, sc_sig, _ = fit_tune_coxnet(expr_tr.loc[avail].T, y_tr, tcga_clin.loc[train_ids, "OS"], l1_ratio=0.2)
    out_rows = []
    for cname, Xev, yev in [("TCGA-Test", expr_te.loc[avail].T, y_te),
                            ("METABRIC", expr_me.loc[avail].T, y_me),
                            ("GSE20685", expr_gs.loc[avail].T, y_gs)]:
        pr = m_sig.predict(sc_sig.transform(Xev))
        cm, lo, hi = bootstrap_cindex(yev, pr)
        max_t = min(np.percentile(y_tr["time"], 90), np.percentile(yev["time"], 90))
        vt = [t for t in [12.0, 36.0, 60.0] if t < max_t and t > yev["time"].min()]
        km = yev["time"] <= max_t
        _, mauc = cumulative_dynamic_auc(y_tr, yev[km], pr[km], vt)
        out_rows.append({"Model": f"{label} ({len(avail)} genes)", "Cohort": cname,
                         "C_Index": round(cm, 4), "CI_95": f"{lo:.4f}-{hi:.4f}", "Mean_AUC": round(float(mauc), 4)})
    return out_rows

h2h_rows = []
h2h_rows += eval_benchmark_gene_set("OncotypeDX-Panel", ONCOTYPE_GENES)
h2h_rows += eval_benchmark_gene_set("PAM50-Panel", PAM50_GENES)

# Add Clinical-Only, 3-Omics Molecular, and Clinico-3Omics Fusion
for label, preds_dict in [
    ("Clinical-Only (Age+Stage)", {"TCGA-Test": fit_tune_coxnet(clin_tr, y_tr, tcga_clin.loc[train_ids, "OS"], 0.1)[0].predict(fit_tune_coxnet(clin_tr, y_tr, tcga_clin.loc[train_ids, "OS"], 0.1)[1].transform(clin_te)),
                                   "METABRIC":  fit_tune_coxnet(clin_tr, y_tr, tcga_clin.loc[train_ids, "OS"], 0.1)[0].predict(fit_tune_coxnet(clin_tr, y_tr, tcga_clin.loc[train_ids, "OS"], 0.1)[1].transform(clin_me))}),
    ("3-Omics-Signature (CNA+SNV+mRNA)", {"TCGA-Full-5Fold-OOF (n=994)": oof_3omics, "TCGA-Test": score_3omics_te, "METABRIC": score_3omics_me}),
    ("Clinico-3Omics-Fusion", {"TCGA-Full-5Fold-OOF (n=994)": oof_clin_3omics, "TCGA-Test": score_cm3_te, "METABRIC": score_cm3_me})
]:
    for cname, pr in preds_dict.items():
        yev = y_all if "994" in cname else (y_te if cname == "TCGA-Test" else y_me)
        cm, lo, hi = bootstrap_cindex(yev, pr)
        max_t = min(np.percentile(y_tr["time"], 90), np.percentile(yev["time"], 90))
        vt = [t for t in [12.0, 36.0, 60.0] if t < max_t and t > yev["time"].min()]
        km = yev["time"] <= max_t
        _, mauc = cumulative_dynamic_auc(y_tr, yev[km], pr[km], vt)
        h2h_rows.append({"Model": label, "Cohort": cname, "C_Index": round(cm, 4), "CI_95": f"{lo:.4f}-{hi:.4f}", "Mean_AUC": round(float(mauc), 4)})

h2h_df = pd.DataFrame(h2h_rows)
h2h_df.to_csv("results/v2_expert/Table_HeadToHead_Benchmark.csv", index=False)
print("\n=== HEAD-TO-HEAD BENCHMARK VS. COMMERCIAL SIGNATURES (v2) ===")
print(h2h_df.to_string(index=False))

# 8. Strict Zero-Leakage Locked Cutoff KM Stratification & Subtype Subgroup Analysis in METABRIC
locked_cutoff = np.median(score_cm3_tr)
print(f"\nLocked TCGA-Train Median Cutoff for Clinico-3Omics: {locked_cutoff:.4f}")

meta_eval = meta_clin.copy()
meta_eval["Score_CM3"] = score_cm3_me
meta_eval["Score_3Omics"] = score_3omics_me
meta_eval["Risk_Locked"] = np.where(meta_eval["Score_CM3"] >= locked_cutoff, "High-Risk", "Low-Risk")

lr_locked = logrank_test(
    meta_eval.loc[meta_eval["Risk_Locked"] == "High-Risk", "OS_MONTHS"],
    meta_eval.loc[meta_eval["Risk_Locked"] == "Low-Risk", "OS_MONTHS"],
    event_observed_A=meta_eval.loc[meta_eval["Risk_Locked"] == "High-Risk", "OS"],
    event_observed_B=meta_eval.loc[meta_eval["Risk_Locked"] == "Low-Risk", "OS"]
)
print(f"METABRIC Strict Locked-Cutoff Log-Rank p = {lr_locked.p_value:.2e} (High-Risk n={(meta_eval['Risk_Locked']=='High-Risk').sum()}, Low-Risk n={(meta_eval['Risk_Locked']=='Low-Risk').sum()})")

# Clinical Receptor Subgroup Analysis (ER+/HER2-, HER2+, TNBC)
sub_rows = []
er_col = [c for c in meta_eval.columns if "ER_IHC" in c.upper() or "ER_STATUS" in c.upper()][0]
her2_col = [c for c in meta_eval.columns if "HER2" in c.upper()][0]

meta_eval["Clinical_Subgroup"] = np.where(
    meta_eval[her2_col].astype(str).str.contains("Pos|Gain", case=False), "HER2-Positive",
    np.where(meta_eval[er_col].astype(str).str.contains("Pos", case=False), "ER+/HER2- (Luminal)", "Triple-Negative (TNBC)")
)

for sg, df_sg in meta_eval.groupby("Clinical_Subgroup"):
    cph_sg = CoxPHFitter()
    tmp_sg = df_sg[["OS_MONTHS", "OS"]].copy()
    tmp_sg["Risk_Z"] = (df_sg["Score_3Omics"] - df_sg["Score_3Omics"].mean()) / df_sg["Score_3Omics"].std()
    cph_sg.fit(tmp_sg, duration_col="OS_MONTHS", event_col="OS")
    hr = cph_sg.summary.loc["Risk_Z", "exp(coef)"]
    lo = cph_sg.summary.loc["Risk_Z", "exp(coef) lower 95%"]
    hi = cph_sg.summary.loc["Risk_Z", "exp(coef) upper 95%"]
    pv = cph_sg.summary.loc["Risk_Z", "p"]
    sub_rows.append({"Subgroup": sg, "N": len(df_sg), "Events": int(df_sg["OS"].sum()),
                     "HR_per_SD": round(hr, 3), "CI_95": f"{lo:.3f}-{hi:.3f}", "P_value": pv})

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv("results/v2_expert/Table_Subgroup_Stratification_METABRIC.csv", index=False)
print("\n=== SUBGROUP PROGNOSTIC VALIDATION IN METABRIC (ER+/HER2-, HER2+, TNBC) ===")
print(sub_df.to_string(index=False))

# 9. 5-Year Decision Curve Analysis (Net Benefit) Figure in METABRIC
# Evaluate 5-year mortality probability from CoxNet baseline survival
def predict_5yr_risk(cox_model, X_scaled, t_horizon=60.0):
    surv_funcs = cox_model.predict_survival_function(X_scaled)
    risks = []
    for fn in surv_funcs:
        t_eval = min(t_horizon, fn.x.max())
        risks.append(1.0 - float(fn(t_eval)))
    return np.array(risks)

p5_clin = predict_5yr_risk(fit_tune_coxnet(clin_tr, y_tr, tcga_clin.loc[train_ids, "OS"], 0.1)[0],
                           fit_tune_coxnet(clin_tr, y_tr, tcga_clin.loc[train_ids, "OS"], 0.1)[1].transform(clin_me))
p5_cm3  = predict_5yr_risk(cox_cm3, scaler_cm3.transform(X_cm3_me))

# Filter patients with known 5-year status (died <= 60 mo OR followed > 60 mo)
known_5yr = ((meta_eval["OS"] == 1) & (meta_eval["OS_MONTHS"] <= 60.0)) | (meta_eval["OS_MONTHS"] > 60.0)
y_5yr = ((meta_eval.loc[known_5yr, "OS"] == 1) & (meta_eval.loc[known_5yr, "OS_MONTHS"] <= 60.0)).astype(int).values
r_clin = p5_clin[known_5yr]
r_cm3  = p5_cm3[known_5yr]

thresholds = np.linspace(0.05, 0.55, 51)
nb_all, nb_clin, nb_cm3 = [], [], []
N_5 = len(y_5yr)
prev = y_5yr.mean()

for pt in thresholds:
    w = pt / (1.0 - pt)
    nb_all.append(prev - (1.0 - prev) * w)
    tp_c = np.sum((r_clin >= pt) & (y_5yr == 1))
    fp_c = np.sum((r_clin >= pt) & (y_5yr == 0))
    nb_clin.append((tp_c / N_5) - (fp_c / N_5) * w)

    tp_m = np.sum((r_cm3 >= pt) & (y_5yr == 1))
    fp_m = np.sum((r_cm3 >= pt) & (y_5yr == 0))
    nb_cm3.append((tp_m / N_5) - (fp_m / N_5) * w)

plt.figure(figsize=(7.5, 5.0))
plt.plot(thresholds * 100, nb_cm3, color="#C0392B", lw=2.4, label="Clinico-3Omics Fusion Model")
plt.plot(thresholds * 100, nb_clin, color="#2980B9", lw=2.0, ls="--", label="Clinical Model (Age + Stage)")
plt.plot(thresholds * 100, nb_all, color="#7F8C8D", lw=1.3, ls=":", label="Treat All")
plt.axhline(0.0, color="black", lw=1.0, label="Treat None")
plt.ylim(-0.02, prev + 0.04)
plt.xlabel("5-Year Mortality Risk Threshold Probability (%)")
plt.ylabel("Net Clinical Benefit")
plt.title(f"5-Year Decision Curve Analysis (METABRIC, n = {N_5})")
plt.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("results/v2_expert/Fig6_Decision_Curve_Analysis.png")
plt.savefig("results/v2_expert/Fig6_Decision_Curve_Analysis.pdf")
plt.close()

print("\nExpert Upgrade (v2) Complete! Check results/v2_expert/ for all upgraded tables and Fig6 DCA.")