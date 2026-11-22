import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gseapy as gp
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# 1. Load Harmonized Datasets
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

# 2. Sample-wise Z-score normalization (platform-invariant)
def sample_zscore(df):
    z = df.sub(df.mean(axis=0), axis=1).div(df.std(axis=0).replace(0, 1), axis=1)
    return z.fillna(0.0)

expr_tr = sample_zscore(tcga_expr[train_ids])
expr_te = sample_zscore(tcga_expr[test_ids])
expr_me = sample_zscore(meta_expr)
expr_gs = sample_zscore(gse_expr)

# Load cached ssGSEA Hallmark pathways
path_tr = pd.read_parquet("data/processed/ssgsea_tcga_train.parquet").fillna(0.0)
path_te = pd.read_parquet("data/processed/ssgsea_tcga_test.parquet").fillna(0.0)
path_me = pd.read_parquet("data/processed/ssgsea_metabric.parquet").fillna(0.0)
path_gs = pd.read_parquet("data/processed/ssgsea_gse20685.parquet").fillna(0.0)
common_paths = sorted(list(set(path_tr.index) & set(path_me.index) & set(path_gs.index)))

# 3. Genomic Features (11 Drivers + Log-TMB)
DRIVER_GENES = ["TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "KMT2C", "PTEN", "AKT1", "RB1", "BRCA1", "BRCA2"]
def build_genomic_df(mut_df, clin_df, sample_list):
    g_df = pd.DataFrame(index=sample_list)
    for g in DRIVER_GENES:
        g_df[f"MUT_{g}"] = mut_df.loc[g, sample_list].fillna(0).values if g in mut_df.index else 0
    g_df["LOG_TMB"] = np.log1p(clin_df.loc[sample_list, "TMB"].fillna(0).values)
    return g_df.fillna(0.0)

gen_tr = build_genomic_df(tcga_mut, tcga_clin, train_ids)
gen_te = build_genomic_df(tcga_mut, tcga_clin, test_ids)
gen_me = build_genomic_df(meta_mut, meta_clin, meta_clin.index.tolist())

# 4. Clinical Features (Age + Numeric Stage where available)
def build_clin_feats(clin_df, sample_list):
    c_df = pd.DataFrame(index=sample_list)
    c_df["Age"] = pd.to_numeric(clin_df.loc[sample_list, "Age"], errors="coerce").fillna(clin_df["Age"].median())
    if "Stage_Simple" in clin_df.columns:
        stage_map = {"Stage_I": 1.0, "Stage_II": 2.0, "Stage_III_IV": 3.0, "Unknown": 2.0}
        c_df["Stage_Num"] = clin_df.loc[sample_list, "Stage_Simple"].map(stage_map).fillna(2.0)
    elif "TUMOR_STAGE" in clin_df.columns:
        c_df["Stage_Num"] = pd.to_numeric(clin_df.loc[sample_list, "TUMOR_STAGE"], errors="coerce").clip(1, 3).fillna(2.0)
    else:
        c_df["Stage_Num"] = 2.0
    return c_df

clin_X_tr = build_clin_feats(tcga_clin, train_ids)
clin_X_te = build_clin_feats(tcga_clin, test_ids)
clin_X_me = build_clin_feats(meta_clin, meta_clin.index.tolist())
clin_X_gs = build_clin_feats(gse_clin, gse_clin.index.tolist())

# 5. High-Variance + FDR-Controlled Univariate Cox Screening on TCGA-Train
# Keep top 50% most variable genes in raw TCGA-Train so low-expression noise is removed
raw_var = tcga_expr[train_ids].var(axis=1)
high_var_genes = raw_var[raw_var >= raw_var.median()].index.tolist()

y_train_df = tcga_clin.loc[train_ids, ["OS_MONTHS", "OS"]].copy()
univ_results = []
for gene in high_var_genes:
    vals = expr_tr.loc[gene].values
    if np.std(vals) < 1e-5:
        continue
    df_tmp = y_train_df.copy()
    df_tmp["gene"] = vals
    try:
        cph = CoxPHFitter()
        cph.fit(df_tmp, duration_col="OS_MONTHS", event_col="OS")
        pval = cph.summary.loc["gene", "p"]
        hr   = cph.summary.loc["gene", "exp(coef)"]
        univ_results.append((gene, hr, pval))
    except Exception:
        continue

univ_df = pd.DataFrame(univ_results, columns=["Gene", "HR", "P_value"]).sort_values("P_value")
univ_df["FDR"] = multipletests(univ_df["P_value"], method="fdr_bh")[1]
univ_df.to_csv("results/tables/TableS1_Univariate_Cox_Genes.csv", index=False)

cand_genes = univ_df[univ_df["P_value"] < 0.01]["Gene"].head(120).tolist()
print(f"High-Variance Univariate Cox Genes (p < 0.01): {len(univ_df[univ_df['P_value'] < 0.01])} (FDR < 0.05: {(univ_df['FDR'] < 0.05).sum()}) -> Top {len(cand_genes)} selected")

# Assemble Raw Modality Matrices
X_trans_tr = pd.concat([expr_tr.loc[cand_genes].T, path_tr.loc[common_paths].T], axis=1)
X_trans_te = pd.concat([expr_te.loc[cand_genes].T, path_te.loc[common_paths].T], axis=1)
X_trans_me = pd.concat([expr_me.loc[cand_genes].T, path_me.loc[common_paths].T], axis=1)
X_trans_gs = pd.concat([expr_gs.loc[cand_genes].T, path_gs.loc[common_paths].T], axis=1)

X_multi_tr = pd.concat([X_trans_tr, gen_tr], axis=1)
X_multi_te = pd.concat([X_trans_te, gen_te], axis=1)
X_multi_me = pd.concat([X_trans_me, gen_me], axis=1)

def to_surv(df):
    return Surv.from_arrays(event=df["OS"].astype(bool), time=df["OS_MONTHS"].astype(float))

y_tr = to_surv(tcga_clin.loc[train_ids])
y_te = to_surv(tcga_clin.loc[test_ids])
y_me = to_surv(meta_clin)
y_gs = to_surv(gse_clin)

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

def fit_and_eval_modality(name, Xtr_raw, Xte_raw, Xme_raw, Xgs_raw=None, l1_ratio=0.7):
    # Feature-wise StandardScaler fitted STRICTLY on TCGA-Train
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(Xtr_raw), index=Xtr_raw.index, columns=Xtr_raw.columns)
    Xte = pd.DataFrame(scaler.transform(Xte_raw), index=Xte_raw.index, columns=Xte_raw.columns)
    Xme = pd.DataFrame(scaler.transform(Xme_raw), index=Xme_raw.index, columns=Xme_raw.columns)
    Xgs = pd.DataFrame(scaler.transform(Xgs_raw), index=Xgs_raw.index, columns=Xgs_raw.columns) if Xgs_raw is not None else None

    coxnet = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.05, max_iter=100000).fit(Xtr, y_tr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_a, best_c = coxnet.alphas_[len(coxnet.alphas_)//2], -1
    for a in coxnet.alphas_:
        cs = []
        for ti, vi in cv.split(Xtr, tcga_clin.loc[train_ids, "OS"]):
            m = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[a]).fit(Xtr.iloc[ti], y_tr[ti])
            p = m.predict(Xtr.iloc[vi])
            cs.append(concordance_index_censored(y_tr[vi]["event"], y_tr[vi]["time"], p)[0])
        if np.mean(cs) > best_c:
            best_c, best_a = np.mean(cs), a

    final_cox = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alphas=[best_a], fit_baseline_model=True).fit(Xtr, y_tr)
    sel_feats = Xtr.columns[np.abs(final_cox.coef_.ravel()) > 1e-5].tolist()
    if len(sel_feats) < 3:
        sel_feats = Xtr.columns.tolist()

    rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=10, min_samples_leaf=15,
                               max_features="sqrt", n_jobs=-1, random_state=42).fit(Xtr[sel_feats], y_tr)
    gbsa = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.03, max_depth=3,
                                            subsample=0.8, random_state=42).fit(Xtr[sel_feats], y_tr)

    models = {"CoxNet": (final_cox, Xtr.columns), "RSF": (rsf, sel_feats), "GBSA": (gbsa, sel_feats)}
    rows = []
    for m_name, (model, cols) in models.items():
        for cohort_label, X_eval, y_eval in [("TCGA-Train", Xtr, y_tr), ("TCGA-Test", Xte, y_te), ("METABRIC", Xme, y_me), ("GSE20685", Xgs, y_gs)]:
            if X_eval is None:
                continue
            preds = model.predict(X_eval[cols])
            c_mean, c_lo, c_hi = bootstrap_cindex(y_eval, preds)
            max_safe_t = min(np.percentile(y_tr["time"], 90), np.percentile(y_eval["time"], 90))
            valid_t = [t for t in [12.0, 36.0, 60.0] if t < max_safe_t and t > y_eval["time"].min()]
            keep_mask = y_eval["time"] <= max_safe_t
            _, mean_auc = cumulative_dynamic_auc(y_tr, y_eval[keep_mask], preds[keep_mask], valid_t)
            rows.append({
                "Modality": name, "Algorithm": m_name, "Cohort": cohort_label,
                "Selected_Features": len(sel_feats),
                "C_Index": round(c_mean, 4), "CI_95_Low": round(c_lo, 4), "CI_95_High": round(c_hi, 4),
                "Mean_1_3_5yr_AUC": round(float(mean_auc), 4)
            })
    return rows, sel_feats, models, scaler

res_clin, _, clin_models, clin_scaler = fit_and_eval_modality("Clinical-Only", clin_X_tr, clin_X_te, clin_X_me, clin_X_gs, l1_ratio=0.1)
res_gen,  _, gen_models,  gen_scaler  = fit_and_eval_modality("Genomic-Only", gen_tr, gen_te, gen_me, None, l1_ratio=0.5)
res_tra,  _, tra_models,  tra_scaler  = fit_and_eval_modality("Transcriptomic-Only", X_trans_tr, X_trans_te, X_trans_me, X_trans_gs, l1_ratio=0.7)
res_mul, multi_feats, multi_models, multi_scaler = fit_and_eval_modality("Integrated-MultiOmics", X_multi_tr, X_multi_te, X_multi_me, None, l1_ratio=0.7)

# Model 5: Clinico-MultiOmics Fusion (Molecular Risk Score + Age + Stage)
def get_mol_score(X_raw):
    Xs = pd.DataFrame(multi_scaler.transform(X_raw), index=X_raw.index, columns=X_raw.columns)
    return multi_models["CoxNet"][0].predict(Xs)

X_cm_tr = clin_X_tr.copy(); X_cm_tr["MultiOmics_Risk"] = get_mol_score(X_multi_tr)
X_cm_te = clin_X_te.copy(); X_cm_te["MultiOmics_Risk"] = get_mol_score(X_multi_te)
X_cm_me = clin_X_me.copy(); X_cm_me["MultiOmics_Risk"] = get_mol_score(X_multi_me)

res_cm, _, cm_models, cm_scaler = fit_and_eval_modality("Clinico-MultiOmics", X_cm_tr, X_cm_te, X_cm_me, None, l1_ratio=0.1)

bench_df = pd.DataFrame(res_clin + res_gen + res_tra + res_mul + res_cm)
bench_df.to_csv("results/tables/Table1_Model_Benchmarking.csv", index=False)
pd.Series(multi_feats).to_csv("results/tables/Selected_MultiOmics_Signature.csv", index=False, header=["Feature"])

# Save Risk Scores across all cohorts for downstream Kaplan-Meier, Multivariable Cox, SHAP & Drug Sensitivity
pd.DataFrame({
    "MultiOmics_Risk": get_mol_score(X_multi_tr),
    "ClinicoMultiOmics_Risk": cm_models["CoxNet"][0].predict(pd.DataFrame(cm_scaler.transform(X_cm_tr), index=X_cm_tr.index, columns=X_cm_tr.columns))
}, index=train_ids).to_csv("data/processed/risk_scores_tcga_train.csv")

pd.DataFrame({
    "MultiOmics_Risk": get_mol_score(X_multi_te),
    "ClinicoMultiOmics_Risk": cm_models["CoxNet"][0].predict(pd.DataFrame(cm_scaler.transform(X_cm_te), index=X_cm_te.index, columns=X_cm_te.columns))
}, index=test_ids).to_csv("data/processed/risk_scores_tcga_test.csv")

pd.DataFrame({
    "MultiOmics_Risk": get_mol_score(X_multi_me),
    "ClinicoMultiOmics_Risk": cm_models["CoxNet"][0].predict(pd.DataFrame(cm_scaler.transform(X_cm_me), index=X_cm_me.index, columns=X_cm_me.columns))
}, index=meta_clin.index).to_csv("data/processed/risk_scores_metabric.csv")

tra_gs_scaled = pd.DataFrame(tra_scaler.transform(X_trans_gs), index=X_trans_gs.index, columns=X_trans_gs.columns)
pd.DataFrame({
    "Transcriptomic_Risk": tra_models["CoxNet"][0].predict(tra_gs_scaled)
}, index=gse_clin.index).to_csv("data/processed/risk_scores_gse20685.csv")

print("\n=== UPDATED BENCHMARKING SUMMARY (TCGA-Test & External Cohorts) ===")
print(bench_df[bench_df["Cohort"] != "TCGA-Train"].to_string(index=False))
print(f"\nSelected Multi-Omics Signature Features ({len(multi_feats)}): {multi_feats}")