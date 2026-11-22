import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gseapy as gp
from lifelines import CoxPHFitter
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sklearn.model_selection import StratifiedKFold

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# 1. Load Harmonized Datasets & Locked Train/Test Split
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

# 2. Sample-wise Z-score normalization + NaN-safe zero-imputation (strictly within each cohort -> zero leakage)
def sample_zscore(df):
    z = df.sub(df.mean(axis=0), axis=1).div(df.std(axis=0).replace(0, 1), axis=1)
    return z.fillna(0.0)

expr_tr = sample_zscore(tcga_expr[train_ids])
expr_te = sample_zscore(tcga_expr[test_ids])
expr_me = sample_zscore(meta_expr)
expr_gs = sample_zscore(gse_expr)

# 3. Load Cached MSigDB Hallmark ssGSEA Scores
def run_ssgsea_cohort(expr_df, name):
    cache_path = f"data/processed/ssgsea_{name}.parquet"
    if os.path.exists(cache_path):
        mat = pd.read_parquet(cache_path)
        return mat.fillna(0.0)
    ss = gp.ssgsea(data=expr_df, gene_sets="MSigDB_Hallmark_2020", outdir=None, no_plot=True, threads=4, verbose=False)
    mat = ss.res2d.pivot(index="Term", columns="Name", values="NES").astype(float)
    mat.index = ["HALLMARK_" + idx.replace(" ", "_").upper() for idx in mat.index]
    mat = mat[expr_df.columns].fillna(0.0)
    mat.to_parquet(cache_path)
    return mat

path_tr = run_ssgsea_cohort(expr_tr, "tcga_train")
path_te = run_ssgsea_cohort(expr_te, "tcga_test")
path_me = run_ssgsea_cohort(expr_me, "metabric")
path_gs = run_ssgsea_cohort(expr_gs, "gse20685")
common_paths = sorted(list(set(path_tr.index) & set(path_me.index) & set(path_gs.index)))

# 4. Somatic Genomic Driver Matrix (Aligned between TCGA-BRCA and METABRIC)
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

# 5. Univariate Cox Screening on TCGA-Train Genes (p < 0.01)
y_train_df = tcga_clin.loc[train_ids, ["OS_MONTHS", "OS"]].copy()
univ_results = []
for gene in expr_tr.index:
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
        if pval < 0.01:
            univ_results.append((gene, hr, pval))
    except Exception:
        continue

univ_df = pd.DataFrame(univ_results, columns=["Gene", "HR", "P_value"]).sort_values("P_value")
univ_df.to_csv("results/tables/TableS1_Univariate_Cox_Genes.csv", index=False)
cand_genes = univ_df["Gene"].head(150).tolist()

# Assemble Modality Feature Matrices
X_trans_tr = pd.concat([expr_tr.loc[cand_genes].T, path_tr.loc[common_paths].T], axis=1).fillna(0.0)
X_trans_te = pd.concat([expr_te.loc[cand_genes].T, path_te.loc[common_paths].T], axis=1).fillna(0.0)
X_trans_me = pd.concat([expr_me.loc[cand_genes].T, path_me.loc[common_paths].T], axis=1).fillna(0.0)
X_trans_gs = pd.concat([expr_gs.loc[cand_genes].T, path_gs.loc[common_paths].T], axis=1).fillna(0.0)

X_multi_tr = pd.concat([X_trans_tr, gen_tr], axis=1).fillna(0.0)
X_multi_te = pd.concat([X_trans_te, gen_te], axis=1).fillna(0.0)
X_multi_me = pd.concat([X_trans_me, gen_me], axis=1).fillna(0.0)

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

def fit_and_eval_modality(name, Xtr, Xte, Xme, Xgs=None):
    coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.85, alpha_min_ratio=0.05, max_iter=100000)
    coxnet.fit(Xtr, y_tr)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_a, best_c = coxnet.alphas_[len(coxnet.alphas_)//2], -1
    for a in coxnet.alphas_:
        cs = []
        for ti, vi in cv.split(Xtr, tcga_clin.loc[train_ids, "OS"]):
            m = CoxnetSurvivalAnalysis(l1_ratio=0.85, alphas=[a]).fit(Xtr.iloc[ti], y_tr[ti])
            p = m.predict(Xtr.iloc[vi])
            cs.append(concordance_index_censored(y_tr[vi]["event"], y_tr[vi]["time"], p)[0])
        if np.mean(cs) > best_c:
            best_c, best_a = np.mean(cs), a

    final_cox = CoxnetSurvivalAnalysis(l1_ratio=0.85, alphas=[best_a], fit_baseline_model=True).fit(Xtr, y_tr)
    sel_feats = Xtr.columns[np.abs(final_cox.coef_.ravel()) > 1e-5].tolist()
    if len(sel_feats) < 5:
        sel_feats = Xtr.columns[:15].tolist()

    rsf = RandomSurvivalForest(n_estimators=500, min_samples_split=10, min_samples_leaf=15,
                               max_features="sqrt", n_jobs=-1, random_state=42).fit(Xtr[sel_feats], y_tr)
    gbsa = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.05, max_depth=3,
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
    return rows, sel_feats, models

print("Benchmarking Modality 1: Genomic-Only...")
res_gen, _, _ = fit_and_eval_modality("Genomic-Only", gen_tr, gen_te, gen_me, None)

print("Benchmarking Modality 2: Transcriptomic-Only...")
res_tra, _, _ = fit_and_eval_modality("Transcriptomic-Only", X_trans_tr, X_trans_te, X_trans_me, X_trans_gs)

print("Benchmarking Modality 3: Integrated Multi-Omics...")
res_mul, multi_feats, multi_models = fit_and_eval_modality("Integrated-MultiOmics", X_multi_tr, X_multi_te, X_multi_me, None)

bench_df = pd.DataFrame(res_gen + res_tra + res_mul)
bench_df.to_csv("results/tables/Table1_Model_Benchmarking.csv", index=False)
pd.Series(multi_feats).to_csv("results/tables/Selected_MultiOmics_Signature.csv", index=False, header=["Feature"])

print("\n=== BENCHMARKING SUMMARY (TCGA-Test & External Cohorts) ===")
print(bench_df[bench_df["Cohort"] != "TCGA-Train"].to_string(index=False))
print(f"\nSelected Multi-Omics Signature Features ({len(multi_feats)}): {multi_feats}")