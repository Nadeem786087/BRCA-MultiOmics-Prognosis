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
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import shap

# Publication Matplotlib Styling (300 DPI, Clean Sans-Serif)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

# 1. Load Data & Risk Scores
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

risk_tr = pd.read_csv("data/processed/risk_scores_tcga_train.csv", index_col=0)
risk_te = pd.read_csv("data/processed/risk_scores_tcga_test.csv", index_col=0)
risk_me = pd.read_csv("data/processed/risk_scores_metabric.csv", index_col=0)
risk_gs = pd.read_csv("data/processed/risk_scores_gse20685.csv", index_col=0)

bench_df = pd.read_csv("results/tables/Table1_Model_Benchmarking.csv")
multi_feats = pd.read_csv("results/tables/Selected_MultiOmics_Signature.csv")["Feature"].tolist()

# ==============================================================================
# FIGURE 1: Multi-Cohort Model Benchmarking (C-Index & Mean AUC)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
palette = {"Clinical-Only": "#7F8C8D", "Genomic-Only": "#E67E22", "Transcriptomic-Only": "#2980B9",
           "Integrated-MultiOmics": "#8E44AD", "Clinico-MultiOmics": "#C0392B"}

cox_sub = bench_df[(bench_df["Algorithm"] == "CoxNet") & (bench_df["Cohort"] != "TCGA-Train")]
sns.barplot(data=cox_sub, x="Cohort", y="C_Index", hue="Modality", palette=palette, ax=axes[0], edgecolor="black", linewidth=0.6)
axes[0].axhline(0.5, ls="--", color="gray", lw=1)
axes[0].set_ylim(0.40, 0.78)
axes[0].set_title("A. Concordance Index (C-Index) Across Cohorts")
axes[0].set_ylabel("Harrell's C-Index")
axes[0].set_xlabel("")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")

sns.barplot(data=cox_sub, x="Cohort", y="Mean_1_3_5yr_AUC", hue="Modality", palette=palette, ax=axes[1], edgecolor="black", linewidth=0.6)
axes[1].axhline(0.5, ls="--", color="gray", lw=1)
axes[1].set_ylim(0.40, 0.78)
axes[1].set_title("B. Time-Dependent Mean AUC (1-, 3-, 5-Year OS)")
axes[1].set_ylabel("Integrated Dynamic ROC-AUC")
axes[1].set_xlabel("")
axes[1].get_legend().remove()

plt.tight_layout()
plt.savefig("results/figures/Fig1_MultiCohort_Benchmark.png")
plt.savefig("results/figures/Fig1_MultiCohort_Benchmark.pdf")
plt.close()

# ==============================================================================
# FIGURE 2: Kaplan-Meier Survival Stratification Across All 4 Cohorts
# ==============================================================================
cutoff_multi = risk_tr["ClinicoMultiOmics_Risk"].median()

km_cohorts = [
    ("A. TCGA-BRCA Discovery (n = 795)", tcga_clin.loc[train_ids], risk_tr["ClinicoMultiOmics_Risk"], cutoff_multi),
    ("B. TCGA-BRCA Holdout Test (n = 199)", tcga_clin.loc[test_ids], risk_te["ClinicoMultiOmics_Risk"], cutoff_multi),
    ("C. METABRIC External Cohort (n = 1,865)", meta_clin, risk_me["ClinicoMultiOmics_Risk"], risk_me["ClinicoMultiOmics_Risk"].median()),
    ("D. GEO GSE20685 External Cohort (n = 327)", gse_clin, risk_gs["Transcriptomic_Risk"], risk_gs["Transcriptomic_Risk"].median())
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
axes = axes.ravel()

for ax, (title, clin_df, scores, thresh) in zip(axes, km_cohorts):
    high_mask = scores >= thresh
    low_mask  = ~high_mask

    lr = logrank_test(
        clin_df.loc[high_mask, "OS_MONTHS"], clin_df.loc[low_mask, "OS_MONTHS"],
        event_observed_A=clin_df.loc[high_mask, "OS"], event_observed_B=clin_df.loc[low_mask, "OS"]
    )

    kmf_hi = KaplanMeierFitter()
    kmf_lo = KaplanMeierFitter()

    kmf_hi.fit(clin_df.loc[high_mask, "OS_MONTHS"], clin_df.loc[high_mask, "OS"], label=f"High-Risk (n={high_mask.sum()})")
    kmf_lo.fit(clin_df.loc[low_mask, "OS_MONTHS"], clin_df.loc[low_mask, "OS"], label=f"Low-Risk (n={low_mask.sum()})")

    kmf_lo.plot_survival_function(ax=ax, ci_show=True, color="#1F77B4", lw=2.2)
    kmf_hi.plot_survival_function(ax=ax, ci_show=True, color="#D62728", lw=2.2)

    p_txt = f"Log-rank p = {lr.p_value:.2e}" if lr.p_value < 0.001 else f"Log-rank p = {lr.p_value:.4f}"
    ax.text(0.05, 0.12, p_txt, transform=ax.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9", edgecolor="#BDC3C7"))
    ax.set_title(title)
    ax.set_xlabel("Overall Survival Time (Months)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.03, 0.22))

plt.tight_layout()
plt.savefig("results/figures/Fig2_KaplanMeier_AllCohorts.png")
plt.savefig("results/figures/Fig2_KaplanMeier_AllCohorts.pdf")
plt.close()

# ==============================================================================
# FIGURE 3 & TABLE 2: Multivariable Cox Regression (Clinical Independence Proof)
# ==============================================================================
mv_df = pd.DataFrame(index=meta_clin.index)
mv_df["OS_MONTHS"] = meta_clin["OS_MONTHS"]
mv_df["OS"] = meta_clin["OS"]
mv_df["MultiOmics_Risk_Z"] = (risk_me["MultiOmics_Risk"] - risk_me["MultiOmics_Risk"].mean()) / risk_me["MultiOmics_Risk"].std()
mv_df["Age_per_10yr"] = pd.to_numeric(meta_clin["Age"], errors="coerce") / 10.0

npi_col = [c for c in meta_clin.columns if "NPI" in c.upper()]
er_col  = [c for c in meta_clin.columns if "ER_IHC" in c.upper() or "ER_STATUS" in c.upper()]

if npi_col:
    mv_df["NPI_Score"] = pd.to_numeric(meta_clin[npi_col[0]], errors="coerce")
if er_col:
    mv_df["ER_Negative"] = (meta_clin[er_col[0]].astype(str).str.contains("Neg", case=False)).astype(float)

mv_df["Basal_TNBC"] = (meta_clin["Subtype"].astype(str).str.contains("Basal|claudin-low", case=False)).astype(float)
mv_df["HER2_Enriched"] = (meta_clin["Subtype"].astype(str).str.contains("Her2", case=False)).astype(float)
mv_df = mv_df.dropna()

cph_mv = CoxPHFitter()
cph_mv.fit(mv_df, duration_col="OS_MONTHS", event_col="OS")
mv_summary = cph_mv.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
mv_summary.columns = ["HR", "CI_95_Low", "CI_95_High", "P_value"]
mv_summary.to_csv("results/tables/Table2_Multivariable_Cox_METABRIC.csv")
print("\n=== TABLE 2: MULTIVARIABLE COX REGRESSION IN METABRIC (n = {}) ===".format(len(mv_df)))
print(mv_summary.to_string())

label_map = {
    "MultiOmics_Risk_Z": "Multi-Omics Risk Score (per SD)",
    "Age_per_10yr": "Age at Diagnosis (per 10 yrs)",
    "NPI_Score": "Nottingham Prognostic Index (NPI)",
    "ER_Negative": "ER-Negative Status",
    "Basal_TNBC": "Basal / Claudin-Low Subtype",
    "HER2_Enriched": "HER2-Enriched Subtype"
}
labels = [label_map.get(idx, idx) for idx in mv_summary.index]

fig, ax = plt.subplots(figsize=(9.0, 4.5))
y_pos = np.arange(len(mv_summary))
ax.errorbar(mv_summary["HR"], y_pos,
            xerr=[mv_summary["HR"] - mv_summary["CI_95_Low"], mv_summary["CI_95_High"] - mv_summary["HR"]],
            fmt="o", color="#C0392B", ecolor="#2C3E50", elinewidth=1.8, capsize=4, markersize=7)
ax.axvline(1.0, ls="--", color="gray", lw=1.2)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("Multivariable Hazard Ratio (95% CI)")
ax.set_title("Independent Prognostic Value of Multi-Omics Risk Score (METABRIC)")

for i, (_, row) in enumerate(mv_summary.iterrows()):
    p_str = f"p={row['P_value']:.2e}" if row["P_value"] < 0.001 else f"p={row['P_value']:.3f}"
    ax.text(row["CI_95_High"] + 0.03, i, f"HR={row['HR']:.2f} ({row['CI_95_Low']:.2f}-{row['CI_95_High']:.2f}), {p_str}", va="center", fontsize=8.5)
ax.set_xlim(0.65, mv_summary["CI_95_High"].max() + 0.90)

plt.tight_layout()
plt.savefig("results/figures/Fig3_Multivariable_Cox_Forest.png")
plt.savefig("results/figures/Fig3_Multivariable_Cox_Forest.pdf")
plt.close()

# ==============================================================================
# FIGURE 4: SHAP Explainability on Multi-Omics Survival Model
# ==============================================================================
def sample_zscore(df):
    z = df.sub(df.mean(axis=0), axis=1).div(df.std(axis=0).replace(0, 1), axis=1)
    return z.fillna(0.0)

expr_tr = sample_zscore(tcga_expr[train_ids])
expr_me = sample_zscore(meta_expr)
path_tr = pd.read_parquet("data/processed/ssgsea_tcga_train.parquet").fillna(0.0)
path_me = pd.read_parquet("data/processed/ssgsea_metabric.parquet").fillna(0.0)

DRIVER_GENES = ["TP53", "PIK3CA", "GATA3", "CDH1", "MAP3K1", "KMT2C", "PTEN", "AKT1", "RB1", "BRCA1", "BRCA2"]
def get_multi_matrix(expr_df, path_df, mut_df, clin_df, sids):
    g_df = pd.DataFrame(index=sids)
    for g in DRIVER_GENES:
        g_df[f"MUT_{g}"] = mut_df.loc[g, sids].fillna(0).values if g in mut_df.index else 0
    g_df["LOG_TMB"] = np.log1p(clin_df.loc[sids, "TMB"].fillna(0).values)
    full = pd.concat([expr_df[sids].T, path_df[sids].T, g_df], axis=1)
    return full[multi_feats].fillna(0.0)

X_shap_tr = get_multi_matrix(expr_tr, path_tr, tcga_mut, tcga_clin, train_ids)
X_shap_me = get_multi_matrix(expr_me, path_me, meta_mut, meta_clin, meta_clin.index.tolist())

scaler = StandardScaler()
X_shap_tr_s = pd.DataFrame(scaler.fit_transform(X_shap_tr), index=X_shap_tr.index, columns=multi_feats)
X_shap_me_s = pd.DataFrame(scaler.transform(X_shap_me), index=X_shap_me.index, columns=multi_feats)

y_tr_surv = Surv.from_arrays(event=tcga_clin.loc[train_ids, "OS"].astype(bool), time=tcga_clin.loc[train_ids, "OS_MONTHS"].astype(float))
gbsa_shap = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=42).fit(X_shap_tr_s, y_tr_surv)

# Model-agnostic Permutation SHAP Explainer on survival hazard predictions
background = shap.sample(X_shap_tr_s, 100, random_state=42)
explainer = shap.Explainer(gbsa_shap.predict, background, seed=42)
shap_exp = explainer(X_shap_me_s.iloc[:250])

# Save Top SHAP Features Table
mean_abs_shap = pd.DataFrame({
    "Feature": multi_feats,
    "Mean_Abs_SHAP": np.abs(shap_exp.values).mean(axis=0)
}).sort_values("Mean_Abs_SHAP", ascending=False)
mean_abs_shap.to_csv("results/tables/TableS2_SHAP_Feature_Importance.csv", index=False)
print("\n=== TOP 10 FEATURES BY MEAN |SHAP| VALUE (METABRIC) ===")
print(mean_abs_shap.head(10).to_string(index=False))

plt.figure(figsize=(9.5, 6.5))
shap.summary_plot(shap_exp.values, X_shap_me_s.iloc[:250], max_display=18, show=False)
plt.title("SHAP Feature Attribution Across Multi-Omics Signature (METABRIC Validation)")
plt.tight_layout()
plt.savefig("results/figures/Fig4_SHAP_Interpretability.png")
plt.savefig("results/figures/Fig4_SHAP_Interpretability.pdf")
plt.close()

# ==============================================================================
# FIGURE 5: Translational Biology — Hallmark Pathway & Immune Microenvironment
# ==============================================================================
target_pathways = [
    "HALLMARK_ESTROGEN_RESPONSE_LATE",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_G2M_CHECKPOINT",
    "HALLMARK_E2F_TARGETS",
    "HALLMARK_INTERFERON_GAMMA_RESPONSE",
    "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"
]
avail_paths = [p for p in target_pathways if p in path_me.index]

hi_ids = risk_me.index[risk_me["MultiOmics_Risk"] >= risk_me["MultiOmics_Risk"].median()]
lo_ids = risk_me.index[risk_me["MultiOmics_Risk"] <  risk_me["MultiOmics_Risk"].median()]

rows_bio = []
plot_records = []
for p in avail_paths:
    hi_v = path_me.loc[p, hi_ids].values
    lo_v = path_me.loc[p, lo_ids].values
    stat, pval = mannwhitneyu(hi_v, lo_v)
    clean_name = p.replace("HALLMARK_", "").replace("_", " ").title()
    rows_bio.append({"Pathway": clean_name, "HighRisk_Median": round(float(np.median(hi_v)), 4),
                     "LowRisk_Median": round(float(np.median(lo_v)), 4), "P_value": pval})
    for v in hi_v:
        plot_records.append({"Pathway": clean_name, "Risk_Group": "High-Risk", "ssGSEA_NES": v})
    for v in lo_v:
        plot_records.append({"Pathway": clean_name, "Risk_Group": "Low-Risk", "ssGSEA_NES": v})

bio_df = pd.DataFrame(rows_bio)
bio_df["FDR"] = multipletests(bio_df["P_value"], method="fdr_bh")[1]
bio_df.to_csv("results/tables/Table3_Biological_Pathway_Differences.csv", index=False)
print("\n=== TABLE 3: HALLMARK PATHWAY ACTIVITY (HIGH-RISK VS LOW-RISK, METABRIC) ===")
print(bio_df.to_string(index=False))

plot_df = pd.DataFrame(plot_records)
plt.figure(figsize=(11, 5.2))
sns.boxplot(data=plot_df, x="Pathway", y="ssGSEA_NES", hue="Risk_Group",
            palette={"Low-Risk": "#1F77B4", "High-Risk": "#D62728"}, fliersize=1.5, linewidth=0.9)
plt.xticks(rotation=28, ha="right")
plt.ylabel("Pathway Enrichment Score (ssGSEA NES)")
plt.xlabel("")
plt.title("Therapeutic & Immune Pathway Activity by Multi-Omics Risk Stratum (METABRIC)")
plt.legend(frameon=False, title="")
plt.tight_layout()
plt.savefig("results/figures/Fig5_Immune_and_Therapeutic_Vulnerability.png")
plt.savefig("results/figures/Fig5_Immune_and_Therapeutic_Vulnerability.pdf")
plt.close()

print("\nAll 5 Publication Figures (PNG + PDF, 300 DPI) and Tables 2-3 Generated Successfully!")