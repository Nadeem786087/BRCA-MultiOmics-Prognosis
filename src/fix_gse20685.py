import pandas as pd
import numpy as np

gse_pheno = pd.read_csv("data/processed/gse20685_raw_pheno.csv", index_col=0)
gse_expr  = pd.read_parquet("data/processed/gse20685_expr.parquet")
tcga_expr = pd.read_parquet("data/processed/harmonized_tcga_expr.parquet")

gse_clin = pd.DataFrame(index=gse_pheno.index)
gse_clin["OS"]        = pd.to_numeric(gse_pheno["characteristics_ch1.3.event_death"], errors="coerce")
gse_clin["OS_MONTHS"] = pd.to_numeric(gse_pheno["characteristics_ch1.4.follow_up_duration (years)"], errors="coerce") * 12.0
gse_clin["Age"]       = pd.to_numeric(gse_pheno["characteristics_ch1.2.age at diagnosis"], errors="coerce")
gse_clin["Subtype"]   = gse_pheno["characteristics_ch1.6.subtype"]

gse_clin = gse_clin.dropna(subset=["OS", "OS_MONTHS"])
gse_clin = gse_clin[gse_clin["OS_MONTHS"] > 0]
gse_common = sorted(list(set(gse_clin.index.astype(str)) & set(gse_expr.columns.astype(str))))

gse_clin = gse_clin.loc[gse_common]
gse_expr = gse_expr.loc[tcga_expr.index, gse_common]

gse_clin.to_csv("data/processed/harmonized_gse_clin.csv")
gse_expr.to_parquet("data/processed/harmonized_gse_expr.parquet")
print(f"Fixed GSE20685 Matched Cohort: n = {len(gse_common)}, OS Events = {int(gse_clin['OS'].sum())} ({gse_clin['OS'].mean()*100:.1f}%), Median Follow-up = {gse_clin['OS_MONTHS'].median():.1f} mo, Genes = {gse_expr.shape[0]}")