import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

def download_file(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"Found cached {os.path.basename(dest)}")
        return
    print(f"Downloading {os.path.basename(dest)}...")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

# 1. Download & Extract METABRIC (n ~ 1,904) from cBioPortal DataHub
meta_tar = "data/raw/brca_metabric.tar.gz"
download_file("https://datahub.assets.cbioportal.org/brca_metabric.tar.gz", meta_tar)
if not os.path.exists("data/raw/brca_metabric"):
    with tarfile.open(meta_tar, "r:gz") as tar:
        tar.extractall("data/raw/")

print("\n--- 1. Enriching TCGA-BRCA Clinical & Genomic Features ---")
tcga_base = "data/raw/brca_tcga_pan_can_atlas_2018"
tcga_expr = pd.read_parquet("data/processed/tcga_brca_expr.parquet")
tcga_mut  = pd.read_parquet("data/processed/tcga_brca_mut.parquet")
tcga_surv = pd.read_csv("data/processed/tcga_brca_surv.csv", index_col=0)

tcga_pat  = pd.read_csv(f"{tcga_base}/data_clinical_patient.txt", sep="\t", comment="#").set_index("PATIENT_ID")
tcga_samp = pd.read_csv(f"{tcga_base}/data_clinical_sample.txt", sep="\t", comment="#").set_index("SAMPLE_ID")

# Map patient & sample covariates onto TCGA-BRCA (-01 samples)
tcga_surv["PATIENT_ID"] = [s[:12] for s in tcga_surv.index]
tcga_surv["Age"] = pd.to_numeric(tcga_pat.reindex(tcga_surv["PATIENT_ID"])["AGE"].values, errors="coerce")
tcga_surv["Subtype"] = tcga_pat.reindex(tcga_surv["PATIENT_ID"])["SUBTYPE"].values
stage_raw = tcga_pat.reindex(tcga_surv["PATIENT_ID"])["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).values
tcga_surv["Stage_Simple"] = [
    "Stage_I" if "STAGE I" in s and "II" not in s and "IV" not in s else
    ("Stage_II" if "STAGE II" in s and "III" not in s else
    ("Stage_III_IV" if ("STAGE III" in s or "STAGE IV" in s) else "Unknown"))
    for s in stage_raw
]
if "TMB_NONSYNONYMOUS" in tcga_samp.columns:
    tcga_surv["TMB"] = pd.to_numeric(tcga_samp.reindex(tcga_surv.index)["TMB_NONSYNONYMOUS"].values, errors="coerce")
else:
    tcga_surv["TMB"] = tcga_mut.sum(axis=0).values

tcga_surv["OS_MONTHS"] = tcga_surv["OS.time"] / 30.4375

print("\n--- 2. Harmonizing METABRIC External Cohort ---")
meta_base = "data/raw/brca_metabric"
meta_expr = pd.read_csv(f"{meta_base}/data_mrna_illumina_microarray.txt", sep="\t").dropna(subset=["Hugo_Symbol"])
meta_expr = meta_expr.groupby("Hugo_Symbol").mean(numeric_only=True)

meta_pat = pd.read_csv(f"{meta_base}/data_clinical_patient.txt", sep="\t", comment="#").set_index("PATIENT_ID")
meta_pat["OS"] = meta_pat["OS_STATUS"].apply(lambda x: 1 if str(x).startswith("1:") else (0 if str(x).startswith("0:") else np.nan))
meta_pat["OS_MONTHS"] = pd.to_numeric(meta_pat["OS_MONTHS"], errors="coerce")
meta_pat = meta_pat.dropna(subset=["OS", "OS_MONTHS"])
meta_pat = meta_pat[meta_pat["OS_MONTHS"] > 0]

meta_maf = pd.read_csv(f"{meta_base}/data_mutations.txt", sep="\t", comment="#", low_memory=False)
meta_mut_ct = pd.crosstab(meta_maf["Hugo_Symbol"], meta_maf["Tumor_Sample_Barcode"])
meta_mut = (meta_mut_ct > 0).astype(int)

meta_common = sorted(list(set(meta_pat.index) & set(meta_expr.columns) & set(meta_mut.columns)))
meta_clin = meta_pat.loc[meta_common].copy()
meta_clin["Age"] = pd.to_numeric(meta_clin["AGE_AT_DIAGNOSIS"], errors="coerce")
meta_clin["Subtype"] = meta_clin["CLAUDIN_SUBTYPE"] if "CLAUDIN_SUBTYPE" in meta_clin.columns else "Unknown"
meta_expr = meta_expr[meta_common]
meta_mut  = meta_mut.reindex(columns=meta_common, fill_value=0)
meta_clin["TMB"] = meta_mut.sum(axis=0).values

print(f"METABRIC Matched Multi-Omics Cohort: n = {len(meta_common)}, OS Events = {int(meta_clin['OS'].sum())} ({meta_clin['OS'].mean()*100:.1f}%), Median Follow-up = {meta_clin['OS_MONTHS'].median():.1f} mo")

print("\n--- 3. Harmonizing GEO GSE20685 External Cohort ---")
gse_pheno = pd.read_csv("data/processed/gse20685_raw_pheno.csv", index_col=0)
gse_expr  = pd.read_parquet("data/processed/gse20685_expr.parquet")

# Parse survival fields from GSE20685 phenotype columns
os_event, os_months = [], []
for idx, row in gse_pheno.iterrows():
    row_str = " | ".join([f"{col}:{val}" for col, val in row.items() if pd.notna(val)])
    ev, tm = np.nan, np.nan
    for col, val in row.items():
        s = str(val).lower()
        if "event_death" in s or "status" in col.lower():
            if ": 1" in s or s.strip() == "1" or "dead" in s:
                ev = 1
            elif ": 0" in s or s.strip() == "0" or "alive" in s:
                ev = 0
        if "follow_up_time" in s or "survival" in s or "years" in s:
            parts = str(val).split(":")
            try:
                num = float(parts[-1].strip())
                tm = num * 12.0 if "year" in s else num
            except:
                pass
    os_event.append(ev)
    os_months.append(tm)

gse_clin = pd.DataFrame({"OS": os_event, "OS_MONTHS": os_months}, index=gse_pheno.index)
gse_clin = gse_clin.dropna(subset=["OS", "OS_MONTHS"])
gse_clin = gse_clin[gse_clin["OS_MONTHS"] > 0]
gse_common = sorted(list(set(gse_clin.index) & set(gse_expr.columns)))
gse_clin = gse_clin.loc[gse_common]
gse_expr = gse_expr[gse_common]

print(f"GSE20685 Matched Cohort: n = {len(gse_common)}, OS Events = {int(gse_clin['OS'].sum())} ({gse_clin['OS'].mean()*100:.1f}%), Median Follow-up = {gse_clin['OS_MONTHS'].median():.1f} mo")

# 4. Intersect Common Gene Universe Across All 3 Cohorts & Lock 80/20 Stratified Split
common_genes = sorted(list(set(tcga_expr.index) & set(meta_expr.index) & set(gse_expr.index)))
print(f"\nCommon Protein-Coding / Annotated Gene Universe Across All 3 Cohorts: {len(common_genes)} genes")

tcga_expr = tcga_expr.loc[common_genes]
meta_expr = meta_expr.loc[common_genes]
gse_expr  = gse_expr.loc[common_genes]

# Stratified 80/20 split on TCGA-BRCA by (OS event + Stage_Simple)
strat_label = tcga_surv["OS"].astype(int).astype(str) + "_" + tcga_surv["Stage_Simple"].astype(str)
# Merge rare strata (<2 samples) into event-only label so StratifiedSplit never fails
vc = strat_label.value_counts()
strat_label = strat_label.apply(lambda x: x if vc[x] >= 2 else x.split("_")[0])

train_ids, test_ids = train_test_split(
    tcga_surv.index.tolist(), test_size=0.20, random_state=42, stratify=strat_label
)

pd.Series(train_ids).to_csv("data/processed/tcga_train_ids.csv", index=False, header=["sample_id"])
pd.Series(test_ids).to_csv("data/processed/tcga_test_ids.csv", index=False, header=["sample_id"])

# Save Harmonized Multi-Cohort Artifacts
tcga_expr.to_parquet("data/processed/harmonized_tcga_expr.parquet")
tcga_mut.to_parquet("data/processed/harmonized_tcga_mut.parquet")
tcga_surv.to_csv("data/processed/harmonized_tcga_clin.csv")

meta_expr.to_parquet("data/processed/harmonized_meta_expr.parquet")
meta_mut.to_parquet("data/processed/harmonized_meta_mut.parquet")
meta_clin.to_csv("data/processed/harmonized_meta_clin.csv")

gse_expr.to_parquet("data/processed/harmonized_gse_expr.parquet")
gse_clin.to_csv("data/processed/harmonized_gse_clin.csv")

print(f"\nLocked Stratified Split -> TCGA-Train: n = {len(train_ids)} (Events: {int(tcga_surv.loc[train_ids, 'OS'].sum())}) | TCGA-Test: n = {len(test_ids)} (Events: {int(tcga_surv.loc[test_ids, 'OS'].sum())})")
print("All 3 Cohorts Harmonized & Saved!")