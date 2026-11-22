import os
import urllib.request
import tarfile
import pandas as pd
import numpy as np
import GEOparse

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def download_with_headers(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"Found cached {os.path.basename(dest)}")
        return True
    print(f"Downloading {os.path.basename(dest)}...")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return True

# 1. UCSC Xena TCGA-BRCA Endpoints (with Browser Headers)
URLS = {
    "tcga_expr": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2.gz",
    "tcga_clin": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/survival%2FBRCA_survival.txt",
    "tcga_pheno": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix",
    "tcga_maf": "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz"
}

xena_ok = True
for key, url in URLS.items():
    ext = ".gz" if url.endswith(".gz") else ".tsv"
    try:
        download_with_headers(url, f"data/raw/{key}{ext}")
    except Exception as e:
        print(f"Xena download failed for {key}: {e}")
        xena_ok = False
        break

# 2. Process TCGA-BRCA (or automatic cBioPortal PanCancer Atlas fallback)
if xena_ok:
    print("\n--- Processing TCGA-BRCA (UCSC Xena) ---")
    expr = pd.read_csv("data/raw/tcga_expr.gz", sep="\t", index_col=0)
    primary_samples = [c for c in expr.columns if c.endswith("-01")]
    expr = expr[primary_samples]

    surv = pd.read_csv("data/raw/tcga_clin.tsv", sep="\t")
    surv = surv[surv["sample"].isin(primary_samples)].copy()
    surv = surv.dropna(subset=["OS", "OS.time"])
    surv = surv[surv["OS.time"] > 0].drop_duplicates(subset=["sample"])

    pheno = pd.read_csv("data/raw/tcga_pheno.tsv", sep="\t", index_col=0)
    mut = pd.read_csv("data/raw/tcga_maf.gz", sep="\t", index_col=0)

    common_samples = sorted(list(set(surv["sample"]) & set(expr.columns) & set(mut.columns)))
    surv_clean = surv.set_index("sample").loc[common_samples]
    # Join key clinical covariates (Age, Stage, PAM50, ER/PR/HER2) if present
    pheno_cols = [c for c in pheno.columns if any(k in c.lower() for k in ["age_at_initial", "pathologic_stage", "pam50", "er_status", "pr_status", "her2_status", "gender"])]
    surv_clean = surv_clean.join(pheno[pheno_cols], how="left")

    expr_clean = expr[common_samples]
    mut_clean  = mut[common_samples]
else:
    print("\n--- Switching to cBioPortal TCGA-BRCA PanCancer Atlas Mirror ---")
    cbio_url = "https://datahub.assets.cbioportal.org/brca_tcga_pan_can_atlas_2018.tar.gz"
    tar_path = "data/raw/brca_tcga_pan_can_atlas_2018.tar.gz"
    download_with_headers(cbio_url, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("data/raw/")
    base = "data/raw/brca_tcga_pan_can_atlas_2018"
    expr = pd.read_csv(f"{base}/data_mrna_seq_v2_rsem.txt", sep="\t").dropna(subset=["Hugo_Symbol"])
    expr = expr.groupby("Hugo_Symbol").mean(numeric_only=True)
    primary_samples = [c for c in expr.columns if c.endswith("-01")]
    expr = np.log2(expr[primary_samples].clip(lower=0) + 1)

    clin = pd.read_csv(f"{base}/data_clinical_patient.txt", sep="\t", comment="#")
    clin["sample"] = clin["PATIENT_ID"] + "-01"
    clin["OS"] = clin["OS_STATUS"].apply(lambda x: 1 if str(x).startswith("1:") else (0 if str(x).startswith("0:") else np.nan))
    clin["OS.time"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce") * 30.4375
    surv = clin.dropna(subset=["OS", "OS.time"])
    surv = surv[surv["OS.time"] > 0].drop_duplicates(subset=["sample"])

    maf = pd.read_csv(f"{base}/data_mutations.txt", sep="\t", comment="#", low_memory=False)
    maf_nonsilent = maf[maf["Variant_Classification"] != "Silent"]
    mut_ct = pd.crosstab(maf_nonsilent["Hugo_Symbol"], maf_nonsilent["Tumor_Sample_Barcode"])
    mut = (mut_ct > 0).astype(int)

    common_samples = sorted(list(set(surv["sample"]) & set(expr.columns) & set(mut.columns)))
    surv_clean = surv.set_index("sample").loc[common_samples]
    expr_clean = expr[common_samples]
    mut_clean  = mut.reindex(columns=common_samples, fill_value=0)

print(f"TCGA-BRCA Matched Multi-Omics Primary Tumors: n = {len(common_samples)}")
print(f"TCGA-BRCA Overall Survival Events (Deaths): {int(surv_clean['OS'].sum())} ({surv_clean['OS'].mean()*100:.1f}%)")
print(f"TCGA-BRCA Median Follow-up (days): {surv_clean['OS.time'].median():.1f}")
print(f"TCGA-BRCA Genes Available: {expr_clean.shape[0]}")

expr_clean.to_parquet("data/processed/tcga_brca_expr.parquet")
mut_clean.to_parquet("data/processed/tcga_brca_mut.parquet")
surv_clean.to_csv("data/processed/tcga_brca_surv.csv")

# 3. Download & Parse External Cohort 2: GEO GSE20685 (n = 327)
print("\n--- Downloading & Parsing GEO GSE20685 ---")
gse = GEOparse.get_GEO(geo="GSE20685", destdir="data/raw", silent=True)
gse_pheno = gse.phenotype_data
print(f"GSE20685 Total Samples Downloaded: n = {gse_pheno.shape[0]}")
gse_pheno.to_csv("data/processed/gse20685_raw_pheno.csv")

# Also extract GSE20685 Gene-Level Expression Matrix (GPL570 -> Gene Symbol)
gpl = list(gse.gpls.values())[0].table
probe_to_gene = gpl[["ID", "Gene Symbol"]].dropna()
probe_to_gene = probe_to_gene[probe_to_gene["Gene Symbol"] != ""]
probe_to_gene["Gene Symbol"] = probe_to_gene["Gene Symbol"].apply(lambda x: str(x).split(" /// ")[0].strip())

piv = gse.pivot_samples("VALUE")
piv = piv.join(probe_to_gene.set_index("ID"), how="inner")
gse_expr_gene = piv.groupby("Gene Symbol").mean(numeric_only=True)
gse_expr_gene.to_parquet("data/processed/gse20685_expr.parquet")
print(f"GSE20685 Mapped Unique Genes: {gse_expr_gene.shape[0]}")
print("\nDay-1 Acquisition Complete!")