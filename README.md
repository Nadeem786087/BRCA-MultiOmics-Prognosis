# TRIPOD+AI Compliant Multi-Omics Integration of Copy-Number Alterations, Somatic Mutations, and Pathway Transcriptomics Improves Cancer-Specific Survival Prediction Across 3,186 Breast Cancer Patients

[![bioRxiv Preprint](https://img.shields.io/badge/bioRxiv-Preprint-b31b1b.svg)](https://www.biorxiv.org/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--3193--3882-a6ce39.svg)](https://orcid.org/0009-0003-3193-3882)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**Author:** Muhammad Nadeem  
**Affiliation:** Computational Oncology and Bioinformatics Research, Faisalabad 38000, Punjab, Pakistan  
**Corresponding Author:** `muhmmad.nadeem.bioinfo@gmail.com` | **ORCID:** [0009-0003-3193-3882](https://orcid.org/0009-0003-3193-3882)

---

## Overview

Multi-omics prognostic models in breast cancer frequently fail external validation due to cross-platform batch effects, training-to-test data leakage, pseudogene probe artifacts, and uncalibrated survival probability estimates. Adhering to **TRIPOD+AI** and **SurvBoard** evaluation standards, this repository provides a leakage-free, platform-invariant three-omics survival framework integrating:

1. **Focal Copy-Number Alterations (`CNAs`)** and **Tumor Mutational Burden (`TMB`)**
2. **Somatic Driver Mutations (`SNVs`)**
3. **Within-Sample Z-Score Standardized Protein-Coding mRNA Expression** and **MSigDB Hallmark `ssGSEA` Pathway Activity Scores**

All models were trained and locked strictly on discovery data without external re-tuning and validated across **$N = 3,186$** primary breast cancer patients spanning three independent cohorts:

| Cohort | Sample Size ($n$) | Events / Clinical Endpoints | Platform & Role |
| :--- | :---: | :--- | :--- |
| **TCGA-BRCA** (`brca_tcga_pan_can_atlas_2018`) | `994` | `136` OS events (`13.7%`); locked 80/20 split (`Train n = 795`, `Test n = 199`) + 5-fold stratified OOF CV | RNA-seq + GISTIC2 CNA + MAF SNV (Discovery & Internal Validation) |
| **METABRIC** (`brca_metabric`) | `1,865` | `1,091` OS events (`58.5%`), `615` breast-cancer-specific DSS events (`33.0%`); median follow-up `116.1` months | Illumina HT-12 v3 microarray + GISTIC2 CNA + SNV (Primary External 3-Omics Cohort) |
| **GEO GSE20685** (`GSE20685`) | `327` | `83` OS events (`25.4%`); median follow-up `97.2` months | Affymetrix HG-U133 Plus 2.0 microarray (Independent Transcriptomic External Cohort) |

---

## Key Findings

* **Parsimonious 42-Feature Three-Omics Signature:** Elastic-Net Cox regularization (`CoxNet`, $\ell_1/\ell_2 = 0.75$) across `15,473` curated protein-coding genes selected `4` focal CNA drivers (`CNA_ERBB2`, `CNA_CDK4`, `CNA_CDK6`, `CNA_PIK3CA`), `4` somatic mutations (`SNV_TP53`, `SNV_PIK3CA`, `SNV_KMT2C`, `SNV_AKT1`), `7` MSigDB Hallmark pathways, and `27` protein-coding transcripts.
* **Cross-Cohort Performance:**
  * **TCGA-BRCA 5-Fold Out-of-Fold CV ($n = 994$):** `Clinico-3Omics` achieved $\text{C-index} = 0.7185$ ($95\%$ CI: $0.6676\text{--}0.7715$) and mean $1\text{--}5$ year dynamic ROC-AUC $= 0.7260$.
  * **External METABRIC Breast-Cancer-Specific Survival (`DSS`, $n = 1,865$):** `Clinico-3Omics` achieved $\text{DSS C-index} = 0.6413$, mean $1\text{--}5$ year AUC $= 0.6298$, and Integrated Brier Score $\text{IBS} = 0.2597$ (vs. `0.3329` for `Clinical-Only`), significantly outperforming `Clinical-Only` ($\Delta\text{C-index} = +0.0408$, $95\%$ CI: $+0.0173\text{ to }+0.0626$, $1,000$-iteration paired bootstrap $p < 0.001$) and re-trained `Oncotype DX`-gene panel ($\Delta\text{C-index} = +0.0584$, $p < 0.001$).
  * **External GEO GSE20685 ($n = 327$):** Transcriptomic signature achieved $\text{C-index} = 0.6642$ and mean $1\text{--}5$ year AUC $= 0.7005$.
* **Locked Discovery Cutoff (`-0.0077`) & Subgroup Stratification:** Applying the locked `TCGA-Train` median cutoff (`-0.0077`) directly to METABRIC stratified survival at log-rank $p = 3.06 \times 10^{-19}$ (cohort-median split: $p = 4.33 \times 10^{-17}$), remained significant in multivariable Cox regression ($\text{HR} = 1.151$ per SD, $p = 7.48 \times 10^{-6}$), and stratified both **ER+/HER2- Luminal** ($n = 1,159$, $\text{HR} = 1.342$, $p = 4.47 \times 10^{-13}$) and **Triple-Negative Breast Cancer (`TNBC`)** ($n = 299$, $\text{HR} = 1.308$, $p = 7.35 \times 10^{-4}$).
* **Missing-Modality Resilience & Clinical Net Benefit:** Masking `CNA` ($\text{C-index} = 0.5819$), `SNV` ($\text{C-index} = 0.5793$), or all DNA layers ($\text{C-index} = 0.5788$) preserved $>99.3\%$ of full three-omics discrimination ($0.5825$), while $5$-year Decision Curve Analysis (`DCA`, $n = 1,808$) confirmed superior net clinical benefit across $12\%\text{--}38\%$ risk thresholds.
* **Tumor Immune Microenvironment (`TIME`) & Pathway Biology:** High-risk tumors combine `ESTROGEN_RESPONSE_LATE` depletion ($\text{FDR} = 4.27 \times 10^{-44}$) and `E2F_TARGETS` activation ($\text{FDR} = 1.90 \times 10^{-8}$) with marked depletion of `MHC-I/II` antigen presentation ($\text{FDR} = 1.15 \times 10^{-15}$), cytolytic activity (`CYT`, $\text{FDR} = 6.23 \times 10^{-5}$), and `CD8+` T-cell effectors ($\text{FDR} = 3.16 \times 10^{-3}$), alongside regulatory T-cell (`Treg`) enrichment ($\text{FDR} = 6.69 \times 10^{-4}$).

---

## Repository Structure & Figures

* `Fig1_MultiCohort_Benchmark.png` — Multi-cohort cross-platform C-index and $1\text{--}5$ year dynamic ROC-AUC benchmark.
* `Fig2_KaplanMeier_AllCohorts.png` — Kaplan--Meier survival stratification across TCGA-Train, TCGA-Test, METABRIC, and GEO GSE20685.
* `Fig3_Multivariable_Cox_Forest.png` — Multivariable Cox proportional hazards forest plot in external METABRIC ($n = 1,865$).
* `Fig4_SHAP_Interpretability.png` — Permutation SHAP feature attribution across the multi-omics signature in METABRIC.
* `Fig5_Immune_and_Therapeutic_Vulnerability.png` — MSigDB Hallmark pathway enrichment across high-risk vs. low-risk METABRIC strata.
* `Fig6_Decision_Curve_Analysis.png` — $5$-year Decision Curve Analysis (`DCA`) in METABRIC ($n = 1,808$).
* `Fig7_Calibration_and_ModalityDropout.png` — $5$-year decile survival calibration and missing-modality DNA dropout evaluation.
* `Fig8_TIME_Immune_Checkpoints.png` — Tumor Immune Microenvironment (`TIME`) cell-state and checkpoint profiles in METABRIC.

---

## Data Sources & Reproducibility

All primary datasets are publicly available through:
* **cBioPortal:** `brca_tcga_pan_can_atlas_2018` and `brca_metabric`
* **NCBI GEO:** `GSE20685`

---

## How to Cite

```bibtex
@article{nadeem2026brca,
  author  = {Nadeem, Muhammad},
  title   = {{TRIPOD+AI} Compliant Multi-Omics Integration of Copy-Number Alterations, Somatic Mutations, and Pathway Transcriptomics Improves Cancer-Specific Survival Prediction Across 3,186 Breast Cancer Patients},
  journal = {bioRxiv preprint},
  year    = {2026},
  url     = {[https://github.com/Nadeem786087/BRCA-MultiOmics-Prognosis](https://github.com/Nadeem786087/BRCA-MultiOmics-Prognosis)}
}
