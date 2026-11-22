# BRCA-MultiOmics-Prognosis

**TRIPOD+AI Compliant Multi-Omics Integration of Copy-Number Alterations, Somatic Mutations, and Pathway Transcriptomics Across 3,186 Primary Breast Cancer Patients**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--3193--3882-A6CE39.svg)](https://orcid.org/0009-0003-3193-3882)
[![TRIPOD+AI](https://img.shields.io/badge/Reporting-TRIPOD%2BAI-red.svg)](manuscript/BRCA_MultiOmics_Manuscript_FINAL.md)

**Author:** Muhammad Nadeem  
**Affiliation:** Department of Bioinformatics and Biotechnology, Government College University Faisalabad, Faisalabad, Pakistan  
**Contact:** muhmmad.nadeem.bioinfo@gmail.com | [LinkedIn](https://linkedin.com/in/mnadeembioinfo) | [ORCID](https://orcid.org/0009-0003-3193-3882)

---

## Overview
\BRCA-MultiOmics-Prognosis\ implements a strictly leakage-free, platform-invariant three-omics survival framework integrating **GISTIC2 focal copy-number alterations (CNAs)**, **somatic driver mutations (SNVs)**, **tumor mutational burden (TMB)**, **within-sample standardized protein-coding mRNA expression**, and **MSigDB Hallmark ssGSEA pathway activity scores** across three independent primary breast cancer cohorts ( = 3,186$):

| Cohort | Platform | Sample Size ($) | Survival Endpoints | Study Role |
| :--- | :--- | :---: | :---: | :--- |
| **TCGA-BRCA** | Illumina RNA-seq + WES + GISTIC2 | 994 | OS (136 events) | Discovery (5-Fold OOF & 80/20 Train/Test) |
| **METABRIC** | Illumina HT-12 + Targeted Seq + GISTIC2 | 1,865 | OS (1,091) & DSS (615) | Primary 3-Omics External Validation |
| **GEO GSE20685** | Affymetrix U133 Plus 2.0 | 327 | OS (83 events) | External Transcriptomic Validation |

---

## Key Empirical Findings
1. **Transparent 42-Feature Three-Omics Signature:** Elastic-Net Cox regression (\CoxNet\) selected 4 focal CNA drivers (\CNA_ERBB2\, \CNA_CDK4\, \CNA_CDK6\, \CNA_PIK3CA\), 4 somatic mutations (\SNV_TP53\, \SNV_PIK3CA\, \SNV_KMT2C\, \SNV_AKT1\), 7 MSigDB Hallmark pathways, and 27 curated protein-coding transcripts (exact coefficients in \esults/v3_tripod_gold/Table_Exact_Signature_Weights.csv\).
2. **External Breast-Cancer-Specific Survival (DSS) & Calibration:** In external METABRIC validation ( = 1,865$), \Clinico-3Omics\ fusion achieved superior cancer-specific mortality discrimination ($\text{DSS C-index} = 0.6413$, $\text{Mean 1–5 yr AUC} = 0.6298$, $\text{Integrated Brier Score} = 0.2597$), significantly outperforming \Clinical-Only\ ($\Delta\text{C-index} = +0.0408$, 1,000-bootstrap  < 0.001$) and re-trained \Oncotype DX\ / \PAM50\ gene panels.
3. **Zero-Leakage Cutoff & Receptor Subgroup Generalization:** Applying the locked \TCGA-Train\ median cutoff (\-0.0077\) directly to METABRIC stratified overall survival at log-rank  = 3.06 \times 10^{-19}$ and independently predicted mortality within both **ER+/HER2- Luminal** ( = 1,159$, $\text{HR} = 1.342$,  = 4.47 \times 10^{-13}$) and **Triple-Negative Breast Cancer (TNBC)** ( = 299$, $\text{HR} = 1.308$,  = 7.35 \times 10^{-4}$).
4. **Missing-Modality Resilience & Clinical Net Benefit:** Masking all DNA layers (\CNA + SNV\) in METABRIC preserved >99.3% of full 3-omics discrimination ($\Delta\text{C-index} = -0.0037$), while 5-year Decision Curve Analysis confirmed higher net clinical benefit across 12%–38% mortality risk thresholds.
5. **Tumor Immune Microenvironment (TIME) Deconvolution:** High-risk tumors exhibit marked depletion of \MHC-I/II\ antigen presentation ($\text{FDR} = 1.15 \times 10^{-15}$), cytolytic activity ($\text{FDR} = 6.23 \times 10^{-5}$), and CD8+ T-cell effectors ($\text{FDR} = 3.16 \times 10^{-3}$) paired with \E2F\ cell-cycle hyperactivation ($\text{FDR} = 1.90 \times 10^{-8}$).

---

## Visual Summary

### Multi-Cohort Kaplan–Meier Survival Stratification
![Kaplan-Meier Survival](results/figures/Fig2_KaplanMeier_AllCohorts.png)

### 5-Year Calibration & Missing-Modality Resilience (TRIPOD+AI)
![Calibration and Dropout](results/v3_tripod_gold/Fig7_Calibration_and_ModalityDropout.png)

### Tumor Immune Microenvironment (TIME) & Checkpoint Deconvolution
![TIME Deconvolution](results/v3_tripod_gold/Fig8_TIME_Immune_Checkpoints.png)

---

## Repository Structure
\\\	ext
BRCA-MultiOmics-Prognosis/
├── src/
│   ├── 01_download_and_qc.py              # Automated data retrieval from cBioPortal & NCBI GEO
│   ├── 02_harmonize_all_cohorts.py        # Cross-platform gene universe & clinical alignment
│   ├── fix_gse20685.py                    # GEO GSE20685 phenotype & probe harmonization
│   ├── 03b_optimized_multiomics.py        # Univariate Cox screening & multi-algorithm benchmark
│   ├── 04_figures_and_translation.py      # KM curves, multivariable Cox, SHAP & Hallmark pathways
│   ├── 05_expert_upgrade_pipeline.py      # 3-Omics (CNA+SNV+mRNA), commercial panels, subgroups & DCA
│   └── 06_tripod_ai_final_validation.py   # DSS, IBS calibration, paired bootstrap, dropout & TIME
├── results/
│   ├── figures/                           # 300-DPI PNG & vector PDF figures (Fig1–Fig5)
│   ├── v2_expert/                         # Decision Curve Analysis (Fig6) & commercial benchmarks
│   └── v3_tripod_gold/                    # Calibration & Dropout (Fig7), TIME (Fig8) & exact weights
├── manuscript/
│   └── BRCA_MultiOmics_Manuscript_FINAL.md
├── CITATION.cff
├── LICENSE
└── requirements.txt
\\\

---

## Step-by-Step Reproduction
\\\ash
git clone https://github.com/Nadeem786087/BRCA-MultiOmics-Prognosis.git
cd BRCA-MultiOmics-Prognosis

conda create -n brca_py python=3.11 -y
conda activate brca_py
pip install -r requirements.txt

python src/01_download_and_qc.py
python src/02_harmonize_all_cohorts.py
python src/fix_gse20685.py
python src/03b_optimized_multiomics.py
python src/04_figures_and_translation.py
python src/05_expert_upgrade_pipeline.py
python src/06_tripod_ai_final_validation.py
\\\
"@ | Set-Content -Encoding UTF8 README.md

# 7. Create final TRIPOD+AI manuscript file (manuscript/BRCA_MultiOmics_Manuscript_FINAL.md)
@"
# TRIPOD+AI Compliant Multi-Omics Integration of Copy-Number Alterations, Somatic Mutations, and Pathway Transcriptomics Improves Cancer-Specific Survival Prediction Across 3,186 Breast Cancer Patients

**Muhammad Nadeem**  
Department of Bioinformatics and Biotechnology, Government College University Faisalabad, Faisalabad, Pakistan  
ORCID: https://orcid.org/0009-0003-3193-3882 | Correspondence: muhmmad.nadeem.bioinfo@gmail.com  

---

## Abstract

**Background:** Multi-omics prognostic models in breast cancer frequently fail external validation due to cross-platform batch effects, training-to-test data leakage, pseudogene artifacts, and uncalibrated probability estimates. Following TRIPOD+AI and SurvBoard reporting standards, we developed and externally validated a platform-invariant three-omics survival framework integrating focal copy-number alterations (CNAs), somatic mutations (SNVs), tumor mutational burden (TMB), curated protein-coding mRNA expression, and MSigDB Hallmark pathway activity scores across N = 3,186 primary breast cancer patients.  
**Methods:** Three independent cohorts were harmonized across 15,473 protein-coding genes: TCGA-BRCA (n = 994; 5-fold stratified out-of-fold cross-validation plus a locked 80/20 discovery [n = 795] and internal holdout [n = 199] split), METABRIC (n = 1,865; overall survival [OS] and breast-cancer-specific survival [DSS]), and GEO GSE20685 (n = 327). Within-sample Z-score normalization and single-sample Gene Set Enrichment Analysis (ssGSEA) were paired with Elastic-Net Cox regression (CoxNet), hierarchical block-wise late fusion, Random Survival Forest (RSF), and Gradient Boosted Survival Analysis (GBSA). Models were benchmarked against clinical covariates (Age + Stage/NPI) and re-trained Oncotype DX and PAM50 gene panels using 1,000-iteration paired bootstrap C-index tests, time-dependent 1-5 year dynamic ROC-AUC, Integrated Brier Score (IBS), 5-year calibration curves, missing-modality dropout stress tests, and Decision Curve Analysis (DCA).  
**Results:** Penalized Elastic-Net Cox modeling selected a transparent 42-feature three-omics signature comprising 4 focal CNA drivers (CNA_ERBB2, CNA_CDK4, CNA_CDK6, CNA_PIK3CA), 4 somatic driver mutations (SNV_TP53, SNV_PIK3CA, SNV_KMT2C, SNV_AKT1), 7 Hallmark pathways, and 27 protein-coding transcripts. Across full TCGA-BRCA 5-fold out-of-fold validation (n = 994), Clinico-3Omics fusion achieved a C-index of 0.7185 (95% CI: 0.6676-0.7715) and mean 1-5 year AUC of 0.7260. In external METABRIC validation (n = 1,865), Clinico-3Omics achieved superior discrimination and probability calibration on breast-cancer-specific survival (DSS C-index = 0.6413, Mean AUC = 0.6298, IBS = 0.2597), significantly outperforming Clinical-Only (Delta C-index = +0.0408, 95% CI: +0.0173 to +0.0626, paired bootstrap p < 0.001) and re-trained Oncotype DX (Delta C-index = +0.0584, p < 0.001). Applying the locked TCGA-Train median cutoff (-0.0077) without external re-tuning stratified METABRIC survival at log-rank p = 3.06e-19, retaining independent significance in multivariable Cox regression (HR = 1.151 per SD, p = 7.48e-06) and within both ER+/HER2- luminal (n = 1,159, HR = 1.342, p = 4.47e-13) and triple-negative (n = 299, HR = 1.308, p = 7.35e-04) subtypes. Modality dropout testing confirmed resilience when DNA layers were masked (Delta C-index = -0.0037), and 5-year DCA demonstrated higher net clinical benefit across 12%-38% risk thresholds. Tumor immune microenvironment deconvolution revealed that high-risk tumors exhibit marked depletion of MHC-I/II antigen presentation (FDR = 1.15e-15), cytolytic activity (FDR = 6.23e-05), and CD8+ T-cell effectors (FDR = 3.16e-03) alongside regulatory T-cell enrichment (FDR = 6.69e-04).  
**Conclusions:** Three-layer multi-omics integration delivers calibrated, cross-platform prognostic utility superior to single-omics panels and delineates an endocrine-resistant, antigen-presentation-deficient, cell-cycle-driven high-risk breast cancer phenotype.

---

## 1. Introduction
Breast cancer progression is governed by interacting chromosomal copy-number alterations (CNAs), somatic driver mutations, and transcriptional state transitions. Conventional clinicopathological indices such as the Nottingham Prognostic Index (NPI) and TNM staging capture anatomical tumor burden but omit underlying molecular vulnerabilities, whereas commercial mRNA panels (Oncotype DX, PAM50) omit structural DNA alterations and lack prognostic utility in triple-negative breast cancer (TNBC). Adhering to TRIPOD+AI and SurvBoard standards, we developed and externally validated a leakage-free three-omics survival framework across N = 3,186 primary breast cancer patients.

## 2. Materials and Methods
Three primary breast cancer cohorts (TCGA-BRCA n = 994; METABRIC n = 1,865; GEO GSE20685 n = 327) were harmonized across 15,473 curated protein-coding genes. Expression values were standardized within each individual sample, MSigDB Hallmark pathway activity scores were computed via ssGSEA, and somatic genomic alterations were encoded across 11 driver SNVs, log2(TMB+1), and 15 GISTIC2 focal CNA drivers. Models were trained strictly on TCGA-Train and evaluated externally using 1,000-iteration paired bootstrap C-index tests, dynamic ROC-AUC, Integrated Brier Score (IBS), decile calibration curves, modality-dropout stress tests, Decision Curve Analysis (DCA), SHAP attribution, and Tumor Immune Microenvironment (TIME) deconvolution.

## 3. Results & Discussion
Elastic-Net Cox regularization selected a 42-feature signature (exact weights archived in results/v3_tripod_gold/Table_Exact_Signature_Weights.csv). Clinico-3Omics fusion achieved C-index = 0.7185 in TCGA-BRCA 5-fold cross-validation and DSS C-index = 0.6413 (IBS = 0.2597) in external METABRIC validation, significantly outperforming Clinical-Only (p < 0.001) and re-trained Oncotype DX (p < 0.001). High-risk tumors demonstrated concurrent suppression of late estrogen response (FDR = 4.27e-44) and MHC-I/II antigen presentation (FDR = 1.15e-15) alongside E2F cell-cycle hyperactivation (FDR = 1.90e-08) and CDK4/6-PIK3CA copy-number gains.

## 4. Data and Code Availability
All code, model weights, and 300-DPI figures are openly archived at https://github.com/Nadeem786087/BRCA-MultiOmics-Prognosis.
