# BRCA-MultiOmics-Prognosis

**TRIPOD+AI Compliant Multi-Omics Integration of Copy-Number Alterations, Somatic Mutations, and Pathway Transcriptomics Across 3,186 Primary Breast Cancer Patients**

**Author:** Muhammad Nadeem ([ORCID: 0009-0003-3193-3882](https://orcid.org/0009-0003-3193-3882))  
**Affiliation:** Department of Bioinformatics and Biotechnology, Government College University Faisalabad, Faisalabad, Pakistan  

## Key Highlights
- **Multi-Cohort Harmonization (N = 3,186):** TCGA-BRCA (n = 994), METABRIC (n = 1,865), and GEO GSE20685 (n = 327) across 15,473 curated protein-coding genes.
- **Three-Layer Omics Signature (42 Features):** Integrates focal GISTIC2 CNAs (\ERBB2\, \CDK4\, \CDK6\, \PIK3CA\), somatic driver mutations (\TP53\, \PIK3CA\, \KMT2C\, \AKT1\), 7 MSigDB Hallmark pathways, and 27 protein-coding transcripts.
- **External Breast-Cancer-Specific Survival (DSS) & Calibration:** Achieves \DSS C-index = 0.6413\ and \IBS = 0.2597\ in METABRIC, outperforming \Clinical-Only\ (\Delta C-index = +0.0408\, paired bootstrap \p < 0.001\) and re-trained \Oncotype DX\ / \PAM50\ gene panels.
- **Subgroup Generalization & TIME Deconvolution:** Independently stratifies \ER+/HER2-\ (\p = 4.47e-13\) and \TNBC\ (\p = 7.35e-04\) subtypes and reveals \MHC-I/II\ antigen-presentation depletion (\FDR = 1.15e-15\) in high-risk tumors.
