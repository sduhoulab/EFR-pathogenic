# **LMEFold: A Deep Learning Framework for Early Folding Residue Prediction and Disease Association Analysis**
## **Workflow**
We established a comprehensive framework integrating deep learning with biophysical and clinical validation to decode the role of Early Folding Residues (EFRs) in protein stability and disease.

**Model Construction:** We developed LMEFold using embeddings from the ESM-2 protein language model to capture latent structural information from sequence alone.

**Benchmarking:** The model was optimized via a nested cross-validation scheme and evaluated against state-of-the-art sequence-based predictors (e.g., EFoldMine) and general PLMs (ProtBERT, ProtT5). Metrics included AUC, Pearson correlation (with HDX-NMR data), and spatial consistency with native structures.

**Clinical Application:** We applied LMEFold to annotate over 6.6 million variants across diverse genomic datasets. We systematically assessed the enrichment of pathogenic mutations in EFRs and, using the MSK-MET pan-cancer cohort, investigated the association between EFR mutations and patient survival outcomes.

## **Datasets**
### Training and Biophysical Validation
**Dataset 1 (Training - Benchmark37):** Derived from the Start2Fold database. Contains 37 proteins with high-quality experimental annotations (HDX/NMR) defining early folding sites.
Source: [sduhoulab/LMEFold at main](https://huggingface.co/datasets/sduhoulab/LMEFold/tree/main/data)


**Dataset 2 (External Validation - PF-HDX-EvalSet):** Consists of 431 residues from 10 proteins with experimentally determined protection factors (PFs) measured via HDX-NMR to validate biophysical relevance.
Source Details: Defined in Table S3 of the manuscript.


**Dataset 3 (Case Studies):** Structural data for E. coli RNase H (PDB: 1F21) and HIV-1 reverse transcriptase (PDB: 1HRH).
Source: RCSB PDB

### Genomic and Clinical Cohorts
**Dataset 4 (Germline Variants):**
**Pathogenic:** 23,370 missense variants from ClinVar.

ClinVar：https://www.ncbi.nlm.nih.gov/clinvar/

**Population Control:** ~6.6 million common variants from gnomAD and UK Biobank (UKB).

gnomAD: https://gnomad.broadinstitute.org/

UK Biobank: https://www.ukbiobank.ac.uk/


**Dataset 5 (Somatic & Clinical - MSK-MET):**
A pan-cancer cohort containing 129,411 somatic nonsynonymous single-nucleotide variants (nsSNVs) with corresponding patient survival data.

Source: [cBioPortal / MSK-MET](https://www.cbioportal.org/)

## **Methods**
### Core Framework
**LMEFold:** The proposed framework based on the ESM-2 protein language model (specifically esm2_t30_650M_UR50D) for EFR prediction.

Repository: https://huggingface.co/datasets/sduhoulab/LMEFold

### Baselines & Comparators
**ESM-2:** Pre-trained evolutionary scale modeling.

Source: https://github.com/facebookresearch/esm

**ProtTrans (ProtBERT / ProtT5):** Transformer models trained on protein sequences.

Source: https://github.com/agemagician/ProtTrans

**EFoldMine:** The state-of-the-art sequence-based predictor for early folding residues using handcrafted features.

Source: http://www.csbio.sjtu.edu.cn/bioinf/EFoldMine/

### Structural Analysis Tools
**FoldX 5.0:** Used for calculating folding free energy changes (ΔΔG).

Source: http://foldxsuite.crg.eu/


**VEP (Variant Effect Predictor):** Used for mapping genomic variants to protein residues.

Source: https://www.ensembl.org/info/docs/tools/vep/index.html
