# **LMEFold: A Deep Learning Framework for Early Folding Residue Prediction and Disease Association Analysis**
## **Workflow**
![5d1acee5a92e6f8e6c24944b2813f7c2](https://github.com/user-attachments/assets/0eb80cac-3e3f-4e21-bc06-a8a6275390cb)


We established a comprehensive framework integrating deep learning with biophysical and clinical validation to decode the role of Early Folding Residues (EFRs) in protein stability and disease.

**Model Construction:** We developed LMEFold using embeddings from the ESM-2 protein language model to capture latent structural information from sequence alone.

**Benchmarking:** The model was optimized via a nested cross-validation scheme and evaluated against state-of-the-art sequence-based predictors (e.g., EFoldMine) and general PLMs (ProtBERT, ProtT5). Metrics included AUC, Pearson correlation (with HDX-NMR data), and spatial consistency with native structures.

**Clinical Application:** We applied LMEFold to annotate over 6.6 million variants across diverse genomic datasets. We systematically assessed the enrichment of pathogenic mutations in EFRs and, using the MSK-MET pan-cancer cohort, investigated the association between EFR mutations and patient survival outcomes.

## **Datasets**
### Training and Biophysical Validation
**Dataset 1 (Training - Benchmark30):** Derived from the Start2Fold database. Contains 30 proteins with high-quality experimental annotations (HDX/NMR) defining early folding sites.

Source: https://huggingface.co/LMEFold/LMEFold


**Dataset 2 (External Validation - PF-HDX-EvalSet):** Consists of 431 residues from 10 proteins with experimentally determined protection factors (PFs) measured via HDX-NMR to validate biophysical relevance.

Source: https://pubmed.ncbi.nlm.nih.gov/34739840/.


**Dataset 3 (Case Studies):** Structural data for myoglobin (PDB: 1MBC) and ubiquitin (PDB: 1UBQ).

Source: https://www.rcsb.org/

### Genomic and Clinical Cohorts
**Dataset 4 (Germline Variant Database):**

**Pathogenic Variants:** 23,369 pathogenic missense variants were extracted and retained from ClinVar.

Source: www.ncbi.nlm.nih.gov/clinvar/


**Population Control:** ~6.6 million common variants from gnomAD and UK Biobank (UKB).


**gnomAD:** A total of 4,667,175 variants were extracted and retained.


Source: https://gnomad.broadinstitute.org/


**UK Biobank (UKB):** A total of 1,952,086 variants were extracted and retained.


Source: https://www.ukbiobank.ac.uk/


**Dataset 5 (Somatic Variant  Database):**

**MSK-MET Cohort:** A total of 129,411 somatic non-synonymous single nucleotide variants were extracted, along with corresponding clinical records.

Source: https://www.cbioportal.org/

## **Methods**
### Core Framework
**LMEFold:** The proposed framework based on the ESM-2 protein language model (specifically esm2_t30_150M_UR50D) for EFR prediction.

Repository: https://www.bio2byte.be/start2fold/

### Baselines & Comparators
**ESM-2:** Pre-trained evolutionary scale modeling.

Source: https://github.com/facebookresearch/esm

**ProtTrans (ProtBERT / ProtT5):** Transformer models trained on protein sequences.

Source: https://github.com/agemagician/ProtTrans

**EFoldMine:** The state-of-the-art sequence-based predictor for early folding residues using handcrafted features.

Source:  https://figshare.com/articles/EFoldMine_code/5649373 

### Structural Analysis Tools
**FoldX 5.0:** Used for calculating folding free energy changes (ΔΔG).

Source: http://foldxsuite.crg.eu/


**VEP (Variant Effect Predictor):** Used for mapping genomic variants to protein residues.

Source: https://www.ensembl.org/info/docs/tools/vep/index.html
