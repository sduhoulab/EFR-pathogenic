# **LMEFold: A Deep Learning Framework for Early Folding Residue Prediction and Disease Association Analysis**
## **Workflow**
![5d1acee5a92e6f8e6c24944b2813f7c2](https://github.com/user-attachments/assets/0eb80cac-3e3f-4e21-bc06-a8a6275390cb)


We established an integrated deep learning–biophysics–clinical framework to elucidate the role of Early Folding Residues (EFRs) in protein stability and disease.

**Model development:** LMEFold leverages ESM-2 embeddings to extract sequence-derived structural signals and is optimized using nested cross-validation.

**Benchmarking and validation:** The model was systematically compared with state-of-the-art sequence-based predictors and general PLMs, with evaluation spanning predictive performance, representation analysis, structural consistency, and biophysical generalization on an independent HDX-based dataset.

**Clinical application:** LMEFold was applied at scale to population and disease variant datasets to characterize the enrichment and clinical relevance of mutations occurring in EFRs.

## **Datasets**
### Training and Biophysical Validation
**Dataset 1 (Training - Benchmark30):** Derived from the Start2Fold database. Contains 30 proteins with high-quality experimental annotations (HDX/NMR) defining early folding sites.

Source: https://www.bio2byte.be/start2fold/


**Dataset 2 (External Validation - PF-HDX-EvalSet):** Consists of 431 residues from 10 proteins with experimentally determined protection factors (PFs) measured via HDX-NMR to validate biophysical relevance.

Source: https://pubmed.ncbi.nlm.nih.gov/34739840/.


**Dataset 3 (Case Studies):** Structural data for myoglobin (PDB: 1MBC) and ubiquitin (PDB: 1UBQ).

Source: https://www.rcsb.org/

### Genomic and Clinical Cohorts
**Dataset 4 (Germline Variant Database):**

**Pathogenic Variants:** 23,369 pathogenic missense variants were extracted and retained from ClinVar.

Source: www.ncbi.nlm.nih.gov/clinvar/


**Population Control:** ~6.6 million common variants from gnomAD and UK Biobank (UKB).


gnomAD: A total of 4,667,175 variants were extracted and retained.


Source: https://gnomad.broadinstitute.org/


UK Biobank (UKB): A total of 1,952,086 variants were extracted and retained.


Source: https://www.ukbiobank.ac.uk/


**Dataset 5 (Somatic Variant  Database):**

MSK-MET Cohort:A total of 129,411 somatic non-synonymous single nucleotide variants were extracted, along with corresponding clinical records.

Source: https://www.cbioportal.org/

## **Methods**
### Core Framework
**LMEFold:** The proposed framework based on the ESM-2 protein language model for EFR prediction.

Repository: https://huggingface.co/LMEFold , which hosts the pretrained LMEFold models and related resources generated in this study.

### Baselines & Comparators
**ESM-2:** Pre-trained evolutionary scale modeling.

Source: https://github.com/facebookresearch/esm

**ProtTrans (ProtBERT / ProtT5):** Transformer models trained on protein sequences.

Source: https://github.com/agemagician/ProtTrans

**EFoldMine:** The state-of-the-art sequence-based predictor for early folding residues using handcrafted features.

Source:  https://figshare.com/articles/EFoldMine_code/5649373 



### Variant Effect Prediction Tools
**FoldX 5.0:** Used for calculating folding free energy changes (ΔΔG).

Source: http://foldxsuite.crg.eu/

**VEP (Variant Effect Predictor):** Used for mapping genomic variants to protein residues and predicting their molecular consequences.

Source: https://www.ensembl.org/info/docs/tools/vep/index.html


**ConSurf:** Used for identifying functional regions in proteins by estimating evolutionary conservation through phylogenetic analysis.

Source: https://consurf.tau.ac.il/


**ConSurf 2016:** An improved methodology for estimating and visualizing evolutionary conservation in macromolecules.

Source: https://consurf.tau.ac.il/


**IUPred3:** Used for predicting intrinsic protein disorder, enhanced with experimental annotations.

Source: https://iupred3.elte.hu/


**SIFT:** Used for predicting whether an amino acid substitution affects protein function based on sequence homology.

Source: https://sift.bii.a-star.edu.sg/

**CD-HIT:** Used for clustering and comparing large sets of protein or nucleotide sequences to reduce redundancy.

Source: http://weizhong-lab.ucsd.edu/cd-hit/


**PolyPhen-2:** Used for predicting the potential impact of missense mutations on protein structure and function.

Source: http://genetics.bwh.harvard.edu/pph2/
