# Cross-Dataset Drug Response Prediction (GDSC → DepMap)

Predict compound response (IC50 / AUC / viability) from baseline gene expression profiles and evaluate cross-dataset generalisation.

This repository is designed as a **portfolio-quality translational ML project**:
- strong reproducibility (pinned environment + config-driven runs)
- leak-resistant evaluation (proper CV + external validation)
- interpretable baselines (Elastic Net)
- extensible to tree models / neural nets
- explicit cross-dataset robustness analysis

---

## 1) Biological question

Can baseline transcriptomic profiles of cancer cell lines predict **drug sensitivity**, and do those predictors generalise across independent pharmacogenomic datasets?

Specifically:
- Train on GDSC
- Test on DepMap (PRISM)
- Quantify the generalisation gap

---

## 2) Data sources (public)

This project is set up to use:

- **GDSC (Genomics of Drug Sensitivity in Cancer)**
  - RNA-seq gene expression
  - Drug response (IC50 / AUC)

- **DepMap (Cancer Dependency Map – PRISM Repurposing)**
  - CCLE RNA-seq expression
  - PRISM compound viability readouts

Download pages:
- GDSC portal
- DepMap portal → Downloads

> Note: You must follow dataset-specific attribution and usage policies when using these data.

---

## 3) Repository structure

```
drug-response-prediction-ML/
├── README.md
├── environment.yml
├── config/
│   └── default_config.yaml        # experiment configuration
├── data/
│   ├── raw/
│   │   ├── gdsc/                  # raw GDSC downloads (not committed)
│   │   └── depmap/                # raw DepMap downloads (not committed)
│   ├── interim/
│   │   └── harmonised/            # gene-aligned datasets
│   └── processed/
│       └── per_drug/              # model-ready per-drug matrices
├── models/
│   ├── elasticnet_model.py
│   ├── xgboost_model.py
│   └── mlp_model.py
├── pipelines/
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── metrics.py
│   └── calibration.py
├── scripts/
│   ├── 00_inspect_datasets.py
│   ├── 01_harmonise_genes.py
│   ├── 02_prepare_drug_datasets.py
│   ├── 03_train_all_drugs.py
│   ├── 04_aggregate_results.py
│   ├── 05_calibration_analysis.py
│   ├── 06_extract_elasticnet_coefficients.py
│   ├── 07_shap_analysis.py
│   ├── 08_pathway_enrichment.py
│   └── 09_generate_summary_tables.py
├── results/
│   ├── metrics/
│   ├── plots/
│   ├── shap/
│   └── enrichment/
└── reports/
    └── tables/
```

---

## 4) Quick start

### Create the environment

```bash
conda env create -f environment.yml
conda activate drug_response_env
```

### Place raw datasets

Download GDSC and DepMap files and place them into:

```
data/raw/gdsc/
data/raw/depmap/
```

### Inspect datasets

```bash
python scripts/00_inspect_datasets.py
```

### Harmonise genes and prepare per-drug datasets

```bash
python scripts/01_harmonise_genes.py
python scripts/02_prepare_drug_datasets.py
```

### Train models

```bash
python scripts/03_train_all_drugs.py
```

### Aggregate results and generate plots

```bash
python scripts/04_aggregate_results.py
```

---

## 5) Outputs

- `data/interim/harmonised/`  
  Gene-aligned expression matrices (HGNC harmonised)

- `data/processed/per_drug/`  
  Per-drug training and testing datasets

- `results/metrics/*.json`  
  Evaluation metrics per drug and model

- `results/plots/*.png`  
  Performance distributions and generalisation gap plots

- `results/shap/`  
  SHAP value outputs for tree models

- `results/enrichment/`  
  Pathway enrichment results for top predictive genes

---

## 6) Evaluation framework

- 5-fold cross-validation within GDSC
- External validation on DepMap
- Median performance reporting across drugs
- Paired Wilcoxon statistical comparison
- Calibration curves + Brier score (classification task)

---

## 7) Future improvements

- Compare response endpoints (IC50 vs AUC vs LFC)
- Nested CV + model selection per drug
- Add mutation/CNV features (multi-omics)
- Multi-task learning across drugs
- Stratified evaluation by tissue type
- Cross-dataset pathway consistency analysis