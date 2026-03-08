# Drug Response Prediction from Transcriptomic Profiles

Machine learning pipeline for predicting **cancer drug response (IC50)** from **gene expression data**, followed by **biological pathway enrichment analysis** to interpret model features.

This project demonstrates how **machine learning models trained on transcriptomic data can identify biologically meaningful pathways associated with drug sensitivity.**

The pipeline integrates:

- transcriptomic datasets
- pharmacological response measurements
- ElasticNet regression modelling
- pathway enrichment analysis
- reproducible bioinformatics workflows

---

# Project Objective

The goal of this project is to determine whether **gene expression profiles can predict cancer cell line sensitivity to targeted therapies** and to interpret the resulting predictive features biologically.

Specifically, this project:

1. Trains a **regularized regression model (ElasticNet)** to predict drug response
2. Identifies genes most strongly associated with drug sensitivity
3. Performs **pathway enrichment analysis** on model-selected genes
4. Generates publication-style plots summarizing enriched pathways

The project focuses on the MEK inhibitor **Selumetinib** as a case study.

---

# Biological Motivation

Drug response in cancer is driven by complex molecular programs involving:

- signalling pathway activation
- immune response
- metabolic state
- transcriptional regulation

While machine learning models can predict drug sensitivity, **interpreting the biological meaning of model features is essential**.

This project therefore integrates **machine learning with pathway enrichment** to determine whether predictive genes converge on meaningful biological processes.

---

# Data Sources

This project uses publicly available pharmacogenomics datasets derived from large-scale cancer cell line screens.

Typical data sources used in these studies include:

- **GDSC (Genomics of Drug Sensitivity in Cancer)**
- **CCLE (Cancer Cell Line Encyclopedia)**

These datasets contain:

- genome-wide gene expression profiles
- drug response measurements (IC50)
- hundreds of cancer cell lines

For each drug, the pipeline constructs a **per-drug dataset combining gene expression features with drug response values**.

---

# Pipeline Overview

The analysis pipeline consists of the following steps.

## 1. Data Integration

Gene expression data are merged with drug response measurements.

Output:

```
data/processed/per_drug/<drug>.parquet
```

Each dataset contains:

- gene expression features
- LN_IC50 drug response values

---

## 2. Machine Learning Model

An **ElasticNet regression model** is trained to predict drug response.

ElasticNet was chosen because it:

- performs feature selection
- handles high-dimensional transcriptomic data
- balances L1 and L2 regularisation

Model evaluation metrics:

- **R²**
- **RMSE**

Example result (Selumetinib):

```
R²   = 0.52
RMSE = 1.30
```

Outputs:

```
results/models/<drug>_elasticnet.joblib
results/metrics/<drug>_elasticnet_metrics.json
```

---

## 3. Feature Extraction

Genes with **non-zero model coefficients** are extracted as candidate predictors of drug response.

Two ranked feature sets are generated:

- **Top model features**
- **All non-zero coefficients**

Outputs:

```
results/metrics/<drug>_elasticnet_top_features.csv
results/metrics/<drug>_elasticnet_all_nonzero_features.csv
```

Gene symbols are then extracted for downstream analysis.

---

## 4. Pathway Enrichment Analysis

Model-derived gene lists are analysed using **Enrichr via gseapy** to identify enriched biological pathways.

Databases queried:

- KEGG
- Reactome
- Gene Ontology (Biological Process)

Outputs:

```
results/enrichment/<drug>_kegg_enrichment.csv
results/enrichment/<drug>_reactome_enrichment.csv
results/enrichment/<drug>_gobp_enrichment.csv
```

Significant results:

```
results/enrichment/*_enrichment_sig.csv
```

---

## 5. Visualization

Pathway enrichment results are visualized as **bar plots of –log10(adjusted p-values)**.

Outputs:

```
results/plots/<drug>_kegg_barplot.png
results/plots/<drug>_reactome_barplot.png
results/plots/<drug>_gobp_barplot.png
```

These figures summarise the most enriched biological processes associated with model-selected genes.

---

# Key Result (Selumetinib)

Pathway enrichment identified a significant biological process:

```
Defense Response To Gram-negative Bacterium
Adjusted p-value = 0.00618
```

Genes contributing to this enrichment include:

```
IL23A
DEFA4
DEFA3
CTSG
DEFA1
LYZ
TLR4
LYPD8
```

These genes are involved in **innate immune signalling and inflammatory responses**, suggesting that transcriptional immune programs may be associated with Selumetinib sensitivity across cell lines.

While further validation is required, this demonstrates that **machine learning-derived gene sets can recover biologically meaningful pathways.**

---

# Repository Structure

```
drug-response-prediction-ML

config/
    default_config.yaml

data/
    processed/

scripts/
    04_train_elasticnet_model.py
    05_extract_gene_symbols.py
    06_run_pathway_enrichment.py
    07_plot_enrichment_results.py

results/
    enrichment/
    metrics/
    models/
    plots/
```

---

# Example Usage

Train the model:

```bash
python scripts/04_train_elasticnet_model.py \
--config config/default_config.yaml \
--drug "Selumetinib"
```

Extract gene symbols:

```bash
python scripts/05_extract_gene_symbols.py \
--drug "Selumetinib" \
--source all_nonzero_features \
--top_n 250
```

Run pathway enrichment:

```bash
python scripts/06_run_pathway_enrichment.py \
--drug "Selumetinib" \
--source all_nonzero_features \
--top_n 250
```

Generate plots:

```bash
python scripts/07_plot_enrichment_results.py \
--drug "Selumetinib"
```

---

# Technologies Used

Python ecosystem:

- pandas
- numpy
- scikit-learn
- matplotlib
- gseapy
- PyYAML
- joblib

Bioinformatics concepts:

- transcriptomics
- pharmacogenomics
- pathway enrichment
- machine learning for omics data

---

# Future Improvements

Potential extensions of this project include:

- training models for multiple drugs
- cross-drug prediction benchmarking
- feature stability analysis
- SHAP model interpretation
- integration of mutation or copy-number features
- drug response classification models

---

# Why This Project Matters

Predicting drug response from molecular data is a key goal of **precision oncology**.

This project demonstrates how:

- machine learning models can be applied to high-dimensional transcriptomic data
- model features can be interpreted biologically
- computational pipelines can link predictive modelling with pathway-level insights.