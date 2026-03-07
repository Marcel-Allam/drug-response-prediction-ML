# GDSC Response Schema Summary

## Dataset Overview
- rows: 242036
- columns: 19

## Detected Key Columns
| field                 | detected_column   |
|:----------------------|:------------------|
| drug_name_column      | DRUG_NAME         |
| cell_line_name_column | CELL_LINE_NAME    |
| cosmic_id_column      | COSMIC_ID         |
| ln_ic50_column        | LN_IC50           |
| auc_column            | AUC               |

## Missing Values (Top 20 Columns)
| column_name     |   missing_count |
|:----------------|----------------:|
| PUTATIVE_TARGET |           27155 |
| TCGA_DESC       |            1067 |
| DATASET         |               0 |
| NLME_CURVE_ID   |               0 |
| NLME_RESULT_ID  |               0 |
| CELL_LINE_NAME  |               0 |
| COSMIC_ID       |               0 |
| SANGER_MODEL_ID |               0 |
| DRUG_ID         |               0 |
| DRUG_NAME       |               0 |
| PATHWAY_NAME    |               0 |
| COMPANY_ID      |               0 |
| WEBRELEASE      |               0 |
| MIN_CONC        |               0 |
| MAX_CONC        |               0 |
| LN_IC50         |               0 |
| AUC             |               0 |
| RMSE            |               0 |
| Z_SCORE         |               0 |

## Full Schema Table
| column_name     | dtype   |   non_null_count |   missing_count |   missing_proportion |
|:----------------|:--------|-----------------:|----------------:|---------------------:|
| DATASET         | str     |           242036 |               0 |           0          |
| NLME_RESULT_ID  | int64   |           242036 |               0 |           0          |
| NLME_CURVE_ID   | int64   |           242036 |               0 |           0          |
| COSMIC_ID       | int64   |           242036 |               0 |           0          |
| CELL_LINE_NAME  | str     |           242036 |               0 |           0          |
| SANGER_MODEL_ID | str     |           242036 |               0 |           0          |
| TCGA_DESC       | str     |           240969 |            1067 |           0.00440844 |
| DRUG_ID         | int64   |           242036 |               0 |           0          |
| DRUG_NAME       | str     |           242036 |               0 |           0          |
| PUTATIVE_TARGET | str     |           214881 |           27155 |           0.112194   |
| PATHWAY_NAME    | str     |           242036 |               0 |           0          |
| COMPANY_ID      | int64   |           242036 |               0 |           0          |
| WEBRELEASE      | str     |           242036 |               0 |           0          |
| MIN_CONC        | float64 |           242036 |               0 |           0          |
| MAX_CONC        | float64 |           242036 |               0 |           0          |
| LN_IC50         | float64 |           242036 |               0 |           0          |
| AUC             | float64 |           242036 |               0 |           0          |
| RMSE            | float64 |           242036 |               0 |           0          |
| Z_SCORE         | float64 |           242036 |               0 |           0          |