# Multi-Dataset b/c Truth Model

## Scope

- Datasets: `Q10R`, `WT_2025-02`, `WT_2025-03`, `S20G`
- Family groups: `Q10R`, `WT`, `S20G`
- Only `b/c` ions
- Truth: manual `Matched=1`
- Precursor lock-mass / chain-to-fragments: `off` during feature extraction
- Existing `truth_score`: `off` during feature extraction to avoid using the previous learned score as an input
- Models compared: `LogisticRegression` and `RandomForestClassifier`

## Dataset

- Total candidates: `538`
- Positives: `351`
- Negatives: `187`
- Candidates in `Q10R`: `238`
- Candidates in `S20G`: `129`
- Candidates in `WT`: `171`

## Model Features

- Features: `fragments_css, fragments_rawcos, fragments_fit_score, fragments_correlation, fragments_pc_missing_peaks, fragments_match_count, fragments_abs_anchor_ppm, diagnose_isodec_css, diagnose_raw_cosine, diagnose_matched_peaks, diagnose_area_covered, diagnose_top_peaks`

## Model Comparison

- `random_forest` scan-holdout: AUC `0.827`, F1@0.5 `0.834`, best-F1 `0.844` @ `0.349860`
- `random_forest` family-holdout: AUC `0.708`, F1@0.5 `0.764`, best-F1 `0.811` @ `0.346027`
- `logistic_regression` scan-holdout: AUC `0.800`, F1@0.5 `0.780`, best-F1 `0.824` @ `0.401379`
- `logistic_regression` family-holdout: AUC `0.699`, F1@0.5 `0.693`, best-F1 `0.793` @ `0.211726`

## Best Model

- Selected by family-holdout best-F1: `random_forest`

## Strongest Single Features

- `diagnose_abs_anchor_ppm`: AUC `0.640`, Spearman `-0.231`, best direction `lower`
- `fragments_abs_anchor_ppm`: AUC `0.632`, Spearman `-0.219`, best direction `lower`
- `diagnose_top_peaks`: AUC `0.632`, Spearman `0.252`, best direction `higher`
- `diagnose_area_covered`: AUC `0.618`, Spearman `0.196`, best direction `higher`
- `fragments_css`: AUC `0.591`, Spearman `-0.151`, best direction `lower`
- `fragments_correlation`: AUC `0.569`, Spearman `-0.114`, best direction `lower`
- `fragments_fit_score`: AUC `0.553`, Spearman `-0.089`, best direction `lower`
- `fragments_obs_rel_int`: AUC `0.553`, Spearman `0.087`, best direction `higher`

## Files

- Dataset: `reports/multiset_bc_truth_model/multiset_bc_truth_dataset.csv`
- Model comparison: `reports/multiset_bc_truth_model/model_comparison.csv`
- Univariate stats: `reports/multiset_bc_truth_model/univariate_stats.csv`
- Scan CV predictions: `reports/multiset_bc_truth_model/scan_cv_predictions.csv`
- Family CV predictions: `reports/multiset_bc_truth_model/family_cv_predictions.csv`
- Feature importance: `reports/multiset_bc_truth_model/model_coefficients.csv`
- Model artifact index: `reports/multiset_bc_truth_model/model_artifact.json`
