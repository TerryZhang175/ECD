# Q10R b/c Truth Scoring Analysis

## Scope

- Dataset: Q10R, only `b/c` ions, base key = `RE + ion_type + pos + charge`
- Truth: manual `Matched=1`
- Diagnose metrics: current headless diagnose logic with `h_transfer = 0`
- Candidate universe: all manual-positive bases, all manual-negative annotated bases, plus all algorithm-selected bases that manual truth did not mark positive

## Sample Counts

- Total candidates: `317`
- Truth positive: `185`
- Truth negative: `132`
- `tp_selected`: `132`
- `fn_missed`: `53`
- `fp_selected`: `122`
- `manual_negative_only`: `10`

## Scoring Logic

- `diagnose` final acceptance is driven by `final_cosine >= MIN_COSINE`, with `isodec_css`, anchor ppm, matched peaks, area covered, and top peaks exposed as diagnostic detail rather than a strict copy of fragments gating.
- `fragments` uses a two-layer logic: first a composite/evidence score, then hard gates on CSS, anchor ppm, local matches, coverage, unexplained fraction, missing core, and other quality checks.
- The current fragments evidence score is approximately: `css + coverage + ppm consistency + spacing consistency + intensity prior + fit/correlation/s2n bonuses - unexplained/missing-core/missing-peaks penalties`.

## Strongest Single Features

- `fragments_fit_score`: best-direction AUC `0.839`, Spearman `0.584`, best direction `higher`
- `fragments_score`: best-direction AUC `0.812`, Spearman `0.536`, best direction `higher`
- `fragments_correlation`: best-direction AUC `0.809`, Spearman `0.534`, best direction `higher`
- `diagnose_raw_cosine`: best-direction AUC `0.800`, Spearman `0.513`, best direction `higher`
- `diagnose_isodec_css`: best-direction AUC `0.784`, Spearman `0.484`, best direction `higher`
- `diagnose_final_cosine`: best-direction AUC `0.773`, Spearman `0.458`, best direction `higher`
- `fragments_coverage`: best-direction AUC `0.755`, Spearman `0.443`, best direction `higher`
- `fragments_css`: best-direction AUC `0.755`, Spearman `0.435`, best direction `higher`
- `fragments_pc_missing_peaks`: best-direction AUC `0.731`, Spearman `-0.401`, best direction `lower`
- `fragments_rawcos`: best-direction AUC `0.730`, Spearman `0.393`, best direction `higher`

## Weak Or Low-Value Features

- `diagnose_isodec_accepted`: best-direction AUC `0.569`, Spearman `0.179`
- `diagnose_ok`: best-direction AUC `0.569`, Spearman `0.179`
- `diagnose_matched_peaks`: best-direction AUC `0.570`, Spearman `0.121`
- `fragments_mass_error_std`: best-direction AUC `0.577`, Spearman `-0.133`
- `fragments_match_count`: best-direction AUC `0.592`, Spearman `0.159`
- `fragments_unexplained`: best-direction AUC `0.604`, Spearman `-0.178`
- `diagnose_abs_anchor_ppm`: best-direction AUC `0.607`, Spearman `-0.183`
- `fragments_interference`: best-direction AUC `0.613`, Spearman `-0.195`

## Current Thresholds

- `diagnose_isodec_css` with `>= 0.70`: precision `0.636`, recall `0.929`, F1 `0.755`
- `diagnose_abs_anchor_ppm` with `<= 30.0 ppm`: precision `0.586`, recall `1.000`, F1 `0.739`
- `diagnose_ok` with `ok == 1 (final_cosine >= 0.70)`: precision `0.625`, recall `0.881`, F1 `0.731`
- `diagnose_isodec_accepted` with `diagnose isodec accepted`: precision `0.625`, recall `0.881`, F1 `0.731`
- `diagnose_matched_peaks` with `>= 2`: precision `0.584`, recall `0.962`, F1 `0.727`
- `diagnose_area_covered` with `>= 0.10`: precision `0.596`, recall `0.929`, F1 `0.726`
- `fragments_coverage` with `>= 0.25`: precision `0.560`, recall `0.994`, F1 `0.716`
- `fragments_match_count` with `>= 2`: precision `0.557`, recall `1.000`, F1 `0.716`
- `fragments_s2n` with `>= 0.00`: precision `0.557`, recall `1.000`, F1 `0.716`
- `fragments_interference` with `<= 1.00`: precision `0.557`, recall `1.000`, F1 `0.716`

## Best Composite Score

- Best leave-one-RE-out model: `fragments_correlation, fragments_pc_missing_peaks, diagnose_isodec_css, diagnose_top_peaks`
- Cross-validated AUC: `0.866`
- Full-data fitted AUC: `0.877`
- Formula: `truth_logit = +0.872 * z(fragments_correlation) +0.933 * z(-fragments_pc_missing_peaks) +0.781 * z(diagnose_isodec_css) +0.503 * z(diagnose_top_peaks) +0.520`

## Suggested Thresholds

- `diagnose_final_cosine`: suggest `>= 0.8072` -> precision `0.725`, recall `0.920`, F1 `0.811`
- `fragments_fit_score`: suggest `>= 0.6056` -> precision `0.773`, recall `0.850`, F1 `0.810`
- `diagnose_raw_cosine`: suggest `>= 0.8063` -> precision `0.723`, recall `0.876`, F1 `0.792`
- `diagnose_isodec_css`: suggest `>= 0.8072` -> precision `0.735`, recall `0.859`, F1 `0.792`
- `fragments_coverage`: suggest `>= 0.8314` -> precision `0.691`, recall `0.894`, F1 `0.779`
- `fragments_pc_missing_peaks`: suggest `<= 52.9412` -> precision `0.654`, recall `0.956`, F1 `0.777`

## Files

- Dataset: `reports/q10r_truth_scoring/q10r_bc_truth_scoring_dataset.csv`
- Univariate stats: `reports/q10r_truth_scoring/q10r_bc_univariate_feature_stats.csv`
- Rule stats: `reports/q10r_truth_scoring/q10r_bc_rule_pass_stats.csv`
- Composite models: `reports/q10r_truth_scoring/q10r_bc_composite_model_stats.csv`
- Suggested thresholds: `reports/q10r_truth_scoring/q10r_bc_suggested_thresholds.csv`
