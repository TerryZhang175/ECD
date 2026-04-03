---
name: ecd-truth-comparison
description: Compare manual b/c-ion truth annotations against the UI-equivalent fragments algorithm over sample/ datasets. Use when you need to auto-discover raw spectra and manual truth CSVs, pair scans, run the same fragments pipeline used by the UI, and score TP/FP/FN using strict prefix plus chemical-formula matching.
---

# ECD Truth Comparison

Use this skill when the task is to compare manual truth and algorithm output for `b`/`c` ions across `sample/`.

## What this skill does

- Discovers dataset roots under `sample/`
- Distinguishes raw spectra from manual truth CSVs
- Pairs raw and truth files by scan number and nearest directory
- Runs the same fragments implementation used by the UI
- Compares ions with a strict identity rule:
  `same scan` + `same charge` + `same prefix` + `same formula`

## Identity rule

- Prefix keeps neutral-loss text, for example `b4-H2O` or `c31-NH3`
- Prefix removes disulfide variant suffixes, for example:
  `c31-BrokenLoopReduced` -> `c31`
  `b4-OxidizedLoop-H2O` -> `b4-H2O`
- Formula comparison uses normalized elemental composition strings
- Manual rows with missing `Molecular Composition` are not counted as strict TP/FN; they are written to a separate report

## UI-equivalent algorithm path

Run:

```bash
python3 scripts/compare_bc_truth_pipeline.py
```

The script calls the same `/api/run/fragments` implementation path used by the UI and restricts output to `ion_types=["b", "c"]`.

## Useful commands

Full run:

```bash
python3 scripts/compare_bc_truth_pipeline.py
```

Single dataset:

```bash
python3 scripts/compare_bc_truth_pipeline.py --dataset Q10R2
```

Single dataset with manual-annotation disulfide rules:

```bash
python3 scripts/compare_bc_truth_pipeline.py --dataset Q10R2 --disulfide-variant-mode manual_annotation
```

Single scan:

```bash
python3 scripts/compare_bc_truth_pipeline.py --dataset Q10R2 --scan 0
```

Custom output directory:

```bash
python3 scripts/compare_bc_truth_pipeline.py --output-dir reports/my_truth_compare
```

## Outputs

The script writes into `reports/bc_truth_comparison/` by default:

- `summary.json`
- `scan_pairs.csv`
- `per_scan_summary.csv`
- `per_dataset_summary.csv`
- `algorithm_predictions.csv`
- `exact_matches.csv`
- `false_negatives.csv`
- `false_positives.csv`
- `manual_missing_formula.csv`
- `unpaired_raw.csv`
- `unpaired_truth.csv`

## Interpretation

- `false_negatives.csv`
  If `conflict_type=wrong_formula`, the algorithm found the same prefix/charge but chose a different formula variant.
- `false_positives.csv`
  If `conflict_type=wrong_formula`, the algorithm prediction collides with a manual-positive prefix/charge but not the same formula.
- `manual_missing_formula.csv`
  These rows need manual completion if you want strict formula-based recall.

## Notes

- This skill is for comparison and reporting, not model training.
- `--disulfide-variant-mode manual_annotation` disables the extra disulfide variant hypotheses and keeps only the formula convention used by the current manual truth CSVs.
- If you need to inspect why a specific ion failed, run the truth comparison first, then use diagnose-mode tools on the exact FN row.
