from __future__ import annotations

import csv
import importlib.util
import io
import itertools
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

import personalized_config as cfg
from personalized_modes import run_diagnose_headless


REPORT_DIR = ROOT / "reports" / "q10r_truth_scoring"
DATASET_CSV = REPORT_DIR / "q10r_bc_truth_scoring_dataset.csv"
UNIVARIATE_CSV = REPORT_DIR / "q10r_bc_univariate_feature_stats.csv"
RULE_CSV = REPORT_DIR / "q10r_bc_rule_pass_stats.csv"
MODEL_CSV = REPORT_DIR / "q10r_bc_composite_model_stats.csv"
THRESHOLD_CSV = REPORT_DIR / "q10r_bc_suggested_thresholds.csv"
SUMMARY_MD = REPORT_DIR / "summary.md"

ANNOTATION_DIR = ROOT / "sample" / "Q10R" / "Q10R_annotated"

FEATURE_SPECS = [
    ("fragments_score", "higher"),
    ("fragments_css", "higher"),
    ("fragments_coverage", "higher"),
    ("fragments_match_count", "higher"),
    ("fragments_unexplained", "lower"),
    ("fragments_missing_core", "lower"),
    ("fragments_interference", "lower"),
    ("fragments_s2n", "higher"),
    ("fragments_ppm_rmse", "lower"),
    ("fragments_rawcos", "higher"),
    ("fragments_fit_score", "higher"),
    ("fragments_correlation", "higher"),
    ("fragments_pc_missing_peaks", "lower"),
    ("fragments_mass_error_std", "lower"),
    ("fragments_obs_rel_int", "higher"),
    ("fragments_abs_anchor_ppm", "lower"),
    ("diagnose_final_cosine", "higher"),
    ("diagnose_isodec_css", "higher"),
    ("diagnose_raw_cosine", "higher"),
    ("diagnose_abs_anchor_ppm", "lower"),
    ("diagnose_matched_peaks", "higher"),
    ("diagnose_area_covered", "higher"),
    ("diagnose_top_peaks", "higher"),
    ("diagnose_isodec_accepted", "higher"),
    ("diagnose_ok", "higher"),
]

COMPOSITE_POOL = [
    "fragments_fit_score",
    "fragments_score",
    "fragments_correlation",
    "fragments_rawcos",
    "fragments_pc_missing_peaks",
    "fragments_css",
    "fragments_coverage",
    "fragments_unexplained",
    "diagnose_raw_cosine",
    "diagnose_isodec_css",
    "diagnose_final_cosine",
    "diagnose_area_covered",
    "diagnose_top_peaks",
]

DIRECTION_MAP = {feature: direction for feature, direction in FEATURE_SPECS}


def load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MISSED_MOD = load_script_module(ROOT / "scripts" / "generate_q10r_missed_static_report.py", "q10r_missed_static_report")


def quiet_call(func: Callable[..., Any], *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(*args, **kwargs)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def parse_int(value: Any) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def parse_numeric_text(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a", "empty", "invalid", "yes", "no"}:
        return None
    text = text.replace("ppm", "").replace("%", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def best_numeric(rows: list[dict[str, Any]], column: str, mode: str = "max") -> float | None:
    values = [parse_float(row.get(column)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if mode == "max":
        return max(values)
    if mode == "min":
        return min(values)
    if mode == "min_abs":
        return min(abs(value) for value in values)
    raise ValueError(f"Unsupported mode: {mode}")


def summarize_manual_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched_rows = [row for row in rows if parse_float(row.get("Matched")) == 1.0]
    selected_rows = [row for row in rows if parse_float(row.get("Selected")) == 1.0]
    ref_rows = matched_rows if matched_rows else rows
    labels = sorted({str(row.get("Name") or "").strip() for row in rows if str(row.get("Name") or "").strip()})
    return {
        "manual_row_count": len(rows),
        "manual_matched_rows": len(matched_rows),
        "manual_selected_rows": len(selected_rows),
        "truth": int(bool(matched_rows)),
        "manual_labels_json": json.dumps(labels, ensure_ascii=False),
        "manual_best_ion_score": best_numeric(ref_rows, "IonScore", "max"),
        "manual_best_gof_confidence": best_numeric(ref_rows, "Gof Confidence", "max"),
        "manual_best_peaks_matched": best_numeric(ref_rows, "Peaks Matched", "max"),
        "manual_best_ion_corr_score": best_numeric(ref_rows, "Ion Correlation Score", "max"),
        "manual_min_abs_ppm_error": best_numeric(ref_rows, "Avg PPM Error", "min_abs"),
    }


def load_manual_aggregates() -> tuple[dict[tuple[int, str, int, int], dict[str, Any]], list[int]]:
    scan_pat = re.compile(r"ECDRE(\d+)-ion-state_man\.csv$")
    aggregates: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    re_nums: list[int] = []

    for ann_path in sorted(ANNOTATION_DIR.glob("ECDRE*-ion-state_man.csv"), key=lambda p: int(scan_pat.search(p.name).group(1))):
        re_num = int(scan_pat.search(ann_path.name).group(1))
        re_nums.append(re_num)
        by_key: dict[tuple[int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
        with ann_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ion_type = str(row.get("Base Type") or row.get("Type") or "").strip().lower()
                if ion_type not in {"b", "c"}:
                    continue
                pos = parse_int(row.get("Pos"))
                charge = parse_int(row.get("Charge"))
                if pos is None or charge is None:
                    continue
                by_key[(re_num, ion_type, pos, charge)].append(row)
        for key, rows in by_key.items():
            aggregates[key] = summarize_manual_rows(rows)
    return aggregates, re_nums


def prediction_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    def safe(value: Any) -> float:
        parsed = parse_float(value)
        return parsed if parsed is not None and np.isfinite(parsed) else float("-inf")

    return (
        safe(item.get("score")),
        safe(item.get("coverage")),
        safe(item.get("css")),
        safe(item.get("obs_int")),
    )


def make_base_maps(re_nums: list[int]) -> tuple[dict[int, dict[str, Any]], dict[tuple[int, str, int, int], dict[str, Any]], dict[tuple[int, str, int, int], dict[str, Any]]]:
    bundles: dict[int, dict[str, Any]] = {}
    selected_map: dict[tuple[int, str, int, int], dict[str, Any]] = {}
    raw_map: dict[tuple[int, str, int, int], dict[str, Any]] = {}

    for re_num in re_nums:
        bundle = MISSED_MOD.get_fragments_scan_bundle(re_num)
        bundles[re_num] = bundle
        fragments_result = bundle["fragmentsResult"]
        for group_name, target_map in (("best", selected_map), ("matches", raw_map)):
            for item in fragments_result.get(group_name, []) or []:
                ion_type = str(item.get("ion_type") or "").lower()
                if ion_type not in {"b", "c"}:
                    continue
                frag_len = parse_int(item.get("frag_len"))
                charge = parse_int(item.get("charge"))
                if frag_len is None or charge is None:
                    continue
                key = (re_num, ion_type, frag_len, charge)
                current = target_map.get(key)
                if current is None or prediction_sort_key(item) > prediction_sort_key(current):
                    target_map[key] = item
    return bundles, selected_map, raw_map


def rule_check_map(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    checks = trace.get("ruleChecks") or []
    out: dict[str, dict[str, Any]] = {}
    for check in checks:
        label = str(check.get("label") or "").strip()
        if label:
            out[label] = check
    return out


def extract_from_rule(trace: dict[str, Any], label: str) -> float | None:
    check = rule_check_map(trace).get(label)
    if not check:
        return None
    return parse_numeric_text(check.get("value_text") or check.get("valueText"))


def build_best_trace(bundles: dict[int, dict[str, Any]], key: tuple[int, str, int, int]) -> dict[str, Any]:
    re_num, ion_type, frag_len, charge = key
    bundle = bundles[re_num]
    residues = bundle["residues"]
    spectrum = bundle["spectrum"]
    isodec_config = bundle["isodecConfig"]
    noise_model = bundle["noiseModel"]

    with MISSED_MOD._override_cfg(bundle["overrides"]):
        frag_name, target_comp = MISSED_MOD.ion_composition_from_sequence(residues, ion_type, frag_len, amidated=cfg.AMIDATED)
        variant_rows = MISSED_MOD.get_disulfide_logic(ion_type, frag_len, len(residues)) or [("", None)]
        traces: list[dict[str, Any]] = []
        for variant_suffix, shift in variant_rows:
            try:
                variant_comp = target_comp + shift if shift is not None else target_comp
            except Exception:
                continue
            trace = MISSED_MOD.execute_hybrid_strategy(
                MISSED_MOD._evaluate_fragment_variant_trace,
                residues,
                spectrum,
                isodec_config,
                noise_model,
                ion_type=ion_type,
                frag_len=int(frag_len),
                z=int(charge),
                frag_name=frag_name,
                variant_suffix=str(variant_suffix or ""),
                variant_comp=variant_comp,
            )
            if isinstance(trace, dict):
                traces.append(trace)

    if traces:
        return max(traces, key=MISSED_MOD._trace_sort_key)
    return {
        "stage": "no_variant_trace",
        "accepted": False,
        "reason": "no_variant_trace",
        "ruleChecks": [],
        "unmetChecks": ["Variant trace"],
    }


def extract_fragments_metrics(
    bundles: dict[int, dict[str, Any]],
    selected_map: dict[tuple[int, str, int, int], dict[str, Any]],
    raw_map: dict[tuple[int, str, int, int], dict[str, Any]],
    key: tuple[int, str, int, int],
) -> dict[str, Any]:
    selected = selected_map.get(key)
    if selected is not None:
        return {
            "fragments_status": "selected_final",
            "fragments_stage": "accepted",
            "fragments_label": str(selected.get("label") or selected.get("frag_id") or ""),
            "fragments_variant_suffix": str(selected.get("variant_suffix") or ""),
            "fragments_score": parse_float(selected.get("score")),
            "fragments_css": parse_float(selected.get("css")),
            "fragments_rawcos": parse_float(selected.get("raw_score")),
            "fragments_coverage": parse_float(selected.get("coverage")),
            "fragments_match_count": parse_float(selected.get("match_count")),
            "fragments_unexplained": parse_float(selected.get("unexplained_fraction")),
            "fragments_missing_core": parse_float(selected.get("missing_core_fraction")),
            "fragments_interference": parse_float(selected.get("interference")),
            "fragments_s2n": parse_float(selected.get("s2n")),
            "fragments_ppm_rmse": parse_float(selected.get("ppm_rmse")),
            "fragments_fit_score": parse_float(selected.get("fit_score")),
            "fragments_correlation": parse_float(selected.get("correlation_coefficient")),
            "fragments_pc_missing_peaks": parse_float(selected.get("pc_missing_peaks")),
            "fragments_mass_error_std": parse_float(selected.get("mass_error_std")),
            "fragments_obs_rel_int": parse_float(selected.get("obs_rel_int")),
            "fragments_abs_anchor_ppm": abs(parse_float(selected.get("ppm")) or 0.0) if parse_float(selected.get("ppm")) is not None else None,
            "fragments_raw_candidate": 1,
            "fragments_selected_final": 1,
        }

    raw = raw_map.get(key)
    if raw is not None:
        return {
            "fragments_status": "raw_only",
            "fragments_stage": "accepted_raw",
            "fragments_label": str(raw.get("label") or raw.get("frag_id") or ""),
            "fragments_variant_suffix": str(raw.get("variant_suffix") or ""),
            "fragments_score": parse_float(raw.get("score")),
            "fragments_css": parse_float(raw.get("css")),
            "fragments_rawcos": parse_float(raw.get("raw_score")),
            "fragments_coverage": parse_float(raw.get("coverage")),
            "fragments_match_count": parse_float(raw.get("match_count")),
            "fragments_unexplained": parse_float(raw.get("unexplained_fraction")),
            "fragments_missing_core": parse_float(raw.get("missing_core_fraction")),
            "fragments_interference": parse_float(raw.get("interference")),
            "fragments_s2n": parse_float(raw.get("s2n")),
            "fragments_ppm_rmse": parse_float(raw.get("ppm_rmse")),
            "fragments_fit_score": parse_float(raw.get("fit_score")),
            "fragments_correlation": parse_float(raw.get("correlation_coefficient")),
            "fragments_pc_missing_peaks": parse_float(raw.get("pc_missing_peaks")),
            "fragments_mass_error_std": parse_float(raw.get("mass_error_std")),
            "fragments_obs_rel_int": parse_float(raw.get("obs_rel_int")),
            "fragments_abs_anchor_ppm": abs(parse_float(raw.get("ppm")) or 0.0) if parse_float(raw.get("ppm")) is not None else None,
            "fragments_raw_candidate": 1,
            "fragments_selected_final": 0,
        }

    trace = build_best_trace(bundles, key)
    ppm = parse_float(trace.get("anchorPpm"))
    return {
        "fragments_status": "trace_only",
        "fragments_stage": str(trace.get("stage") or "trace_only"),
        "fragments_label": str(trace.get("candidateLabel") or ""),
        "fragments_variant_suffix": str(trace.get("variantSuffix") or ""),
        "fragments_score": parse_float(trace.get("score")),
        "fragments_css": parse_float(trace.get("css")) or extract_from_rule(trace, "Fragments CSS"),
        "fragments_rawcos": parse_float(trace.get("rawCosine")) or extract_from_rule(trace, "Raw cosine"),
        "fragments_coverage": parse_float(trace.get("coverage")) or extract_from_rule(trace, "Coverage"),
        "fragments_match_count": parse_float(trace.get("matchCount")) or extract_from_rule(trace, "Local matches"),
        "fragments_unexplained": extract_from_rule(trace, "Unexplained"),
        "fragments_missing_core": extract_from_rule(trace, "Missing core"),
        "fragments_interference": extract_from_rule(trace, "Interference"),
        "fragments_s2n": extract_from_rule(trace, "S/N"),
        "fragments_ppm_rmse": extract_from_rule(trace, "PPM RMSE"),
        "fragments_fit_score": extract_from_rule(trace, "Fit score"),
        "fragments_correlation": extract_from_rule(trace, "Correlation"),
        "fragments_pc_missing_peaks": extract_from_rule(trace, "Missing peaks"),
        "fragments_mass_error_std": extract_from_rule(trace, "Mass error std"),
        "fragments_obs_rel_int": parse_float(trace.get("obsRelInt")),
        "fragments_abs_anchor_ppm": abs(ppm) if ppm is not None else extract_from_rule(trace, "Anchor ppm"),
        "fragments_raw_candidate": int(bool(trace.get("accepted"))),
        "fragments_selected_final": 0,
    }


DIAGNOSE_CACHE: dict[tuple[int, str, int, int], dict[str, Any]] = {}


def extract_diagnose_metrics(bundles: dict[int, dict[str, Any]], key: tuple[int, str, int, int]) -> dict[str, Any]:
    cached = DIAGNOSE_CACHE.get(key)
    if cached is not None:
        return cached

    re_num, ion_type, pos, charge = key
    bundle = bundles[re_num]
    ion_spec = f"{ion_type}{pos}^{charge}+"
    with MISSED_MOD._override_cfg(bundle["overrides"]):
        result = quiet_call(
            run_diagnose_headless,
            bundle["residues"],
            bundle["spectrum"],
            bundle["isodecConfig"],
            ion_spec=ion_spec,
            h_transfer=0,
        )

    best = result.get("best") or {}
    detail = best.get("isodec_detail") if isinstance(best.get("isodec_detail"), dict) else {}
    anchor_ppm = parse_float(best.get("anchor_ppm"))
    metrics = {
        "diagnose_best_label": str(best.get("label") or ""),
        "diagnose_reason": str(best.get("reason") or ""),
        "diagnose_ok": int(bool(best.get("ok"))) if best else 0,
        "diagnose_isodec_accepted": int(bool(best.get("isodec_accepted"))) if best else 0,
        "diagnose_final_cosine": parse_float(best.get("final_cosine")),
        "diagnose_isodec_css": parse_float(best.get("isodec_css")),
        "diagnose_raw_cosine": parse_float(best.get("raw_cosine")),
        "diagnose_abs_anchor_ppm": abs(anchor_ppm) if anchor_ppm is not None else None,
        "diagnose_matched_peaks": parse_float(detail.get("matched_peaks_n")),
        "diagnose_area_covered": parse_float(detail.get("areacovered")),
        "diagnose_top_peaks": (1.0 if detail.get("topthree") is True else 0.0 if detail.get("topthree") is False else None),
        "diagnose_anchor_within_ppm": int(bool(best.get("anchor_within_ppm"))) if best.get("anchor_within_ppm") is not None else None,
    }
    DIAGNOSE_CACHE[key] = metrics
    return metrics


def build_dataset() -> pd.DataFrame:
    manual_aggregates, re_nums = load_manual_aggregates()
    bundles, selected_map, raw_map = make_base_maps(re_nums)

    truth_true = {key for key, agg in manual_aggregates.items() if int(agg["truth"]) == 1}
    truth_false = {key for key, agg in manual_aggregates.items() if int(agg["truth"]) == 0}
    selected_negatives = set(selected_map) - truth_true
    candidate_keys = sorted(truth_true | truth_false | selected_negatives)

    rows: list[dict[str, Any]] = []
    for key in candidate_keys:
        re_num, ion_type, pos, charge = key
        manual = manual_aggregates.get(key, {
            "manual_row_count": 0,
            "manual_matched_rows": 0,
            "manual_selected_rows": 0,
            "truth": 0,
            "manual_labels_json": "[]",
            "manual_best_ion_score": None,
            "manual_best_gof_confidence": None,
            "manual_best_peaks_matched": None,
            "manual_best_ion_corr_score": None,
            "manual_min_abs_ppm_error": None,
        })
        truth = int(manual["truth"])
        selected = key in selected_map
        annotated = manual.get("manual_row_count", 0) > 0
        if truth and selected:
            candidate_group = "tp_selected"
        elif truth:
            candidate_group = "fn_missed"
        elif selected:
            candidate_group = "fp_selected"
        else:
            candidate_group = "manual_negative_only"

        row = {
            "re": f"RE{re_num}",
            "re_index": re_num,
            "ion_type": ion_type,
            "pos": pos,
            "charge": charge,
            "base_ion_spec": f"{ion_type}{pos}^{charge}+",
            "truth": truth,
            "candidate_group": candidate_group,
            "in_manual_annotation": int(annotated),
            "in_fragments_selected": int(selected),
            **manual,
            **extract_fragments_metrics(bundles, selected_map, raw_map, key),
            **extract_diagnose_metrics(bundles, key),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["re_index", "ion_type", "pos", "charge"]).reset_index(drop=True)


def auc_from_scores(y: np.ndarray, x: np.ndarray) -> float | None:
    mask = np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    ranks = stats.rankdata(x)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_univariate_stats(df: pd.DataFrame) -> pd.DataFrame:
    y = df["truth"].to_numpy(dtype=float)
    rows = []
    for feature, expected_direction in FEATURE_SPECS:
        x = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x)
        x_valid = x[mask]
        y_valid = y[mask]
        if x_valid.size == 0 or len(np.unique(y_valid)) < 2:
            continue
        pos = x_valid[y_valid == 1]
        neg = x_valid[y_valid == 0]
        if pos.size == 0 or neg.size == 0:
            continue
        try:
            rho, pval = stats.spearmanr(x_valid, y_valid)
        except Exception:
            rho, pval = np.nan, np.nan
        auc_high = auc_from_scores(y_valid, x_valid)
        auc_best = None if auc_high is None else max(float(auc_high), float(1.0 - auc_high))
        best_direction = "higher" if auc_high is None or auc_high >= 0.5 else "lower"
        rows.append(
            {
                "feature": feature,
                "group": feature.split("_", 1)[0],
                "expected_direction": expected_direction,
                "best_direction": best_direction,
                "n": int(mask.sum()),
                "pos_mean": float(np.mean(pos)),
                "neg_mean": float(np.mean(neg)),
                "pos_median": float(np.median(pos)),
                "neg_median": float(np.median(neg)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p": float(pval) if np.isfinite(pval) else np.nan,
                "auc_higher_is_more_true": float(auc_high) if auc_high is not None else np.nan,
                "auc_best_direction": float(auc_best) if auc_best is not None else np.nan,
                "delta_mean": float(np.mean(pos) - np.mean(neg)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["auc_best_direction", "spearman_rho"], ascending=[False, False]).reset_index(drop=True)


def compute_rule_stats(df: pd.DataFrame) -> pd.DataFrame:
    ppm_rmse_thresh = float(cfg.FRAG_MAX_RESIDUAL_RMSE_PPM) if cfg.FRAG_MAX_RESIDUAL_RMSE_PPM is not None else float(cfg.MATCH_TOL_PPM)
    rules: list[tuple[str, Callable[[pd.Series], pd.Series], str]] = [
        ("diagnose_ok", lambda s: pd.to_numeric(s, errors="coerce") >= 0.5, f"ok == 1 (final_cosine >= {float(cfg.MIN_COSINE):.2f})"),
        ("diagnose_isodec_css", lambda s: pd.to_numeric(s, errors="coerce") >= float(cfg.ISODEC_CSS_THRESH), f">= {float(cfg.ISODEC_CSS_THRESH):.2f}"),
        ("diagnose_matched_peaks", lambda s: pd.to_numeric(s, errors="coerce") >= int(cfg.ISODEC_MINPEAKS), f">= {int(cfg.ISODEC_MINPEAKS)}"),
        ("diagnose_area_covered", lambda s: pd.to_numeric(s, errors="coerce") >= float(cfg.ISODEC_MIN_AREA_COVERED), f">= {float(cfg.ISODEC_MIN_AREA_COVERED):.2f}"),
        ("diagnose_top_peaks", lambda s: pd.to_numeric(s, errors="coerce") >= 0.5, "top peaks = yes"),
        ("diagnose_abs_anchor_ppm", lambda s: pd.to_numeric(s, errors="coerce") <= float(cfg.MATCH_TOL_PPM), f"<= {float(cfg.MATCH_TOL_PPM):.1f} ppm"),
        ("fragments_css", lambda s: pd.to_numeric(s, errors="coerce") >= float(cfg.MIN_COSINE), f">= {float(cfg.MIN_COSINE):.2f}"),
        ("fragments_match_count", lambda s: pd.to_numeric(s, errors="coerce") >= int(cfg.FRAG_MIN_MATCHED_PEAKS), f">= {int(cfg.FRAG_MIN_MATCHED_PEAKS)}"),
        ("fragments_coverage", lambda s: pd.to_numeric(s, errors="coerce") >= float(cfg.FRAG_MIN_COVERAGE), f">= {float(cfg.FRAG_MIN_COVERAGE):.2f}"),
        ("fragments_unexplained", lambda s: pd.to_numeric(s, errors="coerce") <= float(cfg.FRAG_MAX_UNEXPLAINED_FRAC), f"<= {float(cfg.FRAG_MAX_UNEXPLAINED_FRAC):.2f}"),
        ("fragments_missing_core", lambda s: pd.to_numeric(s, errors="coerce") <= float(cfg.FRAG_MAX_MISSING_CORE_FRAC), f"<= {float(cfg.FRAG_MAX_MISSING_CORE_FRAC):.2f}"),
        ("fragments_s2n", lambda s: pd.to_numeric(s, errors="coerce") >= float(cfg.FRAG_MIN_S2N), f">= {float(cfg.FRAG_MIN_S2N):.2f}"),
        ("fragments_interference", lambda s: pd.to_numeric(s, errors="coerce") <= float(cfg.FRAG_MAX_INTERFERENCE), f"<= {float(cfg.FRAG_MAX_INTERFERENCE):.2f}"),
        ("fragments_ppm_rmse", lambda s: pd.to_numeric(s, errors="coerce") <= ppm_rmse_thresh, f"<= {ppm_rmse_thresh:.1f} ppm"),
        ("fragments_fit_score", lambda s: pd.to_numeric(s, errors="coerce") >= float(cfg.FRAG_MIN_FIT_SCORE), f">= {float(cfg.FRAG_MIN_FIT_SCORE):.2f}"),
        ("fragments_pc_missing_peaks", lambda s: pd.to_numeric(s, errors="coerce") <= float(cfg.FRAG_MAX_PC_MISSING_PEAKS), f"<= {float(cfg.FRAG_MAX_PC_MISSING_PEAKS):.1f}%"),
        ("fragments_selected_final", lambda s: pd.to_numeric(s, errors="coerce") >= 0.5, "selected in final fragments"),
        ("fragments_raw_candidate", lambda s: pd.to_numeric(s, errors="coerce") >= 0.5, "reached raw fragments candidate"),
        ("diagnose_isodec_accepted", lambda s: pd.to_numeric(s, errors="coerce") >= 0.5, "diagnose isodec accepted"),
    ]

    y = df["truth"].to_numpy(dtype=float)
    rows = []
    base_rate = float(np.mean(y)) if y.size else np.nan
    for feature, predicate, rule_text in rules:
        series = df[feature]
        valid = pd.to_numeric(series, errors="coerce") if series.dtype != object else series
        if isinstance(valid, pd.Series):
            mask = valid.notna().to_numpy()
        else:
            mask = np.isfinite(valid.to_numpy(dtype=float))
        if mask.sum() == 0:
            continue
        passed = predicate(series).fillna(False).to_numpy(dtype=bool)
        y_valid = y[mask]
        passed_valid = passed[mask]
        pass_n = int(np.sum(passed_valid))
        if pass_n == 0:
            precision = np.nan
        else:
            precision = float(np.mean(y_valid[passed_valid]))
        recall = float(np.mean(passed_valid[y_valid == 1])) if np.any(y_valid == 1) else np.nan
        false_pass_rate = float(np.mean(passed_valid[y_valid == 0])) if np.any(y_valid == 0) else np.nan
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = np.nan
        lift = precision / base_rate if np.isfinite(precision) and base_rate > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "rule": rule_text,
                "n_valid": int(mask.sum()),
                "pass_n": pass_n,
                "pass_rate": float(np.mean(passed_valid)),
                "precision": precision,
                "recall": recall,
                "false_pass_rate": false_pass_rate,
                "f1": f1,
                "lift_vs_base_rate": lift,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).reset_index(drop=True)


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, reg: float = 1.0) -> tuple[np.ndarray, float]:
    n_features = X.shape[1]

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        w = params[:n_features]
        b = float(params[n_features])
        z = np.clip(X @ w + b, -40.0, 40.0)
        p = special.expit(z)
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * reg * np.sum(w * w) / max(len(y), 1)
        diff = p - y
        grad_w = (X.T @ diff) / max(len(y), 1) + reg * w / max(len(y), 1)
        grad_b = float(np.mean(diff))
        grad = np.concatenate([grad_w, np.array([grad_b])])
        return float(loss), grad

    init = np.zeros(n_features + 1, dtype=float)
    result = optimize.minimize(
        fun=lambda params: objective(params)[0],
        x0=init,
        jac=lambda params: objective(params)[1],
        method="L-BFGS-B",
        bounds=[(0.0, 5.0)] * n_features + [(-5.0, 5.0)],
    )
    params = result.x if result.success else init
    return params[:n_features], float(params[n_features])


def build_design_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train = train_df[features].apply(pd.to_numeric, errors="coerce")
    test = test_df[features].apply(pd.to_numeric, errors="coerce")
    for feature in features:
        if DIRECTION_MAP.get(feature) == "lower":
            train[feature] = -train[feature]
            test[feature] = -test[feature]
    medians = train.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train = train.fillna(medians)
    test = test.fillna(medians)
    means = train.mean(axis=0)
    stds = train.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    X_train = ((train - means) / stds).to_numpy(dtype=float)
    X_test = ((test - means) / stds).to_numpy(dtype=float)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train, X_test


def cross_validated_auc(df: pd.DataFrame, features: list[str]) -> float | None:
    fold_preds = []
    fold_truth = []
    for re_value in sorted(df["re"].unique(), key=lambda value: int(str(value).replace("RE", ""))):
        train_df = df[df["re"] != re_value]
        test_df = df[df["re"] == re_value]
        if train_df.empty or test_df.empty:
            continue
        y_train = train_df["truth"].to_numpy(dtype=float)
        if len(np.unique(y_train)) < 2:
            continue
        X_train, X_test = build_design_matrices(train_df, test_df, features)
        w, b = fit_logistic_regression(X_train, y_train, reg=1.0)
        preds = special.expit(X_test @ w + b)
        fold_preds.append(preds)
        fold_truth.append(test_df["truth"].to_numpy(dtype=float))
    if not fold_preds:
        return None
    y_all = np.concatenate(fold_truth)
    p_all = np.concatenate(fold_preds)
    return auc_from_scores(y_all, p_all)


def compute_composite_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for size in range(1, 5):
        for combo in itertools.combinations(COMPOSITE_POOL, size):
            auc = cross_validated_auc(df, list(combo))
            if auc is None:
                continue
            rows.append(
                {
                    "features": ", ".join(combo),
                    "feature_count": size,
                    "cv_auc": float(auc),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["cv_auc", "feature_count", "features"], ascending=[False, True, True]).reset_index(drop=True)


def compute_suggested_thresholds(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    univariate = compute_univariate_stats(df)
    if univariate.empty:
        return pd.DataFrame()
    chosen = univariate.head(top_n)
    y = df["truth"].to_numpy(dtype=float)
    rows = []
    for _, meta in chosen.iterrows():
        feature = str(meta["feature"])
        higher = str(meta["best_direction"]) == "higher"
        x = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x)
        x_valid = x[mask]
        y_valid = y[mask]
        if x_valid.size == 0 or len(np.unique(y_valid)) < 2:
            continue
        best = None
        for threshold in np.unique(x_valid):
            passed = x_valid >= threshold if higher else x_valid <= threshold
            if not np.any(passed):
                continue
            precision = float(np.mean(y_valid[passed]))
            recall = float(np.mean(passed[y_valid == 1])) if np.any(y_valid == 1) else np.nan
            specificity = float(np.mean((~passed)[y_valid == 0])) if np.any(y_valid == 0) else np.nan
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
                f1 = 2.0 * precision * recall / (precision + recall)
            else:
                f1 = np.nan
            balanced_accuracy = (recall + specificity) / 2.0 if np.isfinite(recall) and np.isfinite(specificity) else np.nan
            candidate = (
                -np.inf if not np.isfinite(f1) else f1,
                -np.inf if not np.isfinite(balanced_accuracy) else balanced_accuracy,
                -np.inf if not np.isfinite(precision) else precision,
                -np.inf if not np.isfinite(recall) else recall,
                float(threshold),
            )
            if best is None or candidate > best[0]:
                best = (
                    candidate,
                    {
                        "feature": feature,
                        "direction": ">=" if higher else "<=",
                        "threshold": float(threshold),
                        "precision": precision,
                        "recall": recall,
                        "specificity": specificity,
                        "f1": f1,
                        "balanced_accuracy": balanced_accuracy,
                    },
                )
        if best is not None:
            rows.append(best[1])
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["f1", "balanced_accuracy", "precision"], ascending=[False, False, False]).reset_index(drop=True)


def fit_full_model_formula(df: pd.DataFrame, features: list[str]) -> tuple[str, float | None]:
    subset = df[["truth", *features]].copy()
    X_df = subset[features].apply(pd.to_numeric, errors="coerce")
    for feature in features:
        if DIRECTION_MAP.get(feature) == "lower":
            X_df[feature] = -X_df[feature]
    medians = X_df.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_df = X_df.fillna(medians)
    means = X_df.mean(axis=0)
    stds = X_df.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    X = ((X_df - means) / stds).to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = subset["truth"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return "", None
    w, b = fit_logistic_regression(X, y, reg=1.0)
    parts = []
    for feature, coef in zip(features, w):
        label = f"-{feature}" if DIRECTION_MAP.get(feature) == "lower" else feature
        parts.append(f"{coef:+.3f} * z({label})")
    formula = "truth_logit = " + " ".join(parts) + f" {b:+.3f}"
    auc = auc_from_scores(y, special.expit(X @ w + b))
    return formula, auc


def write_summary(
    df: pd.DataFrame,
    univariate: pd.DataFrame,
    rules: pd.DataFrame,
    models: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    total = len(df)
    positives = int(df["truth"].sum())
    negatives = total - positives
    group_counts = Counter(df["candidate_group"])

    top_uni = univariate.head(10)
    weak_uni = univariate.sort_values(["auc_best_direction", "spearman_rho"], ascending=[True, True]).head(8)
    top_rules = rules.head(10)
    top_thresholds = thresholds.head(6) if not thresholds.empty else thresholds
    best_model_features = []
    best_model_auc = None
    best_model_formula = ""
    if not models.empty:
        best_model_features = [part.strip() for part in str(models.iloc[0]["features"]).split(",")]
        best_model_auc = float(models.iloc[0]["cv_auc"])
        best_model_formula, full_auc = fit_full_model_formula(df, best_model_features)
    else:
        full_auc = None

    lines = []
    lines.append("# Q10R b/c Truth Scoring Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Dataset: Q10R, only `b/c` ions, base key = `RE + ion_type + pos + charge`")
    lines.append("- Truth: manual `Matched=1`")
    lines.append("- Diagnose metrics: current headless diagnose logic with `h_transfer = 0`")
    lines.append("- Candidate universe: all manual-positive bases, all manual-negative annotated bases, plus all algorithm-selected bases that manual truth did not mark positive")
    lines.append("")
    lines.append("## Sample Counts")
    lines.append("")
    lines.append(f"- Total candidates: `{total}`")
    lines.append(f"- Truth positive: `{positives}`")
    lines.append(f"- Truth negative: `{negatives}`")
    lines.append(f"- `tp_selected`: `{group_counts.get('tp_selected', 0)}`")
    lines.append(f"- `fn_missed`: `{group_counts.get('fn_missed', 0)}`")
    lines.append(f"- `fp_selected`: `{group_counts.get('fp_selected', 0)}`")
    lines.append(f"- `manual_negative_only`: `{group_counts.get('manual_negative_only', 0)}`")
    lines.append("")
    lines.append("## Scoring Logic")
    lines.append("")
    lines.append("- `diagnose` final acceptance is driven by `final_cosine >= MIN_COSINE`, with `isodec_css`, anchor ppm, matched peaks, area covered, and top peaks exposed as diagnostic detail rather than a strict copy of fragments gating.")
    lines.append("- `fragments` uses a two-layer logic: first a composite/evidence score, then hard gates on CSS, anchor ppm, local matches, coverage, unexplained fraction, missing core, and other quality checks.")
    lines.append("- The current fragments evidence score is approximately: `css + coverage + ppm consistency + spacing consistency + intensity prior + fit/correlation/s2n bonuses - unexplained/missing-core/missing-peaks penalties`.")
    lines.append("")
    lines.append("## Strongest Single Features")
    lines.append("")
    for _, row in top_uni.iterrows():
        lines.append(
            f"- `{row['feature']}`: best-direction AUC `{row['auc_best_direction']:.3f}`, Spearman `{row['spearman_rho']:.3f}`, best direction `{row['best_direction']}`"
        )
    lines.append("")
    lines.append("## Weak Or Low-Value Features")
    lines.append("")
    for _, row in weak_uni.iterrows():
        lines.append(
            f"- `{row['feature']}`: best-direction AUC `{row['auc_best_direction']:.3f}`, Spearman `{row['spearman_rho']:.3f}`"
        )
    lines.append("")
    lines.append("## Current Thresholds")
    lines.append("")
    for _, row in top_rules.iterrows():
        precision = "n/a" if not np.isfinite(row["precision"]) else f"{row['precision']:.3f}"
        recall = "n/a" if not np.isfinite(row["recall"]) else f"{row['recall']:.3f}"
        f1 = "n/a" if not np.isfinite(row["f1"]) else f"{row['f1']:.3f}"
        lines.append(f"- `{row['feature']}` with `{row['rule']}`: precision `{precision}`, recall `{recall}`, F1 `{f1}`")
    lines.append("")
    lines.append("## Best Composite Score")
    lines.append("")
    if best_model_features:
        lines.append(f"- Best leave-one-RE-out model: `{', '.join(best_model_features)}`")
        lines.append(f"- Cross-validated AUC: `{best_model_auc:.3f}`")
        if full_auc is not None:
            lines.append(f"- Full-data fitted AUC: `{full_auc:.3f}`")
        lines.append(f"- Formula: `{best_model_formula}`")
    else:
        lines.append("- No composite model could be fit.")
    lines.append("")
    lines.append("## Suggested Thresholds")
    lines.append("")
    if thresholds.empty:
        lines.append("- No suggested thresholds could be computed.")
    else:
        for _, row in top_thresholds.iterrows():
            lines.append(
                f"- `{row['feature']}`: suggest `{row['direction']} {row['threshold']:.4f}` -> precision `{row['precision']:.3f}`, recall `{row['recall']:.3f}`, F1 `{row['f1']:.3f}`"
            )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Dataset: `{DATASET_CSV.relative_to(ROOT)}`")
    lines.append(f"- Univariate stats: `{UNIVARIATE_CSV.relative_to(ROOT)}`")
    lines.append(f"- Rule stats: `{RULE_CSV.relative_to(ROOT)}`")
    lines.append(f"- Composite models: `{MODEL_CSV.relative_to(ROOT)}`")
    lines.append(f"- Suggested thresholds: `{THRESHOLD_CSV.relative_to(ROOT)}`")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    univariate = compute_univariate_stats(df)
    rules = compute_rule_stats(df)
    models = compute_composite_models(df)
    thresholds = compute_suggested_thresholds(df)

    df.to_csv(DATASET_CSV, index=False)
    univariate.to_csv(UNIVARIATE_CSV, index=False)
    rules.to_csv(RULE_CSV, index=False)
    models.to_csv(MODEL_CSV, index=False)
    thresholds.to_csv(THRESHOLD_CSV, index=False)
    write_summary(df, univariate, rules, models, thresholds)

    print(f"Dataset: {DATASET_CSV}")
    print(f"Candidates: {len(df)}")
    print(f"Truth positives: {int(df['truth'].sum())}")
    print(f"Truth negatives: {len(df) - int(df['truth'].sum())}")
    if not models.empty:
        print(f"Best model: {models.iloc[0]['features']} | cv_auc={models.iloc[0]['cv_auc']:.3f}")


if __name__ == "__main__":
    main()
