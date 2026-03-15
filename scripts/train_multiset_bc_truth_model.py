from __future__ import annotations

import csv
import importlib.util
import io
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
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
from ecd_api import FragmentsRunRequest, _build_overrides, _override_cfg, parse_custom_sequence
from personalized import load_spectrum, preprocess_spectrum
from personalized_modes import (
    _build_noise_level_model,
    run_diagnose_headless,
    run_fragments_headless,
    run_precursor_headless,
)

REPORT_DIR = ROOT / "reports" / "multiset_bc_truth_model"
DATASET_CSV = REPORT_DIR / "multiset_bc_truth_dataset.csv"
MODELS_CSV = REPORT_DIR / "combo_models.csv"
UNIVARIATE_CSV = REPORT_DIR / "univariate_stats.csv"
SCAN_PRED_CSV = REPORT_DIR / "scan_cv_predictions.csv"
FAMILY_PRED_CSV = REPORT_DIR / "family_cv_predictions.csv"
COEFFICIENTS_CSV = REPORT_DIR / "model_coefficients.csv"
MODEL_JSON = REPORT_DIR / "model_artifact.json"
SUMMARY_MD = REPORT_DIR / "summary.md"

RE_PAT = re.compile(r"re\s*(\d+)", re.I)

FEATURE_SPECS = [
    ("fragments_score", "higher"),
    ("fragments_css", "higher"),
    ("fragments_coverage", "higher"),
    ("fragments_match_count", "higher"),
    ("fragments_rawcos", "higher"),
    ("fragments_fit_score", "higher"),
    ("fragments_correlation", "higher"),
    ("fragments_pc_missing_peaks", "lower"),
    ("fragments_abs_anchor_ppm", "lower"),
    ("fragments_obs_rel_int", "higher"),
    ("diagnose_final_cosine", "higher"),
    ("diagnose_isodec_css", "higher"),
    ("diagnose_raw_cosine", "higher"),
    ("diagnose_abs_anchor_ppm", "lower"),
    ("diagnose_matched_peaks", "higher"),
    ("diagnose_area_covered", "higher"),
    ("diagnose_top_peaks", "higher"),
]
DIRECTION_MAP = {feature: direction for feature, direction in FEATURE_SPECS}
FEATURE_POOL = [
    "fragments_css",
    "fragments_rawcos",
    "fragments_fit_score",
    "fragments_correlation",
    "fragments_pc_missing_peaks",
    "fragments_match_count",
    "fragments_abs_anchor_ppm",
    "diagnose_isodec_css",
    "diagnose_raw_cosine",
    "diagnose_matched_peaks",
    "diagnose_area_covered",
    "diagnose_top_peaks",
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    family: str
    annotation_dir: Path
    spectrum_dir: Path
    peptide: str
    copies: int
    amidated: bool
    disulfide_bonds: int
    disulfide_map: str


@dataclass(frozen=True)
class ScanKey:
    dataset: str
    family: str
    re_num: int
    ion_type: str
    pos: int
    charge: int


def load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACE_MOD = load_script_module(ROOT / "scripts" / "generate_q10r_truthscore_fn_static_report.py", "q10r_truthscore_fn_static_report")
BASE_MOD = TRACE_MOD.base


def quiet_call(func: Callable[..., Any], *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(*args, **kwargs)


@contextmanager
def truthscore_disabled():
    old = (
        cfg.FRAG_TRUTH_SCORE_ENABLE,
        cfg.FRAG_TRUTH_SCORE_THRESHOLD,
        cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING,
    )
    cfg.FRAG_TRUTH_SCORE_ENABLE = False
    cfg.FRAG_TRUTH_SCORE_THRESHOLD = float(old[1])
    cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING = False
    try:
        yield
    finally:
        (
            cfg.FRAG_TRUTH_SCORE_ENABLE,
            cfg.FRAG_TRUTH_SCORE_THRESHOLD,
            cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING,
        ) = old


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


def parse_sequence_meta(path: Path) -> dict[str, Any]:
    seq = None
    copies = 2
    amidated = True
    disulfide_bonds = 2
    disulfide_map = "2-7"
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if line.startswith("Sequence:"):
            seq = line.split(":", 1)[1].strip()
        elif lower.startswith("copies="):
            copies = int(line.split("=", 1)[1].strip())
        elif lower.startswith("admidated=") or lower.startswith("amidated="):
            amidated = line.split("=", 1)[1].strip().lower() == "true"
        elif lower.startswith("disulfide bonds="):
            disulfide_bonds = int(line.split("=", 1)[1].strip())
        elif lower.startswith("disulfide map="):
            disulfide_map = line.split("=", 1)[1].strip()
    if not seq:
        raise RuntimeError(f"No sequence found in {path}")
    return {
        "peptide": seq,
        "copies": copies,
        "amidated": amidated,
        "disulfide_bonds": disulfide_bonds,
        "disulfide_map": disulfide_map,
    }


def dataset_configs() -> list[DatasetConfig]:
    wt_meta = parse_sequence_meta(ROOT / "sample/WT/WT_sequence.txt")
    s20g_meta = parse_sequence_meta(ROOT / "sample/S20G/S20G_sequence.txt")
    return [
        DatasetConfig(
            name="Q10R",
            family="Q10R",
            annotation_dir=ROOT / "sample/Q10R/Q10R_annotated",
            spectrum_dir=ROOT / "sample/Q10R/Centroid (lock mass)_副本",
            peptide="KCNTATCATRRLANFLVHSSNNFGAILSSTNVGSNTY",
            copies=2,
            amidated=True,
            disulfide_bonds=2,
            disulfide_map="2-7",
        ),
        DatasetConfig(
            name="WT_2025-02",
            family="WT",
            annotation_dir=ROOT / "sample/WT/ZD_21to22Feb_2n5_WTECD",
            spectrum_dir=ROOT / "sample/WT/ZD_21to22Feb_2n5_WTECD",
            **wt_meta,
        ),
        DatasetConfig(
            name="WT_2025-03",
            family="WT",
            annotation_dir=ROOT / "sample/WT/ZD_25_03_12_WT2n5_ECD",
            spectrum_dir=ROOT / "sample/WT/ZD_25_03_12_WT2n5_ECD",
            **wt_meta,
        ),
        DatasetConfig(
            name="S20G",
            family="S20G",
            annotation_dir=ROOT / "sample/S20G/S20G_04_10-11ECD_Rep2/Manual",
            spectrum_dir=ROOT / "sample/S20G/S20G_04_10-11ECD_Rep2",
            **s20g_meta,
        ),
    ]


def discover_re_files(directory: Path, suffix: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        match = RE_PAT.search(path.name)
        if not match:
            continue
        out[int(match.group(1))] = path
    return out


def summarize_manual_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched_rows = [row for row in rows if parse_float(row.get("Matched")) == 1.0]
    selected_rows = [row for row in rows if parse_float(row.get("Selected")) == 1.0]
    ref_rows = matched_rows if matched_rows else rows
    labels = sorted({str(row.get("Name") or "").strip() for row in rows if str(row.get("Name") or "").strip()})

    def best_numeric(column: str, mode: str = "max") -> float | None:
        values = [parse_float(row.get(column)) for row in ref_rows]
        values = [value for value in values if value is not None]
        if not values:
            return None
        if mode == "max":
            return max(values)
        if mode == "min_abs":
            return min(abs(value) for value in values)
        raise ValueError(mode)

    return {
        "manual_row_count": len(rows),
        "manual_matched_rows": len(matched_rows),
        "manual_selected_rows": len(selected_rows),
        "truth": int(bool(matched_rows)),
        "manual_labels_json": json.dumps(labels, ensure_ascii=False),
        "manual_best_ion_score": best_numeric("IonScore", "max"),
        "manual_best_gof_confidence": best_numeric("Gof Confidence", "max"),
        "manual_best_peaks_matched": best_numeric("Peaks Matched", "max"),
        "manual_best_ion_corr_score": best_numeric("Ion Correlation Score", "max"),
        "manual_min_abs_ppm_error": best_numeric("Avg PPM Error", "min_abs"),
    }


def load_manual_aggregates(configs: list[DatasetConfig]) -> tuple[dict[ScanKey, dict[str, Any]], dict[str, list[int]]]:
    aggregates: dict[ScanKey, dict[str, Any]] = {}
    re_lists: dict[str, list[int]] = {}
    for config in configs:
        ann_map = discover_re_files(config.annotation_dir, ".csv")
        spec_map = discover_re_files(config.spectrum_dir, ".txt")
        re_nums = sorted(set(ann_map) & set(spec_map))
        re_lists[config.name] = re_nums
        for re_num in re_nums:
            by_key: dict[ScanKey, list[dict[str, Any]]] = defaultdict(list)
            with ann_map[re_num].open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    ion_type = str(row.get("Base Type") or row.get("Type") or "").strip().lower()
                    if ion_type not in {"b", "c"}:
                        continue
                    pos = parse_int(row.get("Pos"))
                    charge = parse_int(row.get("Charge"))
                    if pos is None or charge is None:
                        continue
                    key = ScanKey(config.name, config.family, re_num, ion_type, pos, charge)
                    by_key[key].append(row)
            for key, rows in by_key.items():
                aggregates[key] = summarize_manual_rows(rows)
    return aggregates, re_lists


SCAN_BUNDLE_CACHE: dict[tuple[str, int], dict[str, Any]] = {}


def get_scan_bundle(config: DatasetConfig, re_num: int) -> dict[str, Any]:
    cache_key = (config.name, int(re_num))
    cached = SCAN_BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spec_map = discover_re_files(config.spectrum_dir, ".txt")
    spectrum_path = spec_map[int(re_num)]
    req = FragmentsRunRequest(
        filepath=str(spectrum_path.resolve()),
        scan=1,
        peptide=config.peptide,
        ion_types=["b", "c"],
        frag_min_charge=1,
        frag_max_charge=5,
        copies=config.copies,
        amidated=config.amidated,
        disulfide_bonds=config.disulfide_bonds,
        disulfide_map=config.disulfide_map,
        precursor_calibration=False,
    )
    overrides = _build_overrides(req, str(spectrum_path.resolve()), plot_mode="fragments")

    with _override_cfg(overrides):
        cfg.require_isodec_rules()
        isodec_config = cfg.build_isodec_config()
        residues = parse_custom_sequence(cfg.PEPTIDE)
        spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=bool(cfg.ENABLE_CENTROID))
        spectrum = preprocess_spectrum(spectrum)
        precursor_result = None
        if bool(getattr(cfg, "PRECURSOR_CHAIN_TO_FRAGMENTS", False)):
            precursor_result = run_precursor_headless(
                residues,
                spectrum,
                isodec_config,
                apply_calibration=True,
            )
            spectrum = np.asarray(precursor_result.get("spectrum"), dtype=float)
        fragments_result = run_fragments_headless(residues, spectrum, isodec_config)
        noise_model = _build_noise_level_model(
            np.asarray(spectrum[:, 0], dtype=float),
            np.asarray(spectrum[:, 1], dtype=float),
            num_splits=max(4, int(getattr(cfg, "FRAG_NOISE_MODEL_SPLITS", 50))),
            hist_bins=max(16, int(getattr(cfg, "FRAG_NOISE_HIST_BINS", 128))),
        )

    bundle = {
        "config": config,
        "reNum": re_num,
        "overrides": overrides,
        "residues": residues,
        "spectrum": np.asarray(spectrum, dtype=float),
        "isodecConfig": isodec_config,
        "fragmentsResult": fragments_result,
        "noiseModel": noise_model,
        "precursorResult": precursor_result,
    }
    SCAN_BUNDLE_CACHE[cache_key] = bundle
    return bundle


def prediction_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    def safe(value: Any) -> float:
        parsed = parse_float(value)
        return parsed if parsed is not None and np.isfinite(parsed) else float("-inf")

    return (
        safe(item.get("selection_score")),
        safe(item.get("score")),
        safe(item.get("coverage")),
        safe(item.get("css")),
    )


def make_prediction_maps(configs: list[DatasetConfig], re_lists: dict[str, list[int]]):
    bundles: dict[tuple[str, int], dict[str, Any]] = {}
    selected_map: dict[ScanKey, dict[str, Any]] = {}
    raw_map: dict[ScanKey, dict[str, Any]] = {}

    for config in configs:
        for re_num in re_lists.get(config.name, []):
            bundle = get_scan_bundle(config, re_num)
            bundles[(config.name, re_num)] = bundle
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
                    key = ScanKey(config.name, config.family, re_num, ion_type, frag_len, charge)
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


DIAGNOSE_CACHE: dict[ScanKey, dict[str, Any]] = {}
TRACE_CACHE: dict[ScanKey, dict[str, Any]] = {}


def build_best_trace(bundles: dict[tuple[str, int], dict[str, Any]], key: ScanKey) -> dict[str, Any]:
    cached = TRACE_CACHE.get(key)
    if cached is not None:
        return cached

    bundle = bundles[(key.dataset, key.re_num)]
    residues = bundle["residues"]
    spectrum = bundle["spectrum"]
    isodec_config = bundle["isodecConfig"]
    noise_model = bundle["noiseModel"]

    with _override_cfg(bundle["overrides"]):
        frag_name, target_comp = BASE_MOD.ion_composition_from_sequence(residues, key.ion_type, key.pos, amidated=cfg.AMIDATED)
        variant_rows = BASE_MOD.get_disulfide_logic(key.ion_type, key.pos, len(residues)) or [("", None)]
        traces: list[dict[str, Any]] = []
        for variant_suffix, shift in variant_rows:
            try:
                variant_comp = target_comp + shift if shift is not None else target_comp
            except Exception:
                continue
            trace = BASE_MOD.execute_hybrid_strategy(
                TRACE_MOD._evaluate_fragment_variant_trace,
                residues,
                spectrum,
                isodec_config,
                noise_model,
                ion_type=key.ion_type,
                frag_len=int(key.pos),
                z=int(key.charge),
                frag_name=frag_name,
                variant_suffix=str(variant_suffix or ""),
                variant_comp=variant_comp,
            )
            if isinstance(trace, dict):
                traces.append(trace)

    if traces:
        best = max(traces, key=BASE_MOD._trace_sort_key)
    else:
        best = {
            "stage": "no_variant_trace",
            "accepted": False,
            "reason": "no_variant_trace",
            "ruleChecks": [],
            "unmetChecks": ["Variant trace"],
        }
    TRACE_CACHE[key] = best
    return best


def extract_fragments_metrics(
    bundles: dict[tuple[str, int], dict[str, Any]],
    selected_map: dict[ScanKey, dict[str, Any]],
    raw_map: dict[ScanKey, dict[str, Any]],
    key: ScanKey,
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
            "fragments_fit_score": parse_float(selected.get("fit_score")),
            "fragments_correlation": parse_float(selected.get("correlation_coefficient")),
            "fragments_pc_missing_peaks": parse_float(selected.get("pc_missing_peaks")),
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
            "fragments_fit_score": parse_float(raw.get("fit_score")),
            "fragments_correlation": parse_float(raw.get("correlation_coefficient")),
            "fragments_pc_missing_peaks": parse_float(raw.get("pc_missing_peaks")),
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
        "fragments_css": parse_float(trace.get("css")) or extract_from_rule(trace, "IsoDec CSS") or extract_from_rule(trace, "Fragments CSS"),
        "fragments_rawcos": parse_float(trace.get("rawCosine")) or extract_from_rule(trace, "Raw cosine"),
        "fragments_coverage": parse_float(trace.get("coverage")),
        "fragments_match_count": parse_float(trace.get("matchCount")) or extract_from_rule(trace, "Local matches"),
        "fragments_fit_score": extract_from_rule(trace, "Fit score"),
        "fragments_correlation": extract_from_rule(trace, "Correlation"),
        "fragments_pc_missing_peaks": extract_from_rule(trace, "Missing peaks"),
        "fragments_obs_rel_int": parse_float(trace.get("obsRelInt")),
        "fragments_abs_anchor_ppm": abs(ppm) if ppm is not None else extract_from_rule(trace, "Anchor ppm"),
        "fragments_raw_candidate": int(bool(trace.get("accepted"))),
        "fragments_selected_final": 0,
    }


def extract_diagnose_metrics(bundles: dict[tuple[str, int], dict[str, Any]], key: ScanKey) -> dict[str, Any]:
    cached = DIAGNOSE_CACHE.get(key)
    if cached is not None:
        return cached

    bundle = bundles[(key.dataset, key.re_num)]
    ion_spec = f"{key.ion_type}{key.pos}^{key.charge}+"
    with _override_cfg(bundle["overrides"]):
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
    }
    DIAGNOSE_CACHE[key] = metrics
    return metrics


def build_dataset(configs: list[DatasetConfig]) -> pd.DataFrame:
    manual_aggregates, re_lists = load_manual_aggregates(configs)
    with truthscore_disabled():
        bundles, selected_map, raw_map = make_prediction_maps(configs, re_lists)

        truth_true = {key for key, agg in manual_aggregates.items() if int(agg["truth"]) == 1}
        truth_false = {key for key, agg in manual_aggregates.items() if int(agg["truth"]) == 0}
        selected_negatives = set(selected_map) - truth_true
        candidate_keys = sorted(truth_true | truth_false | selected_negatives, key=lambda k: (k.dataset, k.re_num, k.ion_type, k.pos, k.charge))

        rows: list[dict[str, Any]] = []
        for key in candidate_keys:
            manual = manual_aggregates.get(
                key,
                {
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
                },
            )
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
                "dataset": key.dataset,
                "dataset_family": key.family,
                "scan_key": f"{key.dataset}:RE{key.re_num}",
                "re": f"RE{key.re_num}",
                "re_index": key.re_num,
                "ion_type": key.ion_type,
                "pos": key.pos,
                "charge": key.charge,
                "ion_type_is_c": 1 if key.ion_type == "c" else 0,
                "base_ion_spec": f"{key.ion_type}{key.pos}^{key.charge}+",
                "truth": truth,
                "candidate_group": candidate_group,
                "in_manual_annotation": int(annotated),
                "in_fragments_selected": int(selected),
                **manual,
                **extract_fragments_metrics(bundles, selected_map, raw_map, key),
                **extract_diagnose_metrics(bundles, key),
            }
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["dataset", "re_index", "ion_type", "pos", "charge"]).reset_index(drop=True)


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


def build_design_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]):
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
    return X_train, X_test, medians.to_dict(), means.to_dict(), stds.to_dict()


def predict_group_cv(df: pd.DataFrame, features: list[str], group_col: str) -> pd.DataFrame:
    rows = []
    for group in sorted(df[group_col].dropna().unique()):
        train_df = df[df[group_col] != group]
        test_df = df[df[group_col] == group]
        if train_df.empty or test_df.empty:
            continue
        y_train = train_df["truth"].to_numpy(dtype=float)
        if len(np.unique(y_train)) < 2:
            continue
        X_train, X_test, _, _, _ = build_design_matrices(train_df, test_df, features)
        w, b = fit_logistic_regression(X_train, y_train, reg=1.0)
        probs = special.expit(np.clip(X_test @ w + b, -40.0, 40.0))
        fold = test_df[["dataset", "dataset_family", "scan_key", "re", "base_ion_spec", "truth", "candidate_group"]].copy()
        fold["cv_group"] = str(group)
        fold["probability"] = probs
        rows.append(fold)
    if not rows:
        return pd.DataFrame(columns=["dataset", "dataset_family", "scan_key", "re", "base_ion_spec", "truth", "candidate_group", "cv_group", "probability"])
    return pd.concat(rows, ignore_index=True)


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= float(threshold)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def best_threshold_from_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    y = pred_df["truth"].to_numpy(dtype=float)
    p = pred_df["probability"].to_numpy(dtype=float)
    best = None
    for threshold in np.unique(np.round(p, 6)):
        metrics = metrics_at_threshold(y, p, float(threshold))
        candidate = (metrics["f1"], metrics["precision"], metrics["recall"], -abs(threshold - 0.5))
        if best is None or candidate > best[0]:
            best = (candidate, metrics)
    return best[1] if best is not None else metrics_at_threshold(y, p, 0.5)


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
        rho, _ = stats.spearmanr(x_valid, y_valid)
        auc_high = auc_from_scores(y_valid, x_valid)
        auc_best = None if auc_high is None else max(float(auc_high), float(1.0 - auc_high))
        best_direction = "higher" if auc_high is None or auc_high >= 0.5 else "lower"
        rows.append(
            {
                "feature": feature,
                "expected_direction": expected_direction,
                "best_direction": best_direction,
                "n": int(mask.sum()),
                "pos_mean": float(np.mean(pos)),
                "neg_mean": float(np.mean(neg)),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "auc_best_direction": float(auc_best) if auc_best is not None else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["auc_best_direction", "spearman_rho"], ascending=[False, False]).reset_index(drop=True)


def compute_combo_models(df: pd.DataFrame, max_size: int = 4) -> pd.DataFrame:
    rows = []
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(FEATURE_POOL, size):
            scan_pred = predict_group_cv(df, list(combo), "scan_key")
            if scan_pred.empty:
                continue
            scan_auc = auc_from_scores(scan_pred["truth"].to_numpy(dtype=float), scan_pred["probability"].to_numpy(dtype=float))
            family_pred = predict_group_cv(df, list(combo), "dataset_family")
            family_auc = auc_from_scores(family_pred["truth"].to_numpy(dtype=float), family_pred["probability"].to_numpy(dtype=float)) if not family_pred.empty else None
            rows.append(
                {
                    "features": ", ".join(combo),
                    "feature_count": size,
                    "scan_cv_auc": float(scan_auc) if scan_auc is not None else np.nan,
                    "family_cv_auc": float(family_auc) if family_auc is not None else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["scan_cv_auc", "family_cv_auc", "feature_count", "features"], ascending=[False, False, True, True]).reset_index(drop=True)


def fit_full_model_artifact(df: pd.DataFrame, features: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
    X_train, _, medians, means, stds = build_design_matrices(df, df, features)
    y = df["truth"].to_numpy(dtype=float)
    w, b = fit_logistic_regression(X_train, y, reg=1.0)
    coef_rows = []
    for feature, coef in zip(features, w):
        coef_rows.append(
            {
                "feature": feature,
                "direction": DIRECTION_MAP.get(feature),
                "coefficient": float(coef),
                "mean": float(means[feature]),
                "std": float(stds[feature]),
                "median_fill": float(medians[feature]),
            }
        )
    artifact = {
        "algorithm": "bounded_logistic_regression",
        "features": features,
        "intercept": float(b),
        "coefficients": {row["feature"]: row["coefficient"] for row in coef_rows},
        "directions": {row["feature"]: row["direction"] for row in coef_rows},
        "median_fill": {row["feature"]: row["median_fill"] for row in coef_rows},
        "means": {row["feature"]: row["mean"] for row in coef_rows},
        "stds": {row["feature"]: row["std"] for row in coef_rows},
    }
    return artifact, pd.DataFrame(coef_rows)


def write_summary(
    df: pd.DataFrame,
    univariate: pd.DataFrame,
    combo_models: pd.DataFrame,
    best_features: list[str],
    scan_metrics_050: dict[str, float],
    scan_metrics_best: dict[str, float],
    family_metrics_050: dict[str, float],
    family_metrics_best: dict[str, float],
) -> None:
    total = len(df)
    positives = int(df["truth"].sum())
    negatives = total - positives
    lines = []
    lines.append("# Multi-Dataset b/c Truth Model")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Datasets: `Q10R`, `WT_2025-02`, `WT_2025-03`, `S20G`")
    lines.append("- Family groups: `Q10R`, `WT`, `S20G`")
    lines.append("- Only `b/c` ions")
    lines.append("- Truth: manual `Matched=1`")
    lines.append("- Precursor lock-mass / chain-to-fragments: `off` during feature extraction")
    lines.append("- Existing `truth_score`: `off` during feature extraction to avoid using the previous learned score as an input")
    lines.append("- Model: bounded logistic regression (fallback because `scikit-learn` is not installed in this environment)")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Total candidates: `{total}`")
    lines.append(f"- Positives: `{positives}`")
    lines.append(f"- Negatives: `{negatives}`")
    for family, count in Counter(df["dataset_family"]).items():
        lines.append(f"- Candidates in `{family}`: `{count}`")
    lines.append("")
    lines.append("## Best Feature Set")
    lines.append("")
    lines.append(f"- Selected by scan-holdout AUC: `{', '.join(best_features)}`")
    lines.append("")
    lines.append("## Scan-Holdout CV")
    lines.append("")
    lines.append(f"- Threshold `0.50`: precision `{scan_metrics_050['precision']:.3f}`, recall `{scan_metrics_050['recall']:.3f}`, F1 `{scan_metrics_050['f1']:.3f}`")
    lines.append(f"- Best F1 threshold `{scan_metrics_best['threshold']:.6f}`: precision `{scan_metrics_best['precision']:.3f}`, recall `{scan_metrics_best['recall']:.3f}`, F1 `{scan_metrics_best['f1']:.3f}`")
    lines.append("")
    lines.append("## Family-Holdout CV")
    lines.append("")
    lines.append(f"- Threshold `0.50`: precision `{family_metrics_050['precision']:.3f}`, recall `{family_metrics_050['recall']:.3f}`, F1 `{family_metrics_050['f1']:.3f}`")
    lines.append(f"- Best F1 threshold `{family_metrics_best['threshold']:.6f}`: precision `{family_metrics_best['precision']:.3f}`, recall `{family_metrics_best['recall']:.3f}`, F1 `{family_metrics_best['f1']:.3f}`")
    lines.append("")
    lines.append("## Strongest Single Features")
    lines.append("")
    for _, row in univariate.head(8).iterrows():
        lines.append(f"- `{row['feature']}`: AUC `{row['auc_best_direction']:.3f}`, Spearman `{row['spearman_rho']:.3f}`, best direction `{row['best_direction']}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- Dataset: `{DATASET_CSV.relative_to(ROOT)}`")
    lines.append(f"- Combo models: `{MODELS_CSV.relative_to(ROOT)}`")
    lines.append(f"- Univariate stats: `{UNIVARIATE_CSV.relative_to(ROOT)}`")
    lines.append(f"- Scan CV predictions: `{SCAN_PRED_CSV.relative_to(ROOT)}`")
    lines.append(f"- Family CV predictions: `{FAMILY_PRED_CSV.relative_to(ROOT)}`")
    lines.append(f"- Coefficients: `{COEFFICIENTS_CSV.relative_to(ROOT)}`")
    lines.append(f"- Model artifact: `{MODEL_JSON.relative_to(ROOT)}`")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    configs = dataset_configs()
    df = build_dataset(configs)
    univariate = compute_univariate_stats(df)
    combo_models = compute_combo_models(df)
    if combo_models.empty:
        raise RuntimeError("No valid models could be fit")

    best_features = [part.strip() for part in str(combo_models.iloc[0]["features"]).split(",")]
    scan_pred = predict_group_cv(df, best_features, "scan_key")
    family_pred = predict_group_cv(df, best_features, "dataset_family")
    scan_metrics_050 = metrics_at_threshold(scan_pred["truth"].to_numpy(dtype=float), scan_pred["probability"].to_numpy(dtype=float), 0.5)
    scan_metrics_best = best_threshold_from_predictions(scan_pred)
    family_metrics_050 = metrics_at_threshold(family_pred["truth"].to_numpy(dtype=float), family_pred["probability"].to_numpy(dtype=float), 0.5)
    family_metrics_best = best_threshold_from_predictions(family_pred)
    artifact, coef_df = fit_full_model_artifact(df, best_features)
    artifact["best_scan_cv_threshold"] = scan_metrics_best["threshold"]
    artifact["best_family_cv_threshold"] = family_metrics_best["threshold"]
    artifact["scan_cv_metrics_at_0_5"] = scan_metrics_050
    artifact["scan_cv_metrics_best_f1"] = scan_metrics_best
    artifact["family_cv_metrics_at_0_5"] = family_metrics_050
    artifact["family_cv_metrics_best_f1"] = family_metrics_best

    df.to_csv(DATASET_CSV, index=False)
    combo_models.to_csv(MODELS_CSV, index=False)
    univariate.to_csv(UNIVARIATE_CSV, index=False)
    scan_pred.to_csv(SCAN_PRED_CSV, index=False)
    family_pred.to_csv(FAMILY_PRED_CSV, index=False)
    coef_df.to_csv(COEFFICIENTS_CSV, index=False)
    MODEL_JSON.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_summary(df, univariate, combo_models, best_features, scan_metrics_050, scan_metrics_best, family_metrics_050, family_metrics_best)

    print(f"Dataset rows: {len(df)}")
    print(f"Positives: {int(df['truth'].sum())}")
    print(f"Negatives: {len(df) - int(df['truth'].sum())}")
    print(f"Best features: {', '.join(best_features)}")
    print(f"Scan CV F1@0.5: {scan_metrics_050['f1']:.3f}")
    print(f"Scan CV best-F1: {scan_metrics_best['f1']:.3f} @ {scan_metrics_best['threshold']:.6f}")
    print(f"Family CV F1@0.5: {family_metrics_050['f1']:.3f}")
    print(f"Family CV best-F1: {family_metrics_best['f1']:.3f} @ {family_metrics_best['threshold']:.6f}")


if __name__ == "__main__":
    main()
