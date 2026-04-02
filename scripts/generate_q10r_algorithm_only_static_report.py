from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ecd_api import (
    DiagnoseRunRequest,
    FragmentsRunRequest,
    _build_overrides,
    _override_cfg,
    _run_diagnose_impl,
    _run_fragments_impl,
    parse_custom_sequence,
)
import personalized_config as cfg
from personalized import load_spectrum, preprocess_spectrum
from personalized_match import (
    execute_hybrid_strategy,
    get_local_centroids_window,
    isodec_css_and_accept,
    parse_fragment_spec,
    nearest_peak_index,
    within_ppm,
)
from personalized_modes import (
    _build_noise_level_model,
    _composite_match_components,
    _fragment_noise_core_components,
    _match_theory_local_monotonic,
    run_fragments_headless,
    run_precursor_headless,
)
from personalized_sequence import (
    get_disulfide_logic,
    ion_composition_from_sequence,
    ion_series,
    neutral_loss_label,
)
from personalized_theory import (
    build_sample_axis,
    css_similarity,
    fit_simplex_mixture,
    get_anchor_idx,
    observed_intensities_isodec,
    theoretical_isodist_from_comp,
    vectorize_dist,
)


REPORT_DIR = ROOT / "reports" / "q10r_algorithm_only_static"
ASSETS_DIR = REPORT_DIR / "assets"
ALGORITHM_ONLY_CSV = ROOT / "match_outputs" / "q10r_bc_algorithm_only_manual_unmatched.csv"
ANNOTATION_DIR = ROOT / "sample" / "Q10R" / "Q10R_annotated"
SPECTRUM_DIR = ROOT / "sample" / "Q10R" / "Centroid (lock mass)_副本"

PEPTIDE = "KCNTATCATRRLANFLVHSSNNFGAILSSTNVGSNTY"
COPIES = 2
AMIDATED = True
DISULFIDE_BONDS = 2
DISULFIDE_MAP = "2-7"
FRAG_MIN_CHARGE = 1
FRAG_MAX_CHARGE = 5

UI_PAPER = "#ffffff"
UI_PANEL = "#f8fafc"
UI_GRID = "#dbeafe"
UI_INK = "#0f172a"
UI_MUTED = "#475569"
UI_PURPLE = "#8b5cf6"
UI_GREEN = "#22c55e"
UI_RED = "#ef4444"
UI_BLUE = "#1e3a8a"
UI_AMBER = "#f59e0b"

SCAN_BUNDLE_CACHE: dict[int, dict[str, Any]] = {}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def quiet_call(func, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(*args, **kwargs)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def format_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def bool_text(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if bool(value) else "no"


def make_rule_check(
    label: str,
    *,
    state: str,
    value_text: str,
    target_text: str = "",
    note: str = "",
) -> dict[str, str]:
    return {
        "label": label,
        "state": state,
        "valueText": value_text,
        "targetText": target_text,
        "note": note,
    }


def build_diagnose_rule_checks(best: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    detail = best.get("isodec_detail") if isinstance(best.get("isodec_detail"), dict) else {}
    reason = str(best.get("reason") or "")

    checks: list[dict[str, str]] = []
    unmet: list[str] = []

    anchor_ppm = parse_float(best.get("anchor_ppm"))
    anchor_target = float(getattr(cfg, "MATCH_TOL_PPM", 30.0))
    anchor_within = bool(best.get("anchor_within_ppm", False)) and anchor_ppm is not None
    if anchor_ppm is None:
        checks.append(
            make_rule_check(
                "Anchor ppm",
                state="fail" if reason == "anchor_outside_ppm" else "unavailable",
                value_text="n/a",
                target_text=f"within +/-{anchor_target:.2f} ppm",
                note="no anchor peak found",
            )
        )
        if reason == "anchor_outside_ppm":
            unmet.append("Anchor ppm")
    else:
        anchor_state = "pass" if anchor_within else "fail"
        checks.append(
            make_rule_check(
                "Anchor ppm",
                state=anchor_state,
                value_text=f"{anchor_ppm:+.2f} ppm",
                target_text=f"within +/-{anchor_target:.2f} ppm",
            )
        )
        if anchor_state == "fail":
            unmet.append("Anchor ppm")

    css = parse_float(best.get("isodec_css"))
    css_thresh = parse_float(detail.get("css_thresh"))
    if css is None:
        checks.append(
            make_rule_check(
                "IsoDec CSS",
                state="unavailable",
                value_text="n/a",
                target_text=f">= {float(getattr(cfg, 'ISODEC_CSS_THRESH', 0.70)):.3f}",
            )
        )
    else:
        if css_thresh is None:
            css_thresh = float(getattr(cfg, "ISODEC_CSS_THRESH", 0.70))
        css_state = "pass" if css >= css_thresh else "fail"
        checks.append(
            make_rule_check(
                "IsoDec CSS",
                state=css_state,
                value_text=f"{css:.3f}",
                target_text=f">= {css_thresh:.3f}",
            )
        )
        if css_state == "fail":
            unmet.append("IsoDec CSS")

    matched_n = detail.get("matched_peaks_n")
    minpeaks = detail.get("minpeaks_effective")
    if matched_n is None or minpeaks is None:
        checks.append(
            make_rule_check(
                "Matched peaks",
                state="unavailable",
                value_text="n/a",
                target_text=f">= {int(getattr(cfg, 'ISODEC_MINPEAKS', 2))}",
            )
        )
    else:
        matched_state = "pass" if int(matched_n) >= int(minpeaks) else "fail"
        checks.append(
            make_rule_check(
                "Matched peaks",
                state=matched_state,
                value_text=str(int(matched_n)),
                target_text=f">= {int(minpeaks)}",
            )
        )
        if matched_state == "fail":
            unmet.append("Matched peaks")

    area = parse_float(detail.get("areacovered"))
    area_thresh = parse_float(detail.get("minareacovered"))
    if area is None or area_thresh is None:
        area_default = float(getattr(cfg, "ISODEC_MIN_AREA_COVERED", 0.10))
        checks.append(
            make_rule_check(
                "Area covered",
                state="unavailable",
                value_text="n/a",
                target_text=f">= {area_default:.3f}",
            )
        )
    else:
        area_state = "pass" if area >= area_thresh else "fail"
        checks.append(
            make_rule_check(
                "Area covered",
                state=area_state,
                value_text=f"{area:.3f}",
                target_text=f">= {area_thresh:.3f}",
            )
        )
        if area_state == "fail":
            unmet.append("Area covered")

    topthree = detail.get("topthree")
    if topthree is None:
        checks.append(
            make_rule_check(
                "Top peaks",
                state="unavailable",
                value_text="n/a",
                target_text="required",
            )
        )
    else:
        top_state = "pass" if bool(topthree) else "fail"
        checks.append(
            make_rule_check(
                "Top peaks",
                state=top_state,
                value_text=bool_text(topthree),
                target_text="need yes",
            )
        )
        if top_state == "fail":
            unmet.append("Top peaks")

    raw_cos = parse_float(best.get("raw_cosine"))
    checks.append(
        make_rule_check(
            "Raw cosine",
            state="info",
            value_text=format_num(raw_cos, 3),
            note="pre-anchor diagnose cosine",
        )
    )

    return checks, unmet


def get_fragments_scan_bundle(re_num: int) -> dict[str, Any]:
    re_num = int(re_num)
    cached = SCAN_BUNDLE_CACHE.get(re_num)
    if cached is not None:
        return cached

    spectrum_path = SPECTRUM_DIR / f"ECDRE{re_num}.txt"
    req = FragmentsRunRequest(
        filepath=str(spectrum_path.resolve()),
        scan=1,
        peptide=PEPTIDE,
        ion_types=["b", "c"],
        frag_min_charge=FRAG_MIN_CHARGE,
        frag_max_charge=FRAG_MAX_CHARGE,
        copies=COPIES,
        amidated=AMIDATED,
        disulfide_bonds=DISULFIDE_BONDS,
        disulfide_map=DISULFIDE_MAP,
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
        "reNum": re_num,
        "overrides": overrides,
        "residues": residues,
        "spectrum": np.asarray(spectrum, dtype=float),
        "isodecConfig": isodec_config,
        "fragmentsResult": fragments_result,
        "noiseModel": noise_model,
        "precursorResult": precursor_result,
    }
    SCAN_BUNDLE_CACHE[re_num] = bundle
    return bundle


def _trace_rank_value(stage: str) -> int:
    ranks = {
        "accepted": 6,
        "quality_gates": 5,
        "local_window": 4,
        "isodec_rules": 3,
        "anchor_search": 2,
        "model_setup": 1,
        "no_theory": 0,
    }
    return int(ranks.get(stage, 0))


def _trace_sort_key(trace: dict[str, Any]) -> tuple[float, ...]:
    def _safe(value: Any) -> float:
        try:
            val = float(value)
        except Exception:
            return float("-inf")
        return val if np.isfinite(val) else float("-inf")

    return (
        float(bool(trace.get("accepted"))),
        float(_trace_rank_value(str(trace.get("stage") or ""))),
        _safe(trace.get("score")),
        _safe(trace.get("coverage")),
        _safe(trace.get("css")),
        _safe(trace.get("rawCosine")),
    )


def _evaluate_fragment_variant_trace(
    residues: list[tuple[str, list[str]]],
    spectrum: np.ndarray,
    isodec_config,
    noise_model: dict[str, Any],
    *,
    ion_type: str,
    frag_len: int,
    z: int,
    frag_name: str,
    variant_suffix: str,
    variant_comp,
    use_centroid_logic: bool = True,
) -> dict[str, Any]:
    spectrum_mz = np.asarray(spectrum[:, 0], dtype=float)
    spectrum_int = np.asarray(spectrum[:, 1], dtype=float)
    obs_max = float(np.max(spectrum_int)) if spectrum_int.size else 0.0
    match_tol_ppm = float(cfg.MATCH_TOL_PPM)
    series = ion_series(ion_type)
    allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_1H))
    allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_2H))

    trace: dict[str, Any] = {
        "candidateLabel": f"{frag_name}{variant_suffix}^{int(z)}+",
        "fragName": frag_name,
        "variantSuffix": variant_suffix,
        "charge": int(z),
        "stage": "model_setup",
        "accepted": False,
        "reason": "candidate_setup_failed",
        "ruleChecks": [],
        "unmetChecks": [],
    }

    try:
        dist0 = theoretical_isodist_from_comp(variant_comp, z)
    except ValueError:
        trace["stage"] = "no_theory"
        trace["reason"] = "no_theoretical_distribution"
        trace["ruleChecks"] = [
            make_rule_check(
                "Theoretical model",
                state="fail",
                value_text="n/a",
                note="could not build isotopic distribution",
            )
        ]
        trace["unmetChecks"] = ["Theoretical model"]
        return trace

    if dist0.size == 0:
        trace["stage"] = "no_theory"
        trace["reason"] = "empty_theoretical_distribution"
        trace["ruleChecks"] = [
            make_rule_check(
                "Theoretical model",
                state="fail",
                value_text="empty",
                note="no isotopic peaks available",
            )
        ]
        trace["unmetChecks"] = ["Theoretical model"]
        return trace

    shift_1 = float(cfg.H_TRANSFER_MASS) / float(z) if (allow_1h or allow_2h) else 0.0
    shift_2 = 2.0 * float(cfg.H_TRANSFER_MASS) / float(z) if allow_2h else 0.0

    dist_p1 = dist0.copy()
    dist_p1[:, 0] += shift_1
    dist_m1 = dist0.copy()
    dist_m1[:, 0] -= shift_1

    dist_p2 = None
    dist_m2 = None
    if allow_2h:
        dist_p2 = dist0.copy()
        dist_p2[:, 0] += shift_2
        dist_m2 = dist0.copy()
        dist_m2[:, 0] -= shift_2

    dists_for_union = [dist0]
    if allow_1h:
        dists_for_union.extend([dist_p1, dist_m1])
    if allow_2h:
        dists_for_union.extend([dist_p2, dist_m2])

    sample_keys, sample_mzs, scale = build_sample_axis(
        dists_for_union,
        decimals=6,
        mz_min=cfg.MZ_MIN,
        mz_max=cfg.MZ_MAX,
    )
    if len(sample_mzs) == 0:
        trace["stage"] = "model_setup"
        trace["reason"] = "empty_sample_axis"
        trace["ruleChecks"] = [
            make_rule_check(
                "Sample axis",
                state="fail",
                value_text="empty",
                note="no peaks remain inside fragments m/z window",
            )
        ]
        trace["unmetChecks"] = ["Sample axis"]
        return trace

    peak_mz = float(dist0[get_anchor_idx(dist0), 0])
    y_obs = observed_intensities_isodec(
        spectrum_mz,
        spectrum_int,
        sample_mzs,
        z=int(z),
        match_tol_ppm=match_tol_ppm,
        peak_mz=peak_mz,
    )
    y0 = vectorize_dist(dist0, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX)

    neutral_score_union = css_similarity(y_obs, y0)
    neutral_score = neutral_score_union
    dist0_neutral = dist0
    if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
        mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
        mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
        dist0_neutral = dist0[(dist0[:, 0] >= mz_min) & (dist0[:, 0] <= mz_max)]
    if dist0_neutral.size:
        y_obs_neutral = observed_intensities_isodec(
            spectrum_mz,
            spectrum_int,
            dist0_neutral[:, 0],
            z=int(z),
            match_tol_ppm=match_tol_ppm,
            peak_mz=peak_mz,
        )
        neutral_score = css_similarity(y_obs_neutral, dist0_neutral[:, 1])

    best_model = "neutral"
    best_score = neutral_score_union
    best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}
    best_pred = y0

    if allow_1h or allow_2h:
        yp1 = vectorize_dist(dist_p1, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX) if allow_1h else None
        ym1 = vectorize_dist(dist_m1, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX) if allow_1h else None
        yp2 = vectorize_dist(dist_p2, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX) if allow_2h else None
        ym2 = vectorize_dist(dist_m2, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX) if allow_2h else None

        comps_plus = [("0", y0)]
        comps_minus = [("0", y0)]
        if allow_1h:
            comps_plus.append(("+H", yp1))
            comps_minus.append(("-H", ym1))
        if allow_2h:
            comps_plus.append(("+2H", yp2))
            comps_minus.append(("-2H", ym2))

        names_plus, vecs_plus = zip(*comps_plus)
        w_plus, y_plus, score_plus = fit_simplex_mixture(y_obs, list(vecs_plus))
        weights_plus = dict(zip(names_plus, w_plus))

        names_minus, vecs_minus = zip(*comps_minus)
        w_minus, y_minus, score_minus = fit_simplex_mixture(y_obs, list(vecs_minus))
        weights_minus = dict(zip(names_minus, w_minus))

        if score_plus > best_score:
            best_model = "+mix"
            best_score = score_plus
            best_pred = y_plus
            best_weights = {
                "0": weights_plus.get("0", 0.0),
                "+H": weights_plus.get("+H", 0.0),
                "+2H": weights_plus.get("+2H", 0.0),
                "-H": 0.0,
                "-2H": 0.0,
            }

        if score_minus > best_score:
            best_model = "-mix"
            best_score = score_minus
            best_pred = y_minus
            best_weights = {
                "0": weights_minus.get("0", 0.0),
                "+H": 0.0,
                "+2H": 0.0,
                "-H": weights_minus.get("-H", 0.0),
                "-2H": weights_minus.get("-2H", 0.0),
            }

        rel_improve = (best_score - neutral_score_union) / max(neutral_score_union, 1e-12)
        if best_model != "neutral" and rel_improve < float(cfg.H_TRANSFER_MIN_REL_IMPROVEMENT):
            best_model = "neutral"
            best_score = neutral_score_union
            best_pred = y0
            best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}
    if best_model == "neutral":
        best_score = neutral_score

    trace["rawCosine"] = float(best_score)
    trace["bestModel"] = best_model
    trace["hWeights"] = best_weights

    if float(np.max(best_pred)) <= 0.0:
        trace["stage"] = "model_setup"
        trace["reason"] = "zero_model_intensity"
        trace["ruleChecks"] = [
            make_rule_check(
                "Model intensity",
                state="fail",
                value_text="0.000",
                note="best fragments model has no positive intensity",
            ),
            make_rule_check("Raw cosine", state="info", value_text=format_num(best_score, 3)),
        ]
        trace["unmetChecks"] = ["Model intensity"]
        return trace

    dist_model = np.column_stack([sample_mzs.copy(), best_pred.copy()])
    if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
        mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
        mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
        dist_model = dist_model[(dist_model[:, 0] >= mz_min) & (dist_model[:, 0] <= mz_max)]
    if dist_model.size == 0:
        trace["stage"] = "model_setup"
        trace["reason"] = "empty_model_after_mz_window"
        trace["ruleChecks"] = [
            make_rule_check(
                "Model window",
                state="fail",
                value_text="empty",
                note="all modeled peaks fell outside fragments m/z window",
            )
        ]
        trace["unmetChecks"] = ["Model window"]
        return trace

    max_model = float(np.max(dist_model[:, 1]))
    if max_model <= 0.0:
        trace["stage"] = "model_setup"
        trace["reason"] = "non_positive_model_after_window"
        trace["ruleChecks"] = [
            make_rule_check(
                "Model intensity",
                state="fail",
                value_text="0.000",
                note="modeled peaks are non-positive after windowing",
            )
        ]
        trace["unmetChecks"] = ["Model intensity"]
        return trace

    keep_model = dist_model[:, 1] >= max_model * float(cfg.REL_INTENSITY_CUTOFF)
    dist_model = dist_model[keep_model]
    if dist_model.size == 0:
        trace["stage"] = "model_setup"
        trace["reason"] = "below_relative_intensity_cutoff"
        trace["ruleChecks"] = [
            make_rule_check(
                "Relative intensity cutoff",
                state="fail",
                value_text="empty",
                target_text=f">= {float(cfg.REL_INTENSITY_CUTOFF):.2f} of max",
            )
        ]
        trace["unmetChecks"] = ["Relative intensity cutoff"]
        return trace

    anchor_theory_mz = None
    obs_idx = None
    obs_mz = None
    obs_int = None
    anchor_hits = 0
    anchor_window = float(getattr(cfg, "FRAG_ANCHOR_CENTROID_WINDOW_DA", 0.2))
    sorted_idx = np.argsort(best_pred)[::-1][: int(cfg.ANCHOR_TOP_N)]
    for idx in sorted_idx:
        mz_candidate = float(sample_mzs[int(idx)])
        local_centroids = get_local_centroids_window(
            spectrum_mz,
            spectrum_int,
            center_mz=mz_candidate,
            lb=-anchor_window,
            ub=anchor_window,
            force_hill=bool(use_centroid_logic),
        )
        if isinstance(local_centroids, np.ndarray) and local_centroids.size:
            best_local_idx = int(np.argmax(local_centroids[:, 1]))
            obs_mz_c = float(local_centroids[best_local_idx, 0])
            obs_int_c = float(local_centroids[best_local_idx, 1])
            obs_idx_c = nearest_peak_index(spectrum_mz, obs_mz_c)
        else:
            obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
            obs_mz_c = float(spectrum_mz[obs_idx_c])
            obs_int_c = float(spectrum_int[obs_idx_c])
        if not within_ppm(obs_mz_c, mz_candidate, match_tol_ppm):
            continue
        if float(cfg.MIN_OBS_REL_INT) > 0 and obs_int_c < obs_max * float(cfg.MIN_OBS_REL_INT):
            continue
        anchor_hits += 1
        if anchor_theory_mz is None:
            anchor_theory_mz = mz_candidate
            obs_idx = int(obs_idx_c)
            obs_mz = obs_mz_c
            obs_int = obs_int_c

    if anchor_hits < int(cfg.ANCHOR_MIN_MATCHES) or anchor_theory_mz is None:
        trace["stage"] = "anchor_search"
        trace["reason"] = "no_anchor_match"
        trace["ruleChecks"] = [
            make_rule_check(
                "Anchor search",
                state="fail",
                value_text=str(int(anchor_hits)),
                target_text=f">= {int(cfg.ANCHOR_MIN_MATCHES)} hits",
                note=f"top {int(cfg.ANCHOR_TOP_N)} modeled peaks checked",
            ),
            make_rule_check("Raw cosine", state="info", value_text=format_num(best_score, 3)),
        ]
        trace["unmetChecks"] = ["Anchor search"]
        return trace

    ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6
    dist_plot = dist_model.copy()
    dist_plot[:, 0] += obs_mz - anchor_theory_mz
    dist_plot[:, 1] *= obs_int / float(np.max(dist_plot[:, 1]))

    isodec_css = float(best_score)
    if cfg.ENABLE_ISODEC_RULES and isodec_config is not None:
        local_centroids = get_local_centroids_window(
            spectrum_mz,
            spectrum_int,
            obs_mz,
            isodec_config.mzwindowlb,
            isodec_config.mzwindowub,
            force_hill=bool(use_centroid_logic),
        )
        accepted_iso, isodec_css, shifted_peak = isodec_css_and_accept(
            local_centroids,
            dist_plot,
            z=z,
            peakmz=obs_mz,
            config=isodec_config,
        )
        if not accepted_iso:
            trace["stage"] = "isodec_rules"
            trace["reason"] = "fragments_isodec_rejected"
            trace["css"] = float(isodec_css)
            trace["anchorPpm"] = float(ppm)
            trace["ruleChecks"] = [
                make_rule_check("Anchor ppm", state="pass", value_text=f"{ppm:+.2f} ppm", target_text=f"within +/-{match_tol_ppm * 1.5:.2f} ppm"),
                make_rule_check("Fragments IsoDec", state="fail", value_text=format_num(isodec_css, 3), note="strict fragments IsoDec accept failed"),
                make_rule_check("Raw cosine", state="info", value_text=format_num(best_score, 3)),
            ]
            trace["unmetChecks"] = ["Fragments IsoDec"]
            return trace
        if shifted_peak is not None:
            old_obs_mz = obs_mz
            obs_mz_new = float(shifted_peak)
            obs_idx = nearest_peak_index(spectrum_mz, obs_mz_new)
            obs_mz = float(spectrum_mz[obs_idx])
            obs_int = float(spectrum_int[obs_idx])
            ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6
            dist_plot[:, 0] += obs_mz - old_obs_mz

    local_lb = (
        float(isodec_config.mzwindowlb)
        if isodec_config is not None and hasattr(isodec_config, "mzwindowlb")
        else float(cfg.ISODEC_MZ_WINDOW_LB)
    )
    local_ub = (
        float(isodec_config.mzwindowub)
        if isodec_config is not None and hasattr(isodec_config, "mzwindowub")
        else float(cfg.ISODEC_MZ_WINDOW_UB)
    )
    local_centroids = get_local_centroids_window(
        spectrum_mz,
        spectrum_int,
        obs_mz,
        local_lb,
        local_ub,
        force_hill=bool(use_centroid_logic),
    )
    if not isinstance(local_centroids, np.ndarray) or local_centroids.size == 0:
        local_mask = (spectrum_mz >= obs_mz + local_lb) & (spectrum_mz <= obs_mz + local_ub)
        if not np.any(local_mask):
            trace["stage"] = "local_window"
            trace["reason"] = "empty_local_window"
            trace["css"] = float(isodec_css)
            trace["anchorPpm"] = float(ppm)
            trace["ruleChecks"] = [
                make_rule_check("Local window", state="fail", value_text="empty", note="no local peaks remain around anchor"),
                make_rule_check("IsoDec CSS", state="info", value_text=format_num(isodec_css, 3)),
            ]
            trace["unmetChecks"] = ["Local window"]
            return trace
        local_centroids = np.column_stack((spectrum_mz[local_mask], spectrum_int[local_mask]))
    if local_centroids.ndim != 2 or local_centroids.shape[1] != 2:
        trace["stage"] = "local_window"
        trace["reason"] = "invalid_local_window"
        trace["ruleChecks"] = [
            make_rule_check("Local window", state="fail", value_text="invalid", note="local peak matrix has wrong shape"),
        ]
        trace["unmetChecks"] = ["Local window"]
        return trace

    local_max_int = float(np.max(local_centroids[:, 1])) if local_centroids.size else 0.0
    if local_max_int <= 0.0:
        trace["stage"] = "local_window"
        trace["reason"] = "zero_local_intensity"
        trace["ruleChecks"] = [
            make_rule_check("Local intensity", state="fail", value_text="0.000", note="all local peaks are zero"),
        ]
        trace["unmetChecks"] = ["Local intensity"]
        return trace

    max_anchor_abs_ppm_cfg = getattr(cfg, "FRAG_MAX_ANCHOR_ABS_PPM", None)
    max_anchor_abs_ppm = float(max_anchor_abs_ppm_cfg) if max_anchor_abs_ppm_cfg is not None else (match_tol_ppm * 1.5)
    max_residual_rmse_cfg = getattr(cfg, "FRAG_MAX_RESIDUAL_RMSE_PPM", None)
    max_residual_rmse_ppm = float(max_residual_rmse_cfg) if max_residual_rmse_cfg is not None else float(match_tol_ppm)
    max_mass_error_std_cfg = getattr(cfg, "FRAG_MAX_MASS_ERROR_STD_PPM", None)
    max_mass_error_std_ppm = float(max_mass_error_std_cfg) if max_mass_error_std_cfg is not None else None
    ppm_sigma_cfg = getattr(cfg, "FRAG_PPM_SIGMA", None)
    ppm_sigma = float(ppm_sigma_cfg) if ppm_sigma_cfg is not None else float(match_tol_ppm)
    if ppm_sigma <= 0:
        ppm_sigma = max(match_tol_ppm, 1.0)
    spacing_sigma_cfg = getattr(cfg, "FRAG_SPACING_SIGMA_DA", None)
    if spacing_sigma_cfg is None:
        spacing_sigma_da = abs(anchor_theory_mz) * match_tol_ppm * 1e-6
    else:
        spacing_sigma_da = float(spacing_sigma_cfg)
    if spacing_sigma_da <= 0:
        spacing_sigma_da = max(abs(anchor_theory_mz) * match_tol_ppm * 1e-6, 1e-6)

    score_w_css = float(getattr(cfg, "FRAG_SCORE_W_CSS", 0.40))
    score_w_cov = float(getattr(cfg, "FRAG_SCORE_W_COVERAGE", 0.20))
    score_w_ppm = float(getattr(cfg, "FRAG_SCORE_W_PPM", 0.15))
    score_w_spacing = float(getattr(cfg, "FRAG_SCORE_W_SPACING", 0.10))
    score_w_intensity = float(getattr(cfg, "FRAG_SCORE_W_INTENSITY", 0.15))
    score_w_fit = float(getattr(cfg, "FRAG_SCORE_W_FIT", 0.10))
    score_w_correlation = float(getattr(cfg, "FRAG_SCORE_W_CORRELATION", 0.05))
    score_w_snr = float(getattr(cfg, "FRAG_SCORE_W_SNR", 0.05))
    score_w_sum = score_w_css + score_w_cov + score_w_ppm + score_w_spacing + score_w_intensity + score_w_fit + score_w_correlation + score_w_snr
    if score_w_sum > 0:
        score_w_css /= score_w_sum
        score_w_cov /= score_w_sum
        score_w_ppm /= score_w_sum
        score_w_spacing /= score_w_sum
        score_w_intensity /= score_w_sum
        score_w_fit /= score_w_sum
        score_w_correlation /= score_w_sum
        score_w_snr /= score_w_sum

    obs_rel_int = float(obs_int / obs_max) if obs_max > 0 else 0.0
    anchor_rel_int = float(np.clip(obs_int / local_max_int, 0.0, 1.0))
    shift_da_local = float(obs_mz - anchor_theory_mz)
    local_matches = _match_theory_local_monotonic(local_centroids, dist_model, shift_da_local, match_tol_ppm)

    comp = _composite_match_components(
        css=float(isodec_css),
        matches=local_matches,
        dist_shifted=dist_model,
        anchor_ppm_abs=abs(float(ppm)),
        anchor_theory_mz=float(anchor_theory_mz),
        intensity_ratio=float(anchor_rel_int),
        ppm_sigma=float(ppm_sigma),
        spacing_sigma_da=float(spacing_sigma_da),
        score_w_css=float(score_w_css),
        score_w_cov=float(score_w_cov),
        score_w_ppm=float(score_w_ppm),
        score_w_spacing=float(score_w_spacing),
        score_w_intensity=float(score_w_intensity),
    )

    quality = _fragment_noise_core_components(
        spectrum_mz=spectrum_mz,
        spectrum_int=spectrum_int,
        local=local_centroids,
        matches=local_matches,
        dist_shifted=dist_plot,
        core_top_n=max(1, int(getattr(cfg, "FRAG_CORE_TOP_N", 3))),
        base_score=float(comp["composite_score"]),
        ppm_sigma=float(ppm_sigma),
        anchor_mz=float(obs_mz),
        noise_model=noise_model,
        score_w_fit=float(score_w_fit),
        score_w_correlation=float(score_w_correlation),
        score_w_snr=float(score_w_snr),
        s2n_scale=max(float(getattr(cfg, "FRAG_S2N_SCORE_SCALE", 4.0)), 1e-6),
        penalty_unexplained=float(getattr(cfg, "FRAG_SCORE_PENALTY_UNEXPLAINED", 0.35)),
        penalty_missing_core=float(getattr(cfg, "FRAG_SCORE_PENALTY_MISSING_CORE", 0.25)),
        penalty_missing_peaks=float(getattr(cfg, "FRAG_SCORE_PENALTY_MISSING_PEAKS", 0.10)),
        penalty_mass_error_std=float(getattr(cfg, "FRAG_SCORE_PENALTY_MASS_ERROR_STD", 0.10)),
    )

    min_matched_peaks = max(1, int(getattr(cfg, "FRAG_MIN_MATCHED_PEAKS", 2)))
    min_coverage = float(getattr(cfg, "FRAG_MIN_COVERAGE", 0.25))
    min_s2n = float(getattr(cfg, "FRAG_MIN_S2N", 0.0))
    max_interference = float(getattr(cfg, "FRAG_MAX_INTERFERENCE", 1.0))
    max_pc_missing_peaks = float(getattr(cfg, "FRAG_MAX_PC_MISSING_PEAKS", 100.0))
    min_fit_score = float(getattr(cfg, "FRAG_MIN_FIT_SCORE", 0.0))
    max_unexplained_frac = float(getattr(cfg, "FRAG_MAX_UNEXPLAINED_FRAC", 0.70))
    max_missing_core_frac = float(getattr(cfg, "FRAG_MAX_MISSING_CORE_FRAC", 0.40))
    min_correlation_cfg = getattr(cfg, "FRAG_MIN_CORRELATION", None)
    min_correlation = float(min_correlation_cfg) if min_correlation_cfg is not None else None
    max_chisq_cfg = getattr(cfg, "FRAG_MAX_CHISQ_STAT", None)
    max_chisq_stat = float(max_chisq_cfg) if max_chisq_cfg is not None else None

    checks = [
        make_rule_check("Fragments CSS", state="pass" if float(isodec_css) >= float(cfg.MIN_COSINE) else "fail", value_text=format_num(isodec_css, 3), target_text=f">= {float(cfg.MIN_COSINE):.3f}"),
        make_rule_check("Anchor ppm", state="pass" if abs(float(ppm)) <= float(max_anchor_abs_ppm) else "fail", value_text=f"{ppm:+.2f} ppm", target_text=f"within +/-{float(max_anchor_abs_ppm):.2f} ppm"),
        make_rule_check("Local matches", state="pass" if len(local_matches) >= int(min_matched_peaks) else "fail", value_text=str(int(len(local_matches))), target_text=f">= {int(min_matched_peaks)}"),
        make_rule_check("Coverage", state="pass" if float(comp['coverage']) >= float(min_coverage) else "fail", value_text=format_num(comp["coverage"], 3), target_text=f">= {float(min_coverage):.3f}"),
        make_rule_check("S/N", state="pass" if float(quality['s2n']) >= float(min_s2n) else "fail", value_text=format_num(quality["s2n"], 2), target_text=f">= {float(min_s2n):.2f}"),
        make_rule_check("Interference", state="pass" if float(quality['interference']) <= float(max_interference) else "fail", value_text=format_num(quality["interference"], 3), target_text=f"<= {float(max_interference):.3f}"),
        make_rule_check("Missing peaks", state="pass" if float(quality['pc_missing_peaks']) <= float(max_pc_missing_peaks) else "fail", value_text=format_num(quality["pc_missing_peaks"], 1), target_text=f"<= {float(max_pc_missing_peaks):.1f}%"),
        make_rule_check("Fit score", state="pass" if float(quality['fit_score']) >= float(min_fit_score) else "fail", value_text=format_num(quality["fit_score"], 3), target_text=f">= {float(min_fit_score):.3f}"),
        make_rule_check("Unexplained", state="pass" if float(quality['unexplained_fraction']) <= float(max_unexplained_frac) else "fail", value_text=format_num(quality["unexplained_fraction"], 3), target_text=f"<= {float(max_unexplained_frac):.3f}"),
        make_rule_check("Missing core", state="pass" if float(quality['missing_core_fraction']) <= float(max_missing_core_frac) else "fail", value_text=format_num(quality["missing_core_fraction"], 3), target_text=f"<= {float(max_missing_core_frac):.3f}"),
    ]

    if len(local_matches) >= 2 and np.isfinite(comp["ppm_rmse"]):
        checks.append(
            make_rule_check(
                "PPM RMSE",
                state="pass" if float(comp["ppm_rmse"]) <= float(max_residual_rmse_ppm) else "fail",
                value_text=format_num(comp["ppm_rmse"], 2),
                target_text=f"<= {float(max_residual_rmse_ppm):.2f}",
            )
        )
    if max_mass_error_std_ppm is not None and np.isfinite(float(quality["mass_error_std"])):
        checks.append(
            make_rule_check(
                "Mass error std",
                state="pass" if float(quality["mass_error_std"]) <= float(max_mass_error_std_ppm) else "fail",
                value_text=format_num(quality["mass_error_std"], 2),
                target_text=f"<= {float(max_mass_error_std_ppm):.2f}",
            )
        )
    if min_correlation is not None and np.isfinite(float(quality["correlation_coefficient"])):
        checks.append(
            make_rule_check(
                "Correlation",
                state="pass" if float(quality["correlation_coefficient"]) >= float(min_correlation) else "fail",
                value_text=format_num(quality["correlation_coefficient"], 3),
                target_text=f">= {float(min_correlation):.3f}",
            )
        )
    if max_chisq_stat is not None and np.isfinite(float(quality["chisq_stat"])):
        checks.append(
            make_rule_check(
                "Chi-square",
                state="pass" if float(quality["chisq_stat"]) <= float(max_chisq_stat) else "fail",
                value_text=format_num(quality["chisq_stat"], 3),
                target_text=f"<= {float(max_chisq_stat):.3f}",
            )
        )

    unmet = [check["label"] for check in checks if check["state"] == "fail"]
    accepted = len(unmet) == 0

    trace.update(
        {
            "stage": "accepted" if accepted else "quality_gates",
            "accepted": bool(accepted),
            "reason": "accepted_candidate" if accepted else "fragments_quality_gates_failed",
            "ruleChecks": checks,
            "unmetChecks": unmet,
            "css": float(isodec_css),
            "anchorPpm": float(ppm),
            "score": float(quality["evidence_score"]),
            "coverage": float(comp["coverage"]),
            "matchCount": int(len(local_matches)),
            "obsIdx": int(obs_idx),
            "obsMz": float(obs_mz),
            "obsInt": float(obs_int),
            "obsRelInt": float(obs_rel_int),
        }
    )
    return trace


def build_fragments_trace(row: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    re_num = int(str(row["re"]).replace("RE", ""))
    bundle = get_fragments_scan_bundle(re_num)
    ion_spec = manual_label_to_ion_spec(str(row["ion_label"]))[0]
    ion_type, frag_len, loss_formula, loss_count, charge = parse_fragment_spec(ion_spec)
    charge = int(charge or row["charge"])
    exact_loss_suffix = neutral_loss_label(int(loss_count), loss_formula) if loss_formula and loss_count else ""

    fragments_result = bundle["fragmentsResult"]
    base_raw = [
        match
        for match in fragments_result.get("matches", [])
        if str(match.get("ion_type") or "").lower() == str(ion_type).lower()
        and int(match.get("frag_len") or 0) == int(frag_len)
        and int(match.get("charge") or 0) == int(charge)
    ]
    base_best = [
        match
        for match in fragments_result.get("best", [])
        if str(match.get("ion_type") or "").lower() == str(ion_type).lower()
        and int(match.get("frag_len") or 0) == int(frag_len)
        and int(match.get("charge") or 0) == int(charge)
    ]

    base_raw.sort(key=lambda entry: (
        float(entry.get("score", entry.get("css", 0.0))),
        float(entry.get("coverage", 0.0)),
        float(entry.get("css", 0.0)),
        float(entry.get("obs_int", 0.0)),
    ), reverse=True)

    if base_best:
        selected = base_best[0]
        return {
            "status": "selected",
            "statusLabel": "selected in fragments",
            "reason": "base ion reached final fragments selection",
            "candidateLabel": selected.get("frag_id", ""),
            "rawCount": len(base_raw),
            "bestCount": len(base_best),
            "unmetChecks": [],
            "ruleChecks": [
                make_rule_check("Final selection", state="pass", value_text="yes"),
                make_rule_check("Evidence score", state="info", value_text=format_num(selected.get("score"), 3)),
                make_rule_check("Fragments CSS", state="info", value_text=format_num(selected.get("css"), 3)),
            ],
        }

    if base_raw:
        candidate = base_raw[0]
        conflict = None
        for winner in fragments_result.get("best", []):
            if int(winner.get("obs_idx", -1)) == int(candidate.get("obs_idx", -2)):
                conflict = winner
                break
        unmet = ["Competing candidate"]
        checks = [
            make_rule_check("Raw candidate", state="pass", value_text="yes", note=f"{len(base_raw)} raw match(es) for this base ion"),
            make_rule_check("Evidence score", state="info", value_text=format_num(candidate.get("score"), 3)),
            make_rule_check("Fragments CSS", state="info", value_text=format_num(candidate.get("css"), 3)),
            make_rule_check("Coverage", state="info", value_text=format_num(candidate.get("coverage"), 3)),
        ]
        if conflict is not None:
            checks.append(
                make_rule_check(
                    "Competing candidate",
                    state="fail",
                    value_text=str(conflict.get("frag_id") or "winner"),
                    note=f"same obs_idx won with score {format_num(conflict.get('score'), 3)}",
                )
            )
            reason = f"raw candidate lost on the same observed peak to {conflict.get('frag_id')}"
        else:
            reason = "raw candidate exists but was removed before final best list"
        return {
            "status": "raw_only",
            "statusLabel": "raw candidate only",
            "reason": reason,
            "candidateLabel": str(candidate.get("frag_id") or ""),
            "rawCount": len(base_raw),
            "bestCount": 0,
            "unmetChecks": unmet,
            "ruleChecks": checks,
        }

    residues = bundle["residues"]
    spectrum = bundle["spectrum"]
    isodec_config = bundle["isodecConfig"]
    noise_model = bundle["noiseModel"]

    with _override_cfg(bundle["overrides"]):
        frag_name, target_comp = ion_composition_from_sequence(residues, ion_type, frag_len, amidated=cfg.AMIDATED)
        variant_rows = get_disulfide_logic(ion_type, frag_len, len(residues)) or [("", None)]
        traces = []
        for variant_suffix, shift in variant_rows:
            try:
                variant_comp = target_comp + shift if shift is not None else target_comp
            except Exception:
                continue
            trace = execute_hybrid_strategy(
                _evaluate_fragment_variant_trace,
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

    best_trace = max(traces, key=_trace_sort_key) if traces else {
        "stage": "no_theory",
        "accepted": False,
        "reason": "no_variant_trace",
        "candidateLabel": f"{ion_type}{frag_len}^{int(charge)}+",
        "ruleChecks": [
            make_rule_check("Variant trace", state="fail", value_text="n/a", note="no fragment variant could be traced"),
        ],
        "unmetChecks": ["Variant trace"],
    }

    note_parts = []
    if loss_formula and loss_count:
        note_parts.append(
            f"manual card is {ion_type}{frag_len}{exact_loss_suffix}^{int(charge)}+, but fragments never reached a raw base candidate, so loss variants were not evaluated"
        )
    note = " | ".join(note_parts)
    reason = str(best_trace.get("reason") or "base candidate rejected before entering raw matches")
    if note:
        reason = f"{reason}; {note}"

    return {
        "status": "not_in_raw",
        "statusLabel": "not in raw candidates",
        "reason": reason,
        "candidateLabel": str(best_trace.get("candidateLabel") or f"{ion_type}{frag_len}^{int(charge)}+"),
        "rawCount": 0,
        "bestCount": 0,
        "unmetChecks": list(best_trace.get("unmetChecks") or []),
        "ruleChecks": list(best_trace.get("ruleChecks") or []),
        "stage": best_trace.get("stage"),
        "bestModel": best_trace.get("bestModel"),
    }


def normalize_manual_label(label: str) -> str:
    normalized = label.strip()
    normalized = normalized.replace("(H2O)", "H2O")
    normalized = normalized.replace("(NH3)", "NH3")
    normalized = normalized.replace("(CO2)", "CO2")
    normalized = normalized.replace("(CO)", "CO")
    return normalized


def manual_label_to_ion_spec(label: str) -> tuple[str, int, int]:
    normalized = normalize_manual_label(label)
    match = re.match(r"^([A-Za-z0-9+\-]+)\s+(\d+)\+$", normalized)
    if not match:
        raise ValueError(f"Unsupported manual ion label: {label}")
    core = match.group(1)
    charge = int(match.group(2))
    base = re.match(r"^([bc])(\d+)", core)
    if not base:
        raise ValueError(f"Unsupported ion series in manual ion label: {label}")
    ion_type = base.group(1)
    pos = int(base.group(2))
    return f"{core}^{charge}+", pos, charge


def _prediction_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(item.get("score", item.get("css", 0.0)) or 0.0),
        float(item.get("coverage", 0.0) or 0.0),
        float(item.get("css", 0.0) or 0.0),
        float(item.get("obs_rel_int", 0.0) or 0.0),
        float(item.get("obs_int", 0.0) or 0.0),
    )


def summarize_manual_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = []
    ion_scores = []
    gof_scores = []
    peaks_matched = []

    matched_rows = 0
    selected_rows = 0
    for row in rows:
        matched = parse_float(row.get("Matched")) == 1.0
        selected = parse_float(row.get("Selected")) == 1.0
        matched_rows += int(matched)
        selected_rows += int(selected)
        ion_score = parse_float(row.get("IonScore"))
        gof = parse_float(row.get("Gof Confidence"))
        peaks = parse_float(row.get("Peaks Matched"))
        if ion_score is not None:
            ion_scores.append(ion_score)
        if gof is not None:
            gof_scores.append(gof)
        if peaks is not None:
            peaks_matched.append(peaks)
        labels.append(
            {
                "label": (row.get("Name") or "").strip() or "manual row",
                "matched": matched,
                "selected": selected,
            }
        )

    if not rows:
        status_label = "absent from manual annotation"
    elif selected_rows > 0:
        status_label = "manual rows exist but none reached Matched=1"
    else:
        status_label = "manual rows exist but all stayed unmatched"

    return {
        "rowCount": len(rows),
        "matchedRows": matched_rows,
        "selectedRows": selected_rows,
        "bestIonScore": max(ion_scores) if ion_scores else None,
        "bestGofConfidence": max(gof_scores) if gof_scores else None,
        "bestPeaksMatched": max(peaks_matched) if peaks_matched else None,
        "labels": labels,
        "statusLabel": status_label,
    }


def build_selected_fragments_evidence(row: dict[str, Any]) -> dict[str, Any]:
    css = parse_float(row.get("fragments_css"))
    coverage = parse_float(row.get("fragments_coverage"))
    score = parse_float(row.get("fragments_score"))
    match_count = parse_float(row.get("fragments_match_count"))
    unexplained = parse_float(row.get("fragments_unexplained"))
    missing_core = parse_float(row.get("fragments_missing_core"))
    s2n = parse_float(row.get("fragments_s2n"))
    interference = parse_float(row.get("fragments_interference"))
    ppm_rmse = parse_float(row.get("fragments_ppm_rmse"))
    raw_cos = parse_float(row.get("fragments_rawcos"))

    min_css = float(getattr(cfg, "MIN_COSINE", 0.70))
    min_coverage = float(getattr(cfg, "FRAG_MIN_COVERAGE", 0.25))
    min_matches = int(getattr(cfg, "FRAG_MIN_MATCHED_PEAKS", 2))
    max_unexplained = float(getattr(cfg, "FRAG_MAX_UNEXPLAINED_FRAC", 0.70))
    max_missing_core = float(getattr(cfg, "FRAG_MAX_MISSING_CORE_FRAC", 0.40))
    min_s2n = float(getattr(cfg, "FRAG_MIN_S2N", 0.0))
    max_interference = float(getattr(cfg, "FRAG_MAX_INTERFERENCE", 1.0))
    max_ppm_rmse_cfg = getattr(cfg, "FRAG_MAX_RESIDUAL_RMSE_PPM", None)
    max_ppm_rmse = float(max_ppm_rmse_cfg) if max_ppm_rmse_cfg is not None else float(cfg.MATCH_TOL_PPM)

    checks = [
        make_rule_check(
            "Fragments CSS",
            state="pass" if css is not None and css >= min_css else "fail",
            value_text=format_num(css, 3),
            target_text=f">= {min_css:.3f}",
        ),
        make_rule_check(
            "Coverage",
            state="pass" if coverage is not None and coverage >= min_coverage else "fail",
            value_text=format_num(coverage, 3),
            target_text=f">= {min_coverage:.3f}",
        ),
        make_rule_check(
            "Local matches",
            state="pass" if match_count is not None and int(match_count) >= min_matches else "fail",
            value_text="n/a" if match_count is None else str(int(match_count)),
            target_text=f">= {min_matches}",
        ),
        make_rule_check(
            "Unexplained",
            state="pass" if unexplained is not None and unexplained <= max_unexplained else "fail",
            value_text=format_num(unexplained, 3),
            target_text=f"<= {max_unexplained:.3f}",
        ),
        make_rule_check(
            "Missing core",
            state="pass" if missing_core is not None and missing_core <= max_missing_core else "fail",
            value_text=format_num(missing_core, 3),
            target_text=f"<= {max_missing_core:.3f}",
        ),
        make_rule_check(
            "S/N",
            state="pass" if s2n is not None and s2n >= min_s2n else "fail",
            value_text=format_num(s2n, 2),
            target_text=f">= {min_s2n:.2f}",
        ),
        make_rule_check(
            "Interference",
            state="pass" if interference is not None and interference <= max_interference else "fail",
            value_text=format_num(interference, 3),
            target_text=f"<= {max_interference:.3f}",
        ),
        make_rule_check(
            "PPM RMSE",
            state="pass" if ppm_rmse is not None and ppm_rmse <= max_ppm_rmse else "fail",
            value_text=format_num(ppm_rmse, 2),
            target_text=f"<= {max_ppm_rmse:.2f}",
        ),
        make_rule_check("Evidence score", state="info", value_text=format_num(score, 3)),
        make_rule_check("Raw cosine", state="info", value_text=format_num(raw_cos, 3)),
    ]
    unmet = [check["label"] for check in checks if check["state"] == "fail"]
    return {
        "status": "selected",
        "statusLabel": "selected by fragments",
        "reason": "algorithm included this ion in final fragments output, but manual truth has no Matched=1 row",
        "candidateLabel": str(row.get("algorithm_frag_id") or row.get("ion_spec") or ""),
        "rawCount": 1,
        "bestCount": 1,
        "unmetChecks": unmet,
        "ruleChecks": checks,
    }


def compute_algorithm_only_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scan_pat = re.compile(r"ECDRE(\d+)-ion-state_man\.csv$")

    for ann_path in sorted(ANNOTATION_DIR.glob("ECDRE*-ion-state_man.csv"), key=lambda p: int(scan_pat.search(p.name).group(1))):
        re_num = int(scan_pat.search(ann_path.name).group(1))
        spec_path = SPECTRUM_DIR / f"ECDRE{re_num}.txt"

        manual_true: set[tuple[str, int, int]] = set()
        manual_all: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        with ann_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ion_type = (row.get("Base Type") or row.get("Type") or "").strip().lower()
                if ion_type not in {"b", "c"}:
                    continue
                try:
                    pos = int(float((row.get("Pos") or "").strip()))
                    charge = int(float((row.get("Charge") or "").strip()))
                except Exception:
                    continue
                matched = parse_float(row.get("Matched")) == 1.0
                key = (ion_type, pos, charge)
                manual_all.setdefault(key, []).append(row)
                if matched:
                    manual_true.add(key)

        req = FragmentsRunRequest(
            filepath=str(spec_path.resolve()),
            scan=1,
            peptide=PEPTIDE,
            ion_types=["b", "c"],
            frag_min_charge=FRAG_MIN_CHARGE,
            frag_max_charge=FRAG_MAX_CHARGE,
            copies=COPIES,
            amidated=AMIDATED,
            disulfide_bonds=DISULFIDE_BONDS,
            disulfide_map=DISULFIDE_MAP,
        )
        result = quiet_call(_run_fragments_impl, req)
        predicted_best: dict[tuple[str, int, int], dict[str, Any]] = {}
        for item in result.get("fragments", []):
            ion_type = str(item.get("ion_type") or "").lower()
            if ion_type not in {"b", "c"}:
                continue
            pos = int(item.get("frag_len") or 0)
            charge = int(item.get("charge") or 0)
            key = (ion_type, pos, charge)
            if key in manual_true:
                continue
            current = predicted_best.get(key)
            if current is None or _prediction_sort_key(item) > _prediction_sort_key(current):
                predicted_best[key] = item

        for key, item in sorted(predicted_best.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
            ion_type, pos, charge = key
            manual_context = summarize_manual_context(manual_all.get(key, []))
            frag_id = str(item.get("frag_id") or f"{ion_type}{pos}")
            ion_spec = f"{frag_id}^{charge}+"
            rows.append(
                {
                    "re": f"RE{re_num}",
                    "ion_label": f"{ion_type}{pos} {charge}+",
                    "ion_spec": ion_spec,
                    "ion_type": ion_type,
                    "pos": pos,
                    "charge": charge,
                    "algorithm_frag_id": frag_id,
                    "algorithm_label": str(item.get("label") or ion_spec),
                    "algorithm_variant_suffix": str(item.get("variant_suffix") or ""),
                    "fragments_score": item.get("score", ""),
                    "fragments_css": item.get("css", ""),
                    "fragments_coverage": item.get("coverage", ""),
                    "fragments_obs_mz": item.get("obs_mz", ""),
                    "fragments_obs_int": item.get("obs_int", ""),
                    "fragments_obs_rel_int": item.get("obs_rel_int", ""),
                    "fragments_match_count": item.get("match_count", ""),
                    "fragments_unexplained": item.get("unexplained_fraction", ""),
                    "fragments_missing_core": item.get("missing_core_fraction", ""),
                    "fragments_interference": item.get("interference", ""),
                    "fragments_s2n": item.get("s2n", ""),
                    "fragments_ppm_rmse": item.get("ppm_rmse", ""),
                    "fragments_rawcos": item.get("rawcos", ""),
                    "manual_row_count": manual_context["rowCount"],
                    "manual_matched_rows": manual_context["matchedRows"],
                    "manual_selected_rows": manual_context["selectedRows"],
                    "manual_best_ion_score": "" if manual_context["bestIonScore"] is None else manual_context["bestIonScore"],
                    "manual_best_gof_confidence": "" if manual_context["bestGofConfidence"] is None else manual_context["bestGofConfidence"],
                    "manual_best_peaks_matched": "" if manual_context["bestPeaksMatched"] is None else manual_context["bestPeaksMatched"],
                    "manual_status_label": manual_context["statusLabel"],
                    "manual_labels_json": json.dumps(manual_context["labels"], ensure_ascii=False),
                }
            )

    ALGORITHM_ONLY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ALGORITHM_ONLY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "re",
                "ion_label",
                "ion_spec",
                "ion_type",
                "pos",
                "charge",
                "algorithm_frag_id",
                "algorithm_label",
                "algorithm_variant_suffix",
                "fragments_score",
                "fragments_css",
                "fragments_coverage",
                "fragments_obs_mz",
                "fragments_obs_int",
                "fragments_obs_rel_int",
                "fragments_match_count",
                "fragments_unexplained",
                "fragments_missing_core",
                "fragments_interference",
                "fragments_s2n",
                "fragments_ppm_rmse",
                "fragments_rawcos",
                "manual_row_count",
                "manual_matched_rows",
                "manual_selected_rows",
                "manual_best_ion_score",
                "manual_best_gof_confidence",
                "manual_best_peaks_matched",
                "manual_status_label",
                "manual_labels_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_algorithm_only_rows() -> list[dict[str, Any]]:
    if not ALGORITHM_ONLY_CSV.exists():
        return compute_algorithm_only_rows()
    with ALGORITHM_ONLY_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def render_error_figure(out_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.8), dpi=170)
    fig.patch.set_facecolor(UI_PAPER)
    ax.set_facecolor(UI_PANEL)
    ax.axis("off")
    ax.text(0.04, 0.80, title, fontsize=20, fontweight="bold", color=UI_BLUE, transform=ax.transAxes)
    ax.text(0.04, 0.58, message, fontsize=12, color=UI_RED, transform=ax.transAxes, wrap=True)
    ax.text(0.04, 0.20, "Diagnose render fallback", fontsize=10, color=UI_MUTED, transform=ax.transAxes)
    ax.text(
        0.96,
        0.84,
        "render error",
        fontsize=10,
        color=UI_RED,
        ha="right",
        va="center",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fef2f2", "edgecolor": "#fecaca"},
    )
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_diagnose_figure(out_path: Path, item: dict[str, Any], best: dict[str, Any], spectrum: dict[str, Any]) -> None:
    spectrum_mz = np.asarray(spectrum.get("mz") or [], dtype=float)
    spectrum_int = np.asarray(spectrum.get("intensity") or [], dtype=float)
    theory_mz = np.asarray(best.get("theory_mz") or [], dtype=float)
    theory_int = np.asarray(best.get("theory_int") or [], dtype=float)

    anchors = [parse_float(best.get("anchor_theory_mz")), parse_float(best.get("anchor_obs_mz"))]
    anchors = [value for value in anchors if value is not None]
    if theory_mz.size:
        xmin = float(np.min(theory_mz))
        xmax = float(np.max(theory_mz))
    elif anchors:
        xmin = min(anchors)
        xmax = max(anchors)
    else:
        xmin = float(np.min(spectrum_mz)) if spectrum_mz.size else 0.0
        xmax = float(np.max(spectrum_mz)) if spectrum_mz.size else 1.0
    margin = max(1.5, (xmax - xmin) * 0.55)
    xmin -= margin
    xmax += margin

    if spectrum_mz.size:
        mask = (spectrum_mz >= xmin) & (spectrum_mz <= xmax)
        exp_x = spectrum_mz[mask]
        exp_y = spectrum_int[mask]
        if exp_x.size == 0:
            exp_x = spectrum_mz
            exp_y = spectrum_int
    else:
        exp_x = np.array([], dtype=float)
        exp_y = np.array([], dtype=float)

    max_exp = float(np.max(exp_y)) if exp_y.size else 1.0
    max_theory = float(np.max(theory_int)) if theory_int.size else max_exp * 0.7

    fig, ax = plt.subplots(figsize=(10.8, 4.9), dpi=170)
    fig.patch.set_facecolor(UI_PAPER)
    ax.set_facecolor(UI_PANEL)

    if exp_x.size:
        ax.vlines(exp_x, 0.0, exp_y, color=UI_INK, linewidth=1.0, alpha=0.96, zorder=3)
    if theory_mz.size:
        ax.vlines(theory_mz, 0.0, -theory_int, color=UI_PURPLE, linewidth=1.35, alpha=0.98, zorder=4)

    anchor_theory = parse_float(best.get("anchor_theory_mz"))
    anchor_obs = parse_float(best.get("anchor_obs_mz"))
    if anchor_theory is not None:
        ax.axvline(anchor_theory, color=UI_PURPLE, linewidth=1.0, linestyle="--", alpha=0.55, zorder=2)
    if anchor_obs is not None:
        ax.axvline(
            anchor_obs,
            color=UI_GREEN if bool(best.get("ok")) else UI_RED,
            linewidth=1.1,
            linestyle=":",
            alpha=0.72,
            zorder=2,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-1.2 * max_theory, 1.2 * max_exp)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UI_GRID)
    ax.spines["bottom"].set_color(UI_GRID)
    ax.tick_params(axis="both", colors=UI_MUTED, labelsize=10)
    ax.set_xlabel("m/z", fontsize=11, color=UI_BLUE)
    ax.set_ylabel("Intensity", fontsize=11, color=UI_BLUE)
    ax.grid(axis="both", color=UI_GRID, linewidth=0.8)
    ax.axhline(0.0, color=UI_INK, linewidth=1.0, zorder=1)

    reason = str(best.get("reason") or "n/a")
    css = format_num(parse_float(best.get("isodec_css")), 3)
    raw_cos = format_num(parse_float(best.get("raw_cosine")), 3)
    anchor_ppm = format_num(parse_float(best.get("anchor_ppm")), 2)
    rule_checks, unmet = build_diagnose_rule_checks(best)
    unmet_text = ", ".join(unmet) if unmet else "none"
    issue_color = UI_RED if unmet else UI_GREEN
    issue_fill = "#fef2f2" if unmet else "#f0fdf4"

    title = f"{item['re']} | {item['ion_label']}"
    subtitle = f"css {css} | raw {raw_cos} | anchor ppm {anchor_ppm} | unmet {unmet_text}"
    ax.text(0.015, 1.04, title, transform=ax.transAxes, fontsize=15, fontweight="bold", color=UI_BLUE)
    ax.text(0.015, 0.985, subtitle, transform=ax.transAxes, fontsize=9.5, color=UI_MUTED)
    ax.text(
        0.985,
        1.035,
        str(best.get("label") or item["ion_spec"]),
        transform=ax.transAxes,
        fontsize=9,
        color=UI_PURPLE,
        ha="right",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f3ff", "edgecolor": "#ddd6fe"},
    )
    ax.text(
        0.985,
        0.98,
        f"reason {reason}",
        transform=ax.transAxes,
        fontsize=9,
        color=issue_color,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": issue_fill, "edgecolor": issue_color},
    )

    legend_handles = [
        plt.Line2D([0], [0], color=UI_INK, lw=1.2),
        plt.Line2D([0], [0], color=UI_PURPLE, lw=1.4),
    ]
    legend_labels = ["Experimental", "Diagnose theory"]
    if anchor_obs is not None:
        legend_handles.append(plt.Line2D([0], [0], color=issue_color, lw=1.1, linestyle=":"))
        legend_labels.append("Observed anchor")
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.16),
        frameon=False,
        fontsize=9,
        ncol=min(3, len(legend_labels)),
        labelcolor=UI_MUTED,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_item(row: dict[str, Any], recurring_counts: Counter[tuple[str, int, int]]) -> dict[str, Any]:
    ion_spec = str(row["ion_spec"])
    ion_type = str(row["ion_type"]).lower()
    pos = int(row["pos"])
    charge = int(row["charge"])
    base_ion_spec = f"{ion_type}{pos}^{charge}+"
    re_code = str(row["re"])
    re_num = int(re_code.replace("RE", ""))
    spectrum_path = SPECTRUM_DIR / f"ECDRE{re_num}.txt"

    diagnose_req = DiagnoseRunRequest(
        filepath=str(spectrum_path.resolve()),
        scan=1,
        peptide=PEPTIDE,
        ion_spec=base_ion_spec,
        h_transfer=0,
        frag_min_charge=FRAG_MIN_CHARGE,
        frag_max_charge=FRAG_MAX_CHARGE,
        copies=COPIES,
        amidated=AMIDATED,
        disulfide_bonds=DISULFIDE_BONDS,
        disulfide_map=DISULFIDE_MAP,
    )

    best: dict[str, Any]
    diagnose_result: dict[str, Any]
    slug = slugify(f"{re_code}-{row['ion_label']}")
    image_rel = f"assets/{slug}.png"
    image_path = ASSETS_DIR / f"{slug}.png"

    try:
        diagnose_result = quiet_call(_run_diagnose_impl, diagnose_req)
        results = diagnose_result.get("results") or []
        algorithm_frag_id = str(row.get("algorithm_frag_id") or "")
        match_prefix = f"{algorithm_frag_id}^{charge}+"
        matched_result = next(
            (entry for entry in results if str(entry.get("label") or "") == match_prefix),
            None,
        )
        best = matched_result or diagnose_result.get("best") or (results[0] if results else {})
        if best:
            render_diagnose_figure(image_path, {**row, "ion_spec": ion_spec}, best, diagnose_result.get("spectrum") or {})
        else:
            render_error_figure(image_path, f"{re_code} | {row['ion_label']}", "No diagnose result returned.")
    except Exception as exc:
        diagnose_result = {}
        best = {"ok": False, "reason": f"diagnose_error: {exc}"}
        render_error_figure(image_path, f"{re_code} | {row['ion_label']}", str(exc))

    rule_checks, unmet_checks = build_diagnose_rule_checks(best)
    fragments_trace = build_selected_fragments_evidence(row)
    manual_labels = []
    try:
        parsed_labels = json.loads(str(row.get("manual_labels_json") or "[]"))
        if isinstance(parsed_labels, list):
            manual_labels = parsed_labels
    except Exception:
        manual_labels = []

    return {
        "id": slug,
        "re": re_code,
        "reIndex": re_num,
        "ionLabel": row["ion_label"],
        "ionSpec": ion_spec,
        "ionType": ion_type,
        "pos": pos,
        "charge": charge,
        "recurringCount": recurring_counts[(ion_type, pos, charge)],
        "image": image_rel,
        "manual": {
            "rowCount": int(float(row.get("manual_row_count") or 0)),
            "matchedRows": int(float(row.get("manual_matched_rows") or 0)),
            "selectedRows": int(float(row.get("manual_selected_rows") or 0)),
            "bestIonScore": parse_float(row.get("manual_best_ion_score")),
            "bestGofConfidence": parse_float(row.get("manual_best_gof_confidence")),
            "bestPeaksMatched": parse_float(row.get("manual_best_peaks_matched")),
            "statusLabel": str(row.get("manual_status_label") or "no manual truth row"),
            "labels": manual_labels,
        },
        "algorithm": {
            "fragId": str(row.get("algorithm_frag_id") or ""),
            "label": str(row.get("algorithm_label") or ion_spec),
            "variantSuffix": str(row.get("algorithm_variant_suffix") or ""),
            "score": parse_float(row.get("fragments_score")),
            "css": parse_float(row.get("fragments_css")),
            "coverage": parse_float(row.get("fragments_coverage")),
            "obsMz": parse_float(row.get("fragments_obs_mz")),
            "obsInt": parse_float(row.get("fragments_obs_int")),
            "obsRelInt": parse_float(row.get("fragments_obs_rel_int")),
            "matchCount": parse_float(row.get("fragments_match_count")),
            "unexplained": parse_float(row.get("fragments_unexplained")),
            "missingCore": parse_float(row.get("fragments_missing_core")),
            "interference": parse_float(row.get("fragments_interference")),
            "s2n": parse_float(row.get("fragments_s2n")),
            "ppmRmse": parse_float(row.get("fragments_ppm_rmse")),
            "rawCos": parse_float(row.get("fragments_rawcos")),
        },
        "diagnose": {
            "ok": bool(best.get("ok", False)),
            "reason": str(best.get("reason") or "n/a"),
            "label": str(best.get("label") or ion_spec),
            "variantSuffix": str(best.get("variant_suffix") or ""),
            "variantType": str(best.get("variant_type") or ""),
            "hTransfer": 0,
            "isodecAccepted": bool(best.get("isodec_accepted", False)),
            "isodecCss": parse_float(best.get("isodec_css")),
            "rawCosine": parse_float(best.get("raw_cosine")),
            "anchorTheoryMz": parse_float(best.get("anchor_theory_mz")),
            "anchorObsMz": parse_float(best.get("anchor_obs_mz")),
            "anchorPpm": parse_float(best.get("anchor_ppm")),
            "anchorWithinPpm": bool(best.get("anchor_within_ppm", False)),
            "ruleChecks": rule_checks,
            "unmetChecks": unmet_checks,
            "isodecDetail": {
                "matchedPeaks": int(best.get("isodec_detail", {}).get("matched_peaks_n", 0) or 0)
                if isinstance(best.get("isodec_detail"), dict)
                else None,
                "minPeaks": int(best.get("isodec_detail", {}).get("minpeaks_effective", 0) or 0)
                if isinstance(best.get("isodec_detail"), dict)
                else None,
                "areaCovered": parse_float(best.get("isodec_detail", {}).get("areacovered"))
                if isinstance(best.get("isodec_detail"), dict)
                else None,
                "minAreaCovered": parse_float(best.get("isodec_detail", {}).get("minareacovered"))
                if isinstance(best.get("isodec_detail"), dict)
                else None,
                "topPeaks": best.get("isodec_detail", {}).get("topthree")
                if isinstance(best.get("isodec_detail"), dict)
                else None,
            },
        },
        "fragments": fragments_trace,
}


def build_report(items: list[dict[str, Any]]) -> dict[str, Any]:
    recurring = Counter((item["ionType"], int(item["pos"]), int(item["charge"])) for item in items)
    ok_count = sum(1 for item in items if item["diagnose"]["ok"])
    fail_count = len(items) - ok_count
    reason_counts = Counter(item["diagnose"]["reason"] for item in items if item["diagnose"]["reason"] != "ok")
    unmet_check_counts = Counter(
        label
        for item in items
        for label in item["diagnose"].get("unmetChecks", [])
    )
    by_re = Counter(item["re"] for item in items)
    by_ion_type = Counter(item["ionType"] for item in items)
    unique_base = len(recurring)
    full_rule_count = sum(1 for item in items if not item["diagnose"].get("unmetChecks"))
    manual_overlap_counts = Counter()
    for item in items:
        manual = item.get("manual", {})
        row_count = int(manual.get("rowCount") or 0)
        selected_rows = int(manual.get("selectedRows") or 0)
        if row_count == 0:
            manual_overlap_counts["no manual row"] += 1
        elif selected_rows > 0:
            manual_overlap_counts["manual selected but unmatched"] += 1
        else:
            manual_overlap_counts["manual rows unmatched"] += 1

    recurring_rows = []
    for (ion_type, pos, charge), count in recurring.most_common(12):
        recurring_rows.append(
            {
                "label": f"{ion_type}{pos}^{charge}+",
                "count": count,
            }
        )

    reason_rows = [{"label": key, "count": count} for key, count in reason_counts.most_common(10)]
    failed_check_rows = [{"label": key, "count": count} for key, count in unmet_check_counts.most_common(10)]
    manual_overlap_rows = [{"label": key, "count": count} for key, count in manual_overlap_counts.most_common(10)]
    re_rows = [{"label": key, "count": by_re[key]} for key in sorted(by_re, key=lambda value: int(value.replace("RE", "")))]
    ion_rows = [{"label": key, "count": by_ion_type[key]} for key in sorted(by_ion_type)]

    return {
        "generatedAt": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "source": {
            "algorithmOnlyCsv": str(ALGORITHM_ONLY_CSV.relative_to(ROOT)),
            "annotationDir": str(ANNOTATION_DIR.relative_to(ROOT)),
        },
        "summary": {
            "totalRows": len(items),
            "uniqueBasePredictions": unique_base,
            "okCount": ok_count,
            "failCount": fail_count,
            "fullRuleCount": full_rule_count,
            "unmetRuleCount": len(items) - full_rule_count,
            "byRe": re_rows,
            "byIonType": ion_rows,
            "recurring": recurring_rows,
            "reasons": reason_rows,
            "failedChecks": failed_check_rows,
            "manualOverlap": manual_overlap_rows,
        },
        "items": sorted(items, key=lambda item: (item["reIndex"], item["ionType"], item["pos"], item["charge"], item["ionLabel"])),
    }


def write_data_js(report: dict[str, Any]) -> None:
    payload = json.dumps(report, separators=(",", ":"), ensure_ascii=False)
    target = REPORT_DIR / "data.js"
    target.write_text(f"window.Q10R_ALGO_ONLY_REPORT = {payload};\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_algorithm_only_rows()

    recurring = Counter((str(row["ion_type"]).lower(), int(row["pos"]), int(row["charge"])) for row in rows)
    items = [build_item(row, recurring) for row in rows]
    report = build_report(items)
    write_data_js(report)
    print(f"Generated report: {REPORT_DIR}")
    print(f"Images: {len(items)}")


if __name__ == "__main__":
    main()
