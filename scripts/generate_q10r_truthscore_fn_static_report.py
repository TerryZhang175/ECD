from __future__ import annotations

import csv
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import personalized_config as cfg
import scripts.generate_q10r_missed_static_report as base
from ecd_api import FragmentsRunRequest
from personalized_match import get_local_centroids_window, isodec_css_and_accept, nearest_peak_index, within_ppm
from personalized_modes import _composite_match_components, _fragment_isodec_detail, _fragment_noise_core_components, _fragment_truth_score, _match_theory_local_monotonic

REPORT_DIR = ROOT / "reports" / "q10r_truthscore_fn_static"
ASSETS_DIR = REPORT_DIR / "assets"
FN_CSV = ROOT / "match_outputs" / "q10r_bc_truthscore_fn_manual_matched.csv"
TEMPLATE_DIR = ROOT / "reports" / "q10r_missed_static"
TRUTH_SCORE_THRESHOLD = 0.80


def _configure_base_paths() -> None:
    base.REPORT_DIR = REPORT_DIR
    base.ASSETS_DIR = ASSETS_DIR
    base.MISSED_CSV = FN_CSV
    base.SCAN_BUNDLE_CACHE.clear()


@contextmanager
def _truth_score_cfg():
    old = (
        cfg.FRAG_TRUTH_SCORE_ENABLE,
        cfg.FRAG_TRUTH_SCORE_THRESHOLD,
        cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING,
    )
    cfg.FRAG_TRUTH_SCORE_ENABLE = True
    cfg.FRAG_TRUTH_SCORE_THRESHOLD = float(TRUTH_SCORE_THRESHOLD)
    cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING = True
    try:
        yield
    finally:
        (
            cfg.FRAG_TRUTH_SCORE_ENABLE,
            cfg.FRAG_TRUTH_SCORE_THRESHOLD,
            cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING,
        ) = old
        base.SCAN_BUNDLE_CACHE.clear()


def _sorted_annotation_paths() -> list[Path]:
    return sorted(
        base.ANNOTATION_DIR.glob("ECDRE*-ion-state_man.csv"),
        key=lambda path: int(path.name.split("ECDRE")[1].split("-")[0]),
    )


def compute_fn_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _truth_score_cfg():
        for ann_path in _sorted_annotation_paths():
            re_num = int(ann_path.name.split("ECDRE")[1].split("-")[0])
            spec_path = base.SPECTRUM_DIR / f"ECDRE{re_num}.txt"

            manual: dict[tuple[str, int, int, str], dict[str, Any]] = {}
            with ann_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    ion_type = str(row.get("Base Type") or row.get("Type") or "").strip().lower()
                    if ion_type not in {"b", "c"}:
                        continue
                    pos = base.parse_float(row.get("Pos"))
                    charge = base.parse_float(row.get("Charge"))
                    matched = base.parse_float(row.get("Matched"))
                    if pos is None or charge is None or matched != 1.0:
                        continue
                    label = str(row.get("Name") or "").strip() or f"{ion_type}{int(pos)} {int(charge)}+"
                    manual[(ion_type, int(pos), int(charge), label)] = row

            req = FragmentsRunRequest(
                filepath=str(spec_path.resolve()),
                scan=1,
                peptide=base.PEPTIDE,
                ion_types=["b", "c"],
                frag_min_charge=base.FRAG_MIN_CHARGE,
                frag_max_charge=base.FRAG_MAX_CHARGE,
                copies=base.COPIES,
                amidated=base.AMIDATED,
                disulfide_bonds=base.DISULFIDE_BONDS,
                disulfide_map=base.DISULFIDE_MAP,
            )
            result = base.quiet_call(base._run_fragments_impl, req)
            predicted = {
                (
                    str(item.get("ion_type") or "").lower(),
                    int(item.get("frag_len") or 0),
                    int(item.get("charge") or 0),
                )
                for item in result.get("fragments", [])
                if str(item.get("ion_type") or "").lower() in {"b", "c"}
            }

            for key, row in sorted(manual.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
                ion_type, pos, charge, label = key
                if (ion_type, pos, charge) in predicted:
                    continue
                rows.append(
                    {
                        "re": f"RE{re_num}",
                        "ion_label": label,
                        "ion_type": ion_type,
                        "pos": pos,
                        "charge": charge,
                        "avg_ppm_error": row.get("Avg PPM Error", ""),
                        "ion_score": row.get("IonScore", ""),
                        "gof_confidence": row.get("Gof Confidence", ""),
                        "peaks_matched": row.get("Peaks Matched", ""),
                    }
                )

    FN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with FN_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "re",
                "ion_label",
                "ion_type",
                "pos",
                "charge",
                "avg_ppm_error",
                "ion_score",
                "gof_confidence",
                "peaks_matched",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_fn_rows() -> list[dict[str, Any]]:
    if not FN_CSV.exists():
        return compute_fn_rows()
    with FN_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


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
    series = base.ion_series(ion_type)
    allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_1H))
    allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_2H))
    truth_score_enabled = bool(getattr(cfg, "FRAG_TRUTH_SCORE_ENABLE", False))
    truth_score_threshold = float(getattr(cfg, "FRAG_TRUTH_SCORE_THRESHOLD", TRUTH_SCORE_THRESHOLD))

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
        dist0 = base.theoretical_isodist_from_comp(variant_comp, z)
    except ValueError:
        trace["stage"] = "no_theory"
        trace["reason"] = "no_theoretical_distribution"
        trace["ruleChecks"] = [
            base.make_rule_check(
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
            base.make_rule_check(
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

    sample_keys, sample_mzs, scale = base.build_sample_axis(
        dists_for_union,
        decimals=6,
        mz_min=cfg.MZ_MIN,
        mz_max=cfg.MZ_MAX,
    )
    if len(sample_mzs) == 0:
        trace["stage"] = "model_setup"
        trace["reason"] = "empty_sample_axis"
        trace["ruleChecks"] = [
            base.make_rule_check(
                "Sample axis",
                state="fail",
                value_text="empty",
                note="no peaks remain inside fragments m/z window",
            )
        ]
        trace["unmetChecks"] = ["Sample axis"]
        return trace

    peak_mz = float(dist0[base.get_anchor_idx(dist0), 0])
    y_obs = base.observed_intensities_isodec(
        spectrum_mz,
        spectrum_int,
        sample_mzs,
        z=int(z),
        match_tol_ppm=match_tol_ppm,
        peak_mz=peak_mz,
    )
    y0 = base.vectorize_dist(dist0, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX)

    neutral_score_union = base.css_similarity(y_obs, y0)
    neutral_score = neutral_score_union
    dist0_neutral = dist0
    if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
        mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
        mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
        dist0_neutral = dist0[(dist0[:, 0] >= mz_min) & (dist0[:, 0] <= mz_max)]
    if dist0_neutral.size:
        y_obs_neutral = base.observed_intensities_isodec(
            spectrum_mz,
            spectrum_int,
            dist0_neutral[:, 0],
            z=int(z),
            match_tol_ppm=match_tol_ppm,
            peak_mz=peak_mz,
        )
        neutral_score = base.css_similarity(y_obs_neutral, dist0_neutral[:, 1])

    best_model = "neutral"
    best_score = neutral_score_union
    best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}
    best_pred = y0

    if allow_1h or allow_2h:
        yp1 = base.vectorize_dist(dist_p1, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX) if allow_1h else None
        ym1 = base.vectorize_dist(dist_m1, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX) if allow_1h else None
        yp2 = base.vectorize_dist(dist_p2, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX) if allow_2h else None
        ym2 = base.vectorize_dist(dist_m2, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX) if allow_2h else None

        comps_plus = [("0", y0)]
        comps_minus = [("0", y0)]
        if allow_1h:
            comps_plus.append(("+H", yp1))
            comps_minus.append(("-H", ym1))
        if allow_2h:
            comps_plus.append(("+2H", yp2))
            comps_minus.append(("-2H", ym2))

        names_plus, vecs_plus = zip(*comps_plus)
        w_plus, y_plus, score_plus = base.fit_simplex_mixture(y_obs, list(vecs_plus))
        weights_plus = dict(zip(names_plus, w_plus))

        names_minus, vecs_minus = zip(*comps_minus)
        w_minus, y_minus, score_minus = base.fit_simplex_mixture(y_obs, list(vecs_minus))
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
            base.make_rule_check(
                "Model intensity",
                state="fail",
                value_text="0.000",
                note="best fragments model has no positive intensity",
            ),
            base.make_rule_check("Raw cosine", state="info", value_text=base.format_num(best_score, 3)),
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
            base.make_rule_check(
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
            base.make_rule_check(
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
            base.make_rule_check(
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
            base.make_rule_check(
                "Anchor search",
                state="fail",
                value_text=str(int(anchor_hits)),
                target_text=f">= {int(cfg.ANCHOR_MIN_MATCHES)} hits",
                note=f"top {int(cfg.ANCHOR_TOP_N)} modeled peaks checked",
            ),
            base.make_rule_check("Raw cosine", state="info", value_text=base.format_num(best_score, 3)),
        ]
        trace["unmetChecks"] = ["Anchor search"]
        return trace

    ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6
    dist_plot = dist_model.copy()
    dist_plot[:, 0] += obs_mz - anchor_theory_mz
    dist_plot[:, 1] *= obs_int / float(np.max(dist_plot[:, 1]))

    isodec_css = float(best_score)
    isodec_accepted = True
    isodec_detail: dict[str, Any] = {}
    if cfg.ENABLE_ISODEC_RULES and isodec_config is not None:
        local_centroids = get_local_centroids_window(
            spectrum_mz,
            spectrum_int,
            obs_mz,
            isodec_config.mzwindowlb,
            isodec_config.mzwindowub,
            force_hill=bool(use_centroid_logic),
        )
        isodec_accepted, isodec_css, shifted_peak = isodec_css_and_accept(
            local_centroids,
            dist_plot,
            z=z,
            peakmz=obs_mz,
            config=isodec_config,
        )
        isodec_detail = _fragment_isodec_detail(
            local_centroids=local_centroids,
            dist_plot=dist_plot,
            z=int(z),
            isodec_config=isodec_config,
        )
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
                base.make_rule_check("Local window", state="fail", value_text="empty", note="no local peaks remain around anchor"),
                base.make_rule_check("IsoDec CSS", state="info", value_text=base.format_num(isodec_css, 3)),
            ]
            trace["unmetChecks"] = ["Local window"]
            return trace
        local_centroids = np.column_stack((spectrum_mz[local_mask], spectrum_int[local_mask]))
    if local_centroids.ndim != 2 or local_centroids.shape[1] != 2:
        trace["stage"] = "local_window"
        trace["reason"] = "invalid_local_window"
        trace["ruleChecks"] = [
            base.make_rule_check("Local window", state="fail", value_text="invalid", note="local peak matrix has wrong shape"),
        ]
        trace["unmetChecks"] = ["Local window"]
        return trace

    local_max_int = float(np.max(local_centroids[:, 1])) if local_centroids.size else 0.0
    if local_max_int <= 0.0:
        trace["stage"] = "local_window"
        trace["reason"] = "zero_local_intensity"
        trace["ruleChecks"] = [
            base.make_rule_check("Local intensity", state="fail", value_text="0.000", note="all local peaks are zero"),
        ]
        trace["unmetChecks"] = ["Local intensity"]
        return trace

    max_anchor_abs_ppm_cfg = getattr(cfg, "FRAG_MAX_ANCHOR_ABS_PPM", None)
    max_anchor_abs_ppm = float(max_anchor_abs_ppm_cfg) if max_anchor_abs_ppm_cfg is not None else (match_tol_ppm * 1.5)
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
    min_isodec_css = float(getattr(cfg, "FRAG_MIN_ISODEC_CSS", float(cfg.MIN_COSINE)))
    max_pc_missing_peaks = float(getattr(cfg, "FRAG_MAX_PC_MISSING_PEAKS", 100.0))
    min_fit_score = float(getattr(cfg, "FRAG_MIN_FIT_SCORE", 0.0))
    min_correlation_cfg = getattr(cfg, "FRAG_MIN_CORRELATION", None)
    min_correlation = float(min_correlation_cfg) if min_correlation_cfg is not None else None

    truth_score_logit, truth_score = _fragment_truth_score(
        correlation_coefficient=float(quality["correlation_coefficient"]),
        pc_missing_peaks=float(quality["pc_missing_peaks"]),
        isodec_css=float(isodec_css),
        top_peaks=bool(isodec_detail.get("topthree", False)),
    )
    legacy_accepted = bool(
        float(isodec_css) >= float(min_isodec_css)
        and abs(float(ppm)) <= float(max_anchor_abs_ppm)
        and len(local_matches) >= int(min_matched_peaks)
        and float(quality["pc_missing_peaks"]) <= float(max_pc_missing_peaks)
        and float(quality["fit_score"]) >= float(min_fit_score)
    )
    if min_correlation is not None and np.isfinite(float(quality["correlation_coefficient"])):
        legacy_accepted = legacy_accepted and float(quality["correlation_coefficient"]) >= float(min_correlation)
    truth_score_accepted = bool(
        truth_score_enabled
        and truth_score is not None
        and float(truth_score) >= float(truth_score_threshold)
    )
    accepted = bool(legacy_accepted or truth_score_accepted)

    checks = [
        base.make_rule_check(
            "IsoDec CSS",
            state="pass" if float(isodec_css) >= float(min_isodec_css) else "fail",
            value_text=base.format_num(isodec_css, 3),
            target_text=f">= {float(min_isodec_css):.3f}",
        ),
        base.make_rule_check(
            "Anchor ppm",
            state="pass" if abs(float(ppm)) <= float(max_anchor_abs_ppm) else "fail",
            value_text=f"{ppm:+.2f} ppm",
            target_text=f"within +/-{float(max_anchor_abs_ppm):.2f} ppm",
        ),
        base.make_rule_check(
            "Local matches",
            state="pass" if len(local_matches) >= int(min_matched_peaks) else "fail",
            value_text=str(int(len(local_matches))),
            target_text=f">= {int(min_matched_peaks)}",
        ),
        base.make_rule_check(
            "Missing peaks",
            state="pass" if float(quality['pc_missing_peaks']) <= float(max_pc_missing_peaks) else "fail",
            value_text=base.format_num(quality["pc_missing_peaks"], 1),
            target_text=f"<= {float(max_pc_missing_peaks):.1f}%",
        ),
        base.make_rule_check(
            "Fit score",
            state="pass" if float(quality['fit_score']) >= float(min_fit_score) else "fail",
            value_text=base.format_num(quality["fit_score"], 3),
            target_text=f">= {float(min_fit_score):.3f}",
        ),
    ]
    if min_correlation is not None and np.isfinite(float(quality["correlation_coefficient"])):
        checks.append(
            base.make_rule_check(
                "Correlation",
                state="pass" if float(quality["correlation_coefficient"]) >= float(min_correlation) else "fail",
                value_text=base.format_num(quality["correlation_coefficient"], 3),
                target_text=f">= {float(min_correlation):.3f}",
            )
        )
    else:
        checks.append(
            base.make_rule_check(
                "Correlation",
                state="info",
                value_text=base.format_num(quality["correlation_coefficient"], 3),
                note="not gated in current config",
            )
        )

    truth_note = "second-chance acceptance" if truth_score_enabled else "disabled"
    checks.append(
        base.make_rule_check(
            "Truth score",
            state=("pass" if truth_score_accepted else ("fail" if truth_score_enabled else "info")),
            value_text=base.format_num(truth_score, 3),
            target_text=f">= {float(truth_score_threshold):.3f}" if truth_score_enabled else "disabled",
            note=truth_note,
        )
    )
    checks.append(
        base.make_rule_check(
            "Legacy gate",
            state="pass" if legacy_accepted else "fail",
            value_text=base.bool_text(legacy_accepted),
            target_text="need yes unless truth score rescues",
        )
    )
    checks.append(
        base.make_rule_check(
            "Fragments IsoDec accept",
            state="info",
            value_text=base.bool_text(isodec_accepted),
            note="shown for context only",
        )
    )

    unmet = [check["label"] for check in checks if check["state"] == "fail"]
    trace.update(
        {
            "stage": "accepted" if accepted else "quality_gates",
            "accepted": bool(accepted),
            "reason": "accepted_candidate" if accepted else "fragments_quality_gates_failed",
            "ruleChecks": checks,
            "unmetChecks": [] if accepted else unmet,
            "css": float(isodec_css),
            "anchorPpm": float(ppm),
            "score": float(quality["evidence_score"]),
            "coverage": float(comp["coverage"]),
            "matchCount": int(len(local_matches)),
            "obsIdx": int(obs_idx),
            "obsMz": float(obs_mz),
            "obsInt": float(obs_int),
            "obsRelInt": float(obs_rel_int),
            "truthScore": float(truth_score) if truth_score is not None else None,
            "truthScoreAccepted": bool(truth_score_accepted),
            "legacyAccepted": bool(legacy_accepted),
        }
    )
    return trace


def write_shell_files() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("styles.css", "app.js", "index.html"):
        shutil.copy2(TEMPLATE_DIR / name, REPORT_DIR / name)

    index_path = REPORT_DIR / "index.html"
    index_text = index_path.read_text(encoding="utf-8")
    index_text = index_text.replace("Q10R Missed-Ion Atlas", "Q10R Truth-Score FN Atlas")
    index_text = index_text.replace("Missed-Ion Atlas", "Truth-Score FN Atlas")
    index_text = index_text.replace(
        "A standalone visual board for Q10R missed <code>b</code> and <code>c</code> ions. Each card pairs the\n            manual miss with a diagnose-mode spectrum snapshot and explicit IsoDec rule details generated from the\n            current codebase.",
        "A standalone visual board for Q10R <code>b</code>/<code>c</code> ions that remain false negatives under the current simplified fragments gate plus <code>truth_score@0.80</code>. Each card pairs the manual true ion with a diagnose snapshot and the current fragments-side reject checks.",
    )
    index_text = index_text.replace("Q10R manual misses", "truth-score FN set")
    index_text = index_text.replace("Missed ions", "False negatives")
    index_text = index_text.replace("Recurring misses", "Recurring FN")
    index_text = index_text.replace("miss rows per scan", "FN rows per scan")
    index_path.write_text(index_text, encoding="utf-8")

    app_path = REPORT_DIR / "app.js"
    app_text = app_path.read_text(encoding="utf-8")
    app_text = app_text.replace("q10r-missed-static-review-v1", "q10r-truthscore-fn-static-review-v1")
    app_text = app_text.replace("miss rows", "FN rows")
    app_text = app_text.replace("manual missed rows rendered into static diagnose cards", "manual true rows still missed under simplified gate + truth_score@0.80")
    app_text = app_text.replace("unique base misses", "unique base FN")
    app_text = app_text.replace("normalized by ion type, position and charge", "normalized by ion type, position and charge")
    app_text = app_text.replace("top recurring miss", "top recurring FN")
    app_text = app_text.replace("rows share this miss pattern", "rows share this false-negative pattern")
    app_text = app_text.replace("repeated miss signature", "repeated false-negative signature")
    app_text = app_text.replace("miss rows in this scan", "FN rows in this scan")
    app_text = app_text.replace("manual missed row", "manual true row")
    app_text = app_text.replace("Missed ion", "False-negative ion")
    app_text = app_text.replace("miss rows rendered", "FN rows rendered")
    app_path.write_text(app_text, encoding="utf-8")


def main() -> None:
    _configure_base_paths()
    base._evaluate_fragment_variant_trace = _evaluate_fragment_variant_trace
    write_shell_files()
    with _truth_score_cfg():
        rows = compute_fn_rows()
        recurring = base.Counter((str(row["ion_type"]).lower(), int(row["pos"]), int(row["charge"])) for row in rows)
        items = [base.build_item(row, recurring) for row in rows]
        report = base.build_report(items)
        report["source"]["missedCsv"] = str(FN_CSV.relative_to(ROOT))
        report["source"]["mode"] = "simplified_fragments_plus_truth_score_0_80"
        base.write_data_js(report)
    print(f"Generated report: {REPORT_DIR}")
    print(f"Rows: {len(rows)}")
    print(f"Images: {len(items)}")


if __name__ == "__main__":
    main()
