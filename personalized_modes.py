from __future__ import annotations

from copy import copy
from pathlib import Path

import numpy as np
import personalized_config as cfg
from personalized_match import (
    composition_to_formula,
    compute_fragment_intensity_cap,
    diagnose_candidate,
    execute_hybrid_strategy,
    get_local_centroids_window,
    isodec_css_and_accept,
    match_theory_peaks,
    nearest_peak_index,
    parse_fragment_spec,
    sanitize_filename,
    strip_peaks_above_intensity_cap,
    within_ppm,
    write_csv,
)
from personalized_plot import plot_overlay
from personalized_sequence import (
    apply_neutral_loss,
    get_precursor_composition,
    get_disulfide_logic,
    get_neutral_monomer_composition,
    ion_composition_from_sequence,
    ion_series,
    neutral_loss_columns,
    neutral_loss_label,
    neutral_loss_variants,
)
from personalized_variants import variant_rank_key_from_result, variant_type_from_suffix
from personalized_theory import (
    build_sample_axis,
    css_similarity,
    fit_simplex_mixture,
    get_anchor_idx,
    observed_intensities_isodec,
    theoretical_isodist_from_comp,
    vectorize_dist,
)


def run_raw_mode(spectrum) -> None:
    if spectrum.ndim != 2 or spectrum.shape[1] != 2:
        raise ValueError(f"Expected spectrum shape (N, 2), got {spectrum.shape}")
    plot_overlay(spectrum, [], mz_min=None, mz_max=None)


def run_raw_headless(spectrum) -> dict:
    if spectrum.ndim != 2 or spectrum.shape[1] != 2:
        raise ValueError(f"Expected spectrum shape (N, 2), got {spectrum.shape}")
    spectrum_copy = np.array(spectrum, dtype=float, copy=True)
    return {
        "spectrum": spectrum_copy,
        "spectrum_mz": np.asarray(spectrum_copy[:, 0], dtype=float),
        "spectrum_int": np.asarray(spectrum_copy[:, 1], dtype=float),
    }


def _find_most_intense_window(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    window_da: float,
) -> dict | None:
    spectrum_mz = np.asarray(spectrum_mz, dtype=float)
    spectrum_int = np.asarray(spectrum_int, dtype=float)
    if spectrum_mz.size == 0 or spectrum_int.size == 0:
        return None
    window_da = float(window_da)
    if not np.isfinite(window_da) or window_da <= 0:
        return None

    n = int(spectrum_mz.size)
    j = 0
    current_sum = 0.0
    best_sum = -1.0
    best_start = 0
    best_end = 0

    for i in range(n):
        if j < i:
            j = i
            current_sum = 0.0
        limit = float(spectrum_mz[i]) + window_da
        while j < n and float(spectrum_mz[j]) <= limit:
            current_sum += float(spectrum_int[j])
            j += 1
        if current_sum > best_sum:
            best_sum = current_sum
            best_start = i
            best_end = j
        current_sum -= float(spectrum_int[i])

    mz_min = float(spectrum_mz[best_start])
    mz_max = float(mz_min + window_da)
    return {
        "start_idx": int(best_start),
        "end_idx": int(best_end),
        "mz_min": mz_min,
        "mz_max": mz_max,
        "sum_intensity": float(best_sum),
    }


def _build_noise_level_model(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    *,
    num_splits: int,
    hist_bins: int,
) -> dict:
    spectrum_mz = np.asarray(spectrum_mz, dtype=float)
    spectrum_int = np.asarray(spectrum_int, dtype=float)
    positive = spectrum_int[np.isfinite(spectrum_int) & (spectrum_int > 0)]
    fallback = float(np.median(positive)) if positive.size else 0.0
    if spectrum_mz.size == 0 or spectrum_int.size == 0:
        return {"coeffs": None, "fallback": fallback}

    split_count = max(1, int(num_splits))
    split_size = max(1, int(np.ceil(spectrum_mz.size / split_count)))
    avg_mz: list[float] = []
    modal_intens: list[float] = []

    for start in range(0, spectrum_mz.size, split_size):
        stop = min(start + split_size, spectrum_mz.size)
        chunk_mz = spectrum_mz[start:stop]
        chunk_int = spectrum_int[start:stop]
        chunk_int = chunk_int[np.isfinite(chunk_int) & (chunk_int > 0)]
        if chunk_mz.size == 0 or chunk_int.size == 0:
            continue
        if np.allclose(chunk_int, chunk_int[0]):
            mode_value = float(chunk_int[0])
        else:
            bins = max(8, min(int(hist_bins), int(chunk_int.size)))
            hist, edges = np.histogram(chunk_int, bins=bins)
            mode_idx = int(np.argmax(hist))
            mode_value = float((edges[mode_idx] + edges[mode_idx + 1]) / 2.0)
        avg_mz.append(float(np.mean(chunk_mz)))
        modal_intens.append(mode_value)

    if not modal_intens:
        return {"coeffs": None, "fallback": fallback}

    fallback = float(np.median(modal_intens))
    coeffs = None
    if len(avg_mz) >= 2:
        deg = 2 if len(avg_mz) >= 3 else 1
        try:
            coeffs = np.polyfit(
                np.asarray(avg_mz, dtype=float),
                np.asarray(modal_intens, dtype=float),
                deg,
            )
        except Exception:
            coeffs = None
    return {"coeffs": coeffs, "fallback": fallback}


def _choose_fragment_anchor_hypothesis(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    *,
    sample_mzs: np.ndarray,
    best_pred: np.ndarray,
    dist_model: np.ndarray,
    z: int,
    obs_max: float,
    match_tol_ppm: float,
    use_centroid_logic: bool,
) -> tuple[dict[str, float | int] | None, int]:
    anchor_window = float(getattr(cfg, "FRAG_ANCHOR_CENTROID_WINDOW_DA", 0.2))
    top_n = max(1, int(getattr(cfg, "ANCHOR_TOP_N", 3)))
    local_top_k = max(1, int(getattr(cfg, "FRAG_ANCHOR_LOCAL_TOP_K", 3)))
    min_obs_rel_int = float(getattr(cfg, "MIN_OBS_REL_INT", 0.0))
    sorted_idx = np.argsort(best_pred)[::-1][:top_n]
    hypotheses: list[dict[str, float | int]] = []
    seen_obs_idx: set[int] = set()

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

        candidate_rows: list[tuple[float, float, int]] = []
        if isinstance(local_centroids, np.ndarray) and local_centroids.size:
            intensity_order = np.argsort(local_centroids[:, 1])[::-1][:local_top_k]
            ppm_errors = np.abs(
                (local_centroids[:, 0] - mz_candidate)
                / max(abs(mz_candidate), 1e-12)
                * 1e6
            )
            ppm_order = np.argsort(ppm_errors)[:local_top_k]
            picked: list[int] = []
            for local_idx in np.concatenate([intensity_order, ppm_order]):
                local_idx = int(local_idx)
                if local_idx not in picked:
                    picked.append(local_idx)
            for local_idx in picked:
                candidate_rows.append(
                    (
                        float(local_centroids[local_idx, 0]),
                        float(local_centroids[local_idx, 1]),
                        nearest_peak_index(
                            spectrum_mz, float(local_centroids[local_idx, 0])
                        ),
                    )
                )
        else:
            obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
            candidate_rows.append(
                (
                    float(spectrum_mz[obs_idx_c]),
                    float(spectrum_int[obs_idx_c]),
                    int(obs_idx_c),
                )
            )

        local_max_int = (
            float(np.max(local_centroids[:, 1]))
            if isinstance(local_centroids, np.ndarray) and local_centroids.size
            else 0.0
        )
        for obs_mz_c, obs_int_c, obs_idx_c in candidate_rows:
            if obs_idx_c in seen_obs_idx:
                continue
            if not within_ppm(obs_mz_c, mz_candidate, float(match_tol_ppm)):
                continue
            if min_obs_rel_int > 0 and obs_int_c < obs_max * min_obs_rel_int:
                continue

            dist_shifted = dist_model.copy()
            dist_shifted[:, 0] += float(obs_mz_c - mz_candidate)
            y_obs = observed_intensities_isodec(
                spectrum_mz,
                spectrum_int,
                dist_shifted[:, 0],
                z=int(z),
                match_tol_ppm=float(match_tol_ppm),
                peak_mz=float(obs_mz_c),
            )
            quick_css = float(css_similarity(y_obs, dist_shifted[:, 1]))
            ppm_abs = abs(
                (obs_mz_c - mz_candidate) / max(abs(mz_candidate), 1e-12) * 1e6
            )
            ppm_score = float(
                np.clip(1.0 - (ppm_abs / max(float(match_tol_ppm), 1e-9)), 0.0, 1.0)
            )
            local_int_score = float(
                np.clip(obs_int_c / max(local_max_int, obs_int_c, 1e-12), 0.0, 1.0)
            )
            score = 0.60 * quick_css + 0.25 * local_int_score + 0.15 * ppm_score
            hypotheses.append(
                {
                    "score": float(score),
                    "quick_css": float(quick_css),
                    "obs_mz": float(obs_mz_c),
                    "obs_int": float(obs_int_c),
                    "obs_idx": int(obs_idx_c),
                    "theory_mz": float(mz_candidate),
                    "ppm_abs": float(ppm_abs),
                }
            )
            seen_obs_idx.add(int(obs_idx_c))

    if not hypotheses:
        return None, 0
    best = max(
        hypotheses,
        key=lambda item: (
            float(item["score"]),
            float(item["quick_css"]),
            -float(item["ppm_abs"]),
            float(item["obs_int"]),
        ),
    )
    return best, len(hypotheses)


def _iter_fragment_anchor_candidates_by_intensity(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    *,
    mz_candidate: float,
    use_centroid_logic: bool,
) -> list[tuple[float, float, int]]:
    anchor_window = float(getattr(cfg, "FRAG_ANCHOR_CENTROID_WINDOW_DA", 0.2))
    local_top_k = max(1, int(getattr(cfg, "FRAG_ANCHOR_LOCAL_TOP_K", 3)))
    local_centroids = get_local_centroids_window(
        spectrum_mz,
        spectrum_int,
        center_mz=mz_candidate,
        lb=-anchor_window,
        ub=anchor_window,
        force_hill=bool(use_centroid_logic),
    )
    candidates: list[tuple[float, float, int]] = []
    if isinstance(local_centroids, np.ndarray) and local_centroids.size:
        intensity_order = np.argsort(local_centroids[:, 1])[::-1]
        for local_idx in intensity_order[:local_top_k]:
            local_idx = int(local_idx)
            obs_mz_c = float(local_centroids[local_idx, 0])
            obs_int_c = float(local_centroids[local_idx, 1])
            obs_idx_c = nearest_peak_index(spectrum_mz, obs_mz_c)
            candidates.append((obs_mz_c, obs_int_c, int(obs_idx_c)))
    else:
        obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
        candidates.append(
            (
                float(spectrum_mz[obs_idx_c]),
                float(spectrum_int[obs_idx_c]),
                int(obs_idx_c),
            )
        )
    return candidates


def _evaluate_noise_level(model: dict | None, mz: float) -> float:
    fallback = 0.0
    if isinstance(model, dict):
        try:
            fallback = float(model.get("fallback", 0.0) or 0.0)
        except Exception:
            fallback = 0.0
        coeffs = model.get("coeffs")
        if coeffs is not None:
            try:
                level = float(np.polyval(np.asarray(coeffs, dtype=float), float(mz)))
                if np.isfinite(level) and level > 0:
                    return max(level, fallback, 1e-12)
            except Exception:
                pass
    return max(fallback, 1e-12)


def _scale_theoretical_to_observed(
    theory_int: np.ndarray, obs_int: np.ndarray
) -> tuple[np.ndarray, float]:
    theory = np.asarray(theory_int, dtype=float)
    obs = np.asarray(obs_int, dtype=float)
    denom = float(np.dot(theory, theory))
    if not np.isfinite(denom) or denom <= 0.0:
        return theory.copy(), 1.0
    scale = float(np.dot(theory, obs) / denom)
    if not np.isfinite(scale) or scale < 0.0:
        scale = 0.0
    return theory * scale, scale


def _fit_score_from_envelope(obs_int: np.ndarray, theory_int: np.ndarray) -> float:
    obs = np.asarray(obs_int, dtype=float)
    theory = np.asarray(theory_int, dtype=float)
    total_theory = float(np.sum(theory))
    if theory.size == 0 or total_theory <= 0.0:
        return 0.0
    numerator = 0.0
    for idx, theo_value in enumerate(theory):
        theo = float(theo_value)
        if theo <= 0.0:
            continue
        exp = float(obs[idx]) if idx < obs.size else 0.0
        if exp <= 0.0:
            ratio = 0.0
        else:
            ratio = min(theo / exp, exp / theo)
        numerator += ratio * theo
    return float(np.clip(numerator / total_theory, 0.0, 1.0))


def _safe_pearson(obs_int: np.ndarray, theory_int: np.ndarray) -> float:
    obs = np.asarray(obs_int, dtype=float)
    theory = np.asarray(theory_int, dtype=float)
    if obs.size < 2 or theory.size < 2:
        return 0.0
    if float(np.std(obs)) <= 0.0 or float(np.std(theory)) <= 0.0:
        return 0.0
    corr = np.corrcoef(obs, theory)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def _chisq_statistic(obs_int: np.ndarray, theory_int: np.ndarray) -> float:
    obs = np.asarray(obs_int, dtype=float)
    theory = np.asarray(theory_int, dtype=float)
    obs_sum = float(np.sum(obs))
    theory_sum = float(np.sum(theory))
    if obs.size == 0 or theory.size == 0 or obs_sum <= 0.0 or theory_sum <= 0.0:
        return float("inf")
    obs_norm = obs / obs_sum
    theory_norm = theory / theory_sum
    expected = np.clip(theory_norm, 1e-12, None)
    stat = float(np.sum(((obs_norm - theory_norm) ** 2) / expected))
    return stat if np.isfinite(stat) else float("inf")


def run_precursor_headless(
    residues,
    spectrum,
    isodec_config,
    *,
    apply_calibration: bool | None = None,
    precursor_tol_ppm: float | None = None,
) -> dict:
    # Precursor search uses its own tolerance and must not mutate the shared downstream IsoDec config.
    precursor_tol_ppm = float(
        precursor_tol_ppm
        if precursor_tol_ppm is not None
        else getattr(cfg, "PRECURSOR_MATCH_TOL_PPM", cfg.MATCH_TOL_PPM)
    )
    local_isodec_config = copy(isodec_config) if isodec_config is not None else None
    if local_isodec_config is not None and hasattr(local_isodec_config, "matchtol"):
        local_isodec_config.matchtol = precursor_tol_ppm
    if apply_calibration is None:
        apply_calibration = bool(getattr(cfg, "ENABLE_LOCK_MASS", False))

    complex_comp = get_precursor_composition(residues)
    precursor_theories: dict[int, dict] = {}

    for z in range(int(cfg.PRECURSOR_MIN_CHARGE), int(cfg.PRECURSOR_MAX_CHARGE) + 1):
        try:
            dist = theoretical_isodist_from_comp(complex_comp, z)
        except ValueError:
            continue
        if dist.size == 0:
            continue
        if dist.ndim != 2 or dist.shape[1] != 2:
            continue
        dist = dist[np.argsort(dist[:, 0])]
        anchor_idx = get_anchor_idx(dist)
        anchor_mz = float(dist[anchor_idx, 0])
        mz_min = float(dist[0, 0])
        mz_max = float(dist[-1, 0])
        precursor_theories[z] = {
            "dist": dist,
            "anchor_mz": anchor_mz,
            "mz_min": mz_min,
            "mz_max": mz_max,
        }

    if not precursor_theories:
        print("No precursor theories generated; check composition and charge range.")
        return {
            "spectrum": np.array(spectrum, dtype=float, copy=True),
            "spectrum_mz": np.asarray(spectrum[:, 0], dtype=float),
            "spectrum_int": np.asarray(spectrum[:, 1], dtype=float),
            "match_found": False,
            "search_status": "not_found",
            "best_z": None,
            "best_state": None,
            "best_css": 0.0,
            "best_composite_score": 0.0,
            "best_coverage": 0.0,
            "best_ppm_rmse": None,
            "shift_ppm": 0.0,
            "best_obs_mz": None,
            "best_theory_mz": None,
            "best_theory_dist": None,
            "plot_window": None,
            "precursor_tol_ppm": float(precursor_tol_ppm),
            "calibration_requested": bool(apply_calibration),
            "calibration_applied": False,
            "search_window": None,
            "ambiguous_window": None,
            "ambiguous_candidates": [],
            "candidates": [],
        }

    state_shifts = [("0", 0)]
    if bool(getattr(cfg, "ENABLE_H_TRANSFER", False)):
        state_shifts = [
            ("0", 0),
            ("+H", 1),
            ("+2H", 2),
            ("-H", -1),
            ("-2H", -2),
        ]

    all_theory_dist = None
    dist_list = []
    for z, theory in precursor_theories.items():
        base_dist = theory.get("dist")
        if not isinstance(base_dist, np.ndarray) or base_dist.size == 0:
            continue
        for _state, h_shift in state_shifts:
            shift_mz = (float(h_shift) * float(cfg.H_TRANSFER_MASS)) / float(z)
            if h_shift:
                dist_shifted = base_dist.copy()
                dist_shifted[:, 0] += shift_mz
            else:
                dist_shifted = base_dist
            dist_list.append(dist_shifted)
    if dist_list:
        all_theory_dist = np.vstack(dist_list)
        all_theory_dist = all_theory_dist[np.argsort(all_theory_dist[:, 0])]

    def _scale_dist_to_obs(dist: np.ndarray, obs_int: float) -> np.ndarray:
        if dist is None or dist.size == 0:
            return dist
        max_theory = float(np.max(dist[:, 1])) if dist.size else 0.0
        if max_theory <= 0 or obs_int <= 0:
            return dist
        dist_scaled = dist.copy()
        dist_scaled[:, 1] *= float(obs_int) / max_theory
        return dist_scaled

    score_w_css = float(getattr(cfg, "PRECURSOR_SCORE_W_CSS", 0.45))
    score_w_cov = float(getattr(cfg, "PRECURSOR_SCORE_W_COVERAGE", 0.25))
    score_w_ppm = float(getattr(cfg, "PRECURSOR_SCORE_W_PPM", 0.20))
    score_w_spacing = float(getattr(cfg, "PRECURSOR_SCORE_W_SPACING", 0.10))
    score_w_intensity = float(getattr(cfg, "PRECURSOR_SCORE_W_INTENSITY", 0.25))
    score_w_sum = (
        score_w_css + score_w_cov + score_w_ppm + score_w_spacing + score_w_intensity
    )
    if score_w_sum <= 0:
        score_w_css, score_w_cov, score_w_ppm, score_w_spacing, score_w_intensity = (
            0.45,
            0.25,
            0.20,
            0.10,
            0.25,
        )
    else:
        score_w_css /= score_w_sum
        score_w_cov /= score_w_sum
        score_w_ppm /= score_w_sum
        score_w_spacing /= score_w_sum
        score_w_intensity /= score_w_sum

    anchor_top_k = int(getattr(cfg, "PRECURSOR_ANCHOR_TOP_K", 3))
    if anchor_top_k <= 0:
        anchor_top_k = 1
    anchor_top_fraction = float(getattr(cfg, "PRECURSOR_ANCHOR_TOP_FRACTION", 0.30))
    if not np.isfinite(anchor_top_fraction) or anchor_top_fraction <= 0:
        anchor_top_fraction = 0.30
    anchor_top_fraction = float(np.clip(anchor_top_fraction, 0.01, 1.0))
    anchor_min_rel_int = float(getattr(cfg, "PRECURSOR_ANCHOR_MIN_REL_INT", 0.40))
    if not np.isfinite(anchor_min_rel_int):
        anchor_min_rel_int = 0.40
    anchor_min_rel_int = float(np.clip(anchor_min_rel_int, 0.0, 1.0))
    min_coverage = float(getattr(cfg, "PRECURSOR_MIN_COVERAGE", 0.30))
    max_anchor_abs_ppm = float(
        getattr(cfg, "PRECURSOR_MAX_ANCHOR_ABS_PPM", precursor_tol_ppm * 1.5)
    )
    max_residual_rmse_ppm = float(
        getattr(cfg, "PRECURSOR_MAX_RESIDUAL_RMSE_PPM", precursor_tol_ppm)
    )
    ppm_sigma = float(getattr(cfg, "PRECURSOR_PPM_SIGMA", precursor_tol_ppm))
    if ppm_sigma <= 0:
        ppm_sigma = max(precursor_tol_ppm, 1.0)
    ambiguity_margin = float(getattr(cfg, "PRECURSOR_AMBIGUITY_MARGIN", 0.03))
    ambiguity_guard = bool(getattr(cfg, "PRECURSOR_ENABLE_AMBIGUITY_GUARD", True))
    ambiguous_result: dict | None = None

    def _build_anchor_indices(
        local: np.ndarray, target_mz: float, top_k: int, tol_ppm: float
    ) -> list[int]:
        if local.ndim != 2 or local.shape[0] == 0:
            return []
        local_mz = np.asarray(local[:, 0], dtype=float)
        local_int = np.asarray(local[:, 1], dtype=float)
        n_local = int(local.shape[0])
        indices: list[int] = []
        top_int_idx = np.argsort(local_int)[::-1]
        top_rank_n = int(np.ceil(n_local * anchor_top_fraction))
        top_rank_n = max(int(top_k), max(1, top_rank_n))
        top_rank_n = min(top_rank_n, n_local)
        allowed_mask = np.zeros(n_local, dtype=bool)
        allowed_mask[top_int_idx[:top_rank_n]] = True

        nearest_idx = int(np.argmin(np.abs(local_mz - float(target_mz))))
        tol_da = (
            abs(float(target_mz)) * float(tol_ppm) * 1e-6
            if float(target_mz) > 0
            else 0.0
        )
        if tol_da > 0:
            in_tol = np.where(np.abs(local_mz - float(target_mz)) <= tol_da)[0]
            if in_tol.size:
                idx_sorted = in_tol[
                    np.argsort(np.abs(local_mz[in_tol] - float(target_mz)))
                ]
                for idx in idx_sorted[:top_k]:
                    i = int(idx)
                    if allowed_mask[i]:
                        indices.append(i)
        if nearest_idx not in indices and allowed_mask[nearest_idx]:
            indices.append(nearest_idx)
        for idx in top_int_idx[:top_k]:
            i = int(idx)
            if i not in indices:
                indices.append(i)
        return indices

    def _match_theory_local(
        local: np.ndarray,
        dist_shifted: np.ndarray,
        shift_da: float,
        tol_ppm: float,
    ) -> list[dict]:
        local_mz = np.asarray(local[:, 0], dtype=float)
        local_int = np.asarray(local[:, 1], dtype=float)
        theory_mz = np.asarray(dist_shifted[:, 0], dtype=float)
        theory_int = np.asarray(dist_shifted[:, 1], dtype=float)
        pred_mz = theory_mz + float(shift_da)

        used = np.zeros(local_mz.shape[0], dtype=bool)
        last_local_idx = -1
        rows: list[dict] = []
        for i in range(pred_mz.shape[0]):
            mz_pred = float(pred_mz[i])
            delta = abs(mz_pred) * float(tol_ppm) * 1e-6 if mz_pred > 0 else 0.0
            low = mz_pred - delta
            high = mz_pred + delta
            start_idx = int(np.searchsorted(local_mz, low, side="left"))
            end_idx = int(np.searchsorted(local_mz, high, side="right"))
            if end_idx <= start_idx:
                continue
            start_idx = max(start_idx, last_local_idx + 1)
            if end_idx <= start_idx:
                continue
            cands = np.arange(start_idx, end_idx, dtype=int)
            if cands.size == 0:
                continue
            cands = cands[~used[cands]]
            if cands.size == 0:
                continue
            pick = int(cands[np.argmin(np.abs(local_mz[cands] - mz_pred))])
            used[pick] = True
            last_local_idx = pick
            mz_obs = float(local_mz[pick])
            mz_theory = float(theory_mz[i])
            residual_da = mz_obs - mz_pred
            residual_ppm = (residual_da / mz_theory) * 1e6 if mz_theory else 0.0
            rows.append(
                {
                    "theory_idx": int(i),
                    "theory_mz": mz_theory,
                    "theory_int": float(theory_int[i]),
                    "pred_mz": mz_pred,
                    "obs_idx": int(pick),
                    "obs_mz": mz_obs,
                    "obs_int": float(local_int[pick]),
                    "residual_da": float(residual_da),
                    "residual_ppm": float(residual_ppm),
                }
            )
        return rows

    def _composite_components(
        *,
        css: float,
        matches: list[dict],
        dist_shifted: np.ndarray,
        anchor_ppm_abs: float,
        anchor_theory_mz: float,
        intensity_ratio: float,
    ) -> dict:
        css_val = float(np.clip(css, 0.0, 1.0))
        theory_int = np.asarray(dist_shifted[:, 1], dtype=float)
        theory_sum = float(np.sum(theory_int)) if theory_int.size else 0.0
        if theory_sum > 0 and matches:
            matched_sum = float(
                sum(max(0.0, float(r.get("theory_int", 0.0))) for r in matches)
            )
            coverage = float(np.clip(matched_sum / theory_sum, 0.0, 1.0))
        else:
            coverage = 0.0

        if matches:
            weights = np.array(
                [max(float(r.get("theory_int", 0.0)), 1e-12) for r in matches],
                dtype=float,
            )
            residual_ppm = np.array(
                [float(r.get("residual_ppm", 0.0)) for r in matches], dtype=float
            )
            ppm_rmse = float(
                np.sqrt(np.average(residual_ppm * residual_ppm, weights=weights))
            )
        else:
            ppm_rmse = float("inf")

        residual_score = (
            0.0 if not np.isfinite(ppm_rmse) else float(np.exp(-ppm_rmse / ppm_sigma))
        )
        anchor_score = float(np.exp(-abs(anchor_ppm_abs) / ppm_sigma))
        ppm_consistency = float(
            np.clip((0.70 * residual_score) + (0.30 * anchor_score), 0.0, 1.0)
        )

        if len(matches) >= 2:
            obs = np.array([float(r.get("obs_mz", 0.0)) for r in matches], dtype=float)
            pred = np.array(
                [float(r.get("pred_mz", 0.0)) for r in matches], dtype=float
            )
            diff_err = np.diff(obs) - np.diff(pred)
            spacing_rmse_da = (
                float(np.sqrt(np.mean(diff_err * diff_err))) if diff_err.size else 0.0
            )
            anchor_tol_da = (
                abs(float(anchor_theory_mz)) * float(precursor_tol_ppm) * 1e-6
            )
            spacing_sigma_cfg = getattr(cfg, "PRECURSOR_SPACING_SIGMA_DA", None)
            if spacing_sigma_cfg is None:
                spacing_sigma_da = float(anchor_tol_da)
            else:
                spacing_sigma_da = float(spacing_sigma_cfg)
            if spacing_sigma_da <= 0:
                spacing_sigma_da = max(anchor_tol_da, 1e-6)
            spacing_consistency = float(np.exp(-spacing_rmse_da / spacing_sigma_da))
        else:
            spacing_rmse_da = float("inf")
            spacing_consistency = 0.5

        intensity_prior = float(np.sqrt(np.clip(float(intensity_ratio), 0.0, 1.0)))
        composite_score = (
            score_w_css * css_val
            + score_w_cov * coverage
            + score_w_ppm * ppm_consistency
            + score_w_spacing * spacing_consistency
            + score_w_intensity * intensity_prior
        )
        return {
            "coverage": float(coverage),
            "ppm_rmse": float(ppm_rmse),
            "ppm_consistency": float(ppm_consistency),
            "spacing_rmse_da": float(spacing_rmse_da),
            "spacing_consistency": float(spacing_consistency),
            "intensity_prior": float(intensity_prior),
            "composite_score": float(composite_score),
        }

    search_spectrum, search_window = _filter_spectrum_by_requested_mz_window(
        np.array(spectrum, dtype=float, copy=True)
    )
    search_mz = (
        np.asarray(search_spectrum[:, 0], dtype=float)
        if search_spectrum.size
        else np.asarray([], dtype=float)
    )
    search_int = (
        np.asarray(search_spectrum[:, 1], dtype=float)
        if search_spectrum.size
        else np.asarray([], dtype=float)
    )
    match_found = False
    best_z = None
    best_state = None
    best_css = 0.0
    best_composite_score = 0.0
    best_coverage = 0.0
    best_ppm_rmse = None
    shift_ppm = 0.0
    best_obs_mz = None
    best_theory_mz = None
    best_theory_dist = None
    first_anchor_int = None
    best_by_charge: dict[int, dict] = {}
    window_da = float(getattr(cfg, "PRECURSOR_WINDOW_DA", 5.1))
    if not np.isfinite(window_da) or window_da <= 0:
        window_da = 5.1
    anchor_search_da = max(float(window_da) / 2.0, 1.0)
    _use_monoisotopic = (
        str(getattr(cfg, "ANCHOR_MODE", "most_intense")).lower() == "monoisotopic"
    )
    all_candidates: list[dict] = []

    print(
        f"--- Starting Precursor Search (candidate-driven anchor search, "
        f"anchor_window={anchor_search_da:.2f} Da) ---"
    )
    if search_window is not None:
        print(
            "Restricting precursor search to requested m/z window "
            f"[{search_window['min'] if search_window['min'] is not None else '-inf'}, "
            f"{search_window['max'] if search_window['max'] is not None else '+inf'}]."
        )
    if search_spectrum.size:
        for z, theory in precursor_theories.items():
            base_dist = theory.get("dist")
            if not isinstance(base_dist, np.ndarray) or base_dist.size == 0:
                continue

            for state, h_shift in state_shifts:
                shift_mz = (float(h_shift) * float(cfg.H_TRANSFER_MASS)) / float(z)
                if h_shift:
                    dist_shifted = base_dist.copy()
                    dist_shifted[:, 0] += shift_mz
                else:
                    dist_shifted = base_dist

                theory_mz = (
                    float(dist_shifted[0, 0])
                    if _use_monoisotopic
                    else float(dist_shifted[get_anchor_idx(dist_shifted), 0])
                )
                anchor_window_mask = (search_mz >= theory_mz - anchor_search_da) & (
                    search_mz <= theory_mz + anchor_search_da
                )
                if not np.any(anchor_window_mask):
                    continue

                anchor_window_idx = np.where(anchor_window_mask)[0]
                ranked_anchor_idx = anchor_window_idx[
                    np.argsort(search_int[anchor_window_idx])[::-1]
                ]
                nearest_anchor_idx = int(
                    anchor_window_idx[
                        np.argmin(np.abs(search_mz[anchor_window_idx] - theory_mz))
                    ]
                )
                selected_anchor_idx: list[int] = []
                for idx in [
                    nearest_anchor_idx,
                    *ranked_anchor_idx[:anchor_top_k].tolist(),
                ]:
                    i = int(idx)
                    if i not in selected_anchor_idx:
                        selected_anchor_idx.append(i)

                left_span = max(float(theory_mz - dist_shifted[0, 0]), 0.0)
                right_span = max(float(dist_shifted[-1, 0] - theory_mz), 0.0)
                anchor_pad_da = max(
                    abs(float(theory_mz)) * float(precursor_tol_ppm) * 3e-6, 0.5
                )
                local_lb = -(left_span + anchor_pad_da)
                local_ub = right_span + anchor_pad_da

                def _score_precursor_candidate(*, use_centroid_logic: bool):
                    local_best_for_state: dict | None = None
                    for obs_idx_global in selected_anchor_idx:
                        obs_anchor_mz = float(search_mz[int(obs_idx_global)])
                        local = get_local_centroids_window(
                            search_mz,
                            search_int,
                            center_mz=obs_anchor_mz,
                            lb=local_lb,
                            ub=local_ub,
                            force_hill=bool(use_centroid_logic),
                        )
                        if not isinstance(local, np.ndarray) or local.size == 0:
                            local_mask = (search_mz >= obs_anchor_mz + local_lb) & (
                                search_mz <= obs_anchor_mz + local_ub
                            )
                            if not np.any(local_mask):
                                continue
                            local = np.column_stack(
                                (search_mz[local_mask], search_int[local_mask])
                            )
                        if local.ndim != 2 or local.shape[1] != 2:
                            continue

                        local_max_int = (
                            float(np.max(local[:, 1])) if local.shape[0] > 0 else 0.0
                        )
                        if local_max_int <= 0:
                            continue

                        anchor_indices = _build_anchor_indices(
                            local, theory_mz, anchor_top_k, precursor_tol_ppm
                        )
                        if not anchor_indices:
                            anchor_indices = [
                                int(np.argmin(np.abs(local[:, 0] - obs_anchor_mz)))
                            ]

                        for anchor_idx in anchor_indices:
                            anchor_idx = int(anchor_idx)
                            obs_mz_seed = float(local[anchor_idx, 0])
                            obs_anchor_int = float(local[anchor_idx, 1])
                            obs_mz_eval = float(obs_mz_seed)
                            if (
                                cfg.ENABLE_ISODEC_RULES
                                and local_isodec_config is not None
                            ):
                                accepted_model, isodec_css, shifted_peak = (
                                    isodec_css_and_accept(
                                        local,
                                        dist_shifted,
                                        z=int(z),
                                        peakmz=obs_mz_seed,
                                        config=local_isodec_config,
                                    )
                                )
                                if shifted_peak is not None:
                                    obs_mz_eval = float(shifted_peak)
                            else:
                                y_obs_seed = observed_intensities_isodec(
                                    local[:, 0],
                                    local[:, 1],
                                    dist_shifted[:, 0],
                                    z=int(z),
                                    match_tol_ppm=precursor_tol_ppm,
                                    peak_mz=obs_mz_seed,
                                )
                                isodec_css = css_similarity(
                                    y_obs_seed, dist_shifted[:, 1]
                                )
                                accepted_model = isodec_css >= float(cfg.MIN_COSINE)

                            obs_idx_eval = int(
                                nearest_peak_index(local[:, 0], obs_mz_eval)
                            )
                            if obs_idx_eval >= 0:
                                obs_anchor_int = float(local[obs_idx_eval, 1])
                            anchor_rel_int = (
                                (float(obs_anchor_int) / local_max_int)
                                if local_max_int > 0
                                else 0.0
                            )

                            shift_da = float(obs_mz_eval - theory_mz)
                            matches = _match_theory_local(
                                local, dist_shifted, shift_da, precursor_tol_ppm
                            )
                            anchor_ppm_abs = (
                                abs(
                                    ((float(obs_mz_eval) - theory_mz) / theory_mz) * 1e6
                                )
                                if theory_mz
                                else 0.0
                            )
                            comp = _composite_components(
                                css=float(isodec_css),
                                matches=matches,
                                dist_shifted=dist_shifted,
                                anchor_ppm_abs=float(anchor_ppm_abs),
                                anchor_theory_mz=float(theory_mz),
                                intensity_ratio=float(anchor_rel_int),
                            )
                            ppm = (
                                ((float(obs_mz_eval) - theory_mz) / theory_mz) * 1e6
                                if theory_mz
                                else 0.0
                            )
                            accepted = bool(
                                bool(accepted_model)
                                and float(isodec_css) >= float(cfg.MIN_COSINE)
                                and float(anchor_rel_int) >= float(anchor_min_rel_int)
                                and float(comp["coverage"]) >= float(min_coverage)
                                and float(anchor_ppm_abs) <= float(max_anchor_abs_ppm)
                                and len(matches) > 0
                            )
                            if len(matches) >= 2 and np.isfinite(comp["ppm_rmse"]):
                                accepted = accepted and float(
                                    comp["ppm_rmse"]
                                ) <= float(max_residual_rmse_ppm)
                            if not accepted:
                                continue

                            dist_plot = _scale_dist_to_obs(dist_shifted, obs_anchor_int)
                            candidate = {
                                "charge": int(z),
                                "state": str(state),
                                "obs_mz": float(obs_mz_eval),
                                "anchor_theory_mz": theory_mz,
                                "ppm": float(ppm),
                                "css": float(isodec_css),
                                "accepted": True,
                                "iteration": 1,
                                "dist": dist_plot,
                                "obs_anchor_int": float(obs_anchor_int),
                                "score": float(comp["composite_score"]),
                                "composite_score": float(comp["composite_score"]),
                                "coverage": float(comp["coverage"]),
                                "ppm_rmse": float(comp["ppm_rmse"]),
                                "ppm_consistency": float(comp["ppm_consistency"]),
                                "spacing_consistency": float(
                                    comp["spacing_consistency"]
                                ),
                                "spacing_rmse_da": float(comp["spacing_rmse_da"]),
                                "intensity_prior": float(comp["intensity_prior"]),
                                "anchor_rel_int": float(anchor_rel_int),
                                "match_count": int(len(matches)),
                                "anchor_seed_mz": float(obs_mz_seed),
                                "anchor_target_mz": float(theory_mz),
                                "local_window_min": float(obs_anchor_mz + local_lb),
                                "local_window_max": float(obs_anchor_mz + local_ub),
                            }
                            if local_best_for_state is None or float(
                                candidate["composite_score"]
                            ) > float(
                                local_best_for_state.get("composite_score", -1.0)
                            ):
                                local_best_for_state = candidate

                    return local_best_for_state

                best_candidate = execute_hybrid_strategy(_score_precursor_candidate)
                if best_candidate is None:
                    continue
                all_candidates.append(best_candidate)
                prev = best_by_charge.get(int(z))
                prev_score = (
                    float(prev.get("composite_score", prev.get("css", -1.0)))
                    if prev is not None
                    else -1.0
                )
                if (
                    prev is None
                    or float(best_candidate.get("composite_score", 0.0)) > prev_score
                ):
                    best_by_charge[int(z)] = best_candidate

    accepted_ranked = sorted(
        all_candidates,
        key=lambda d: float(d.get("composite_score", d.get("css", -1.0))),
        reverse=True,
    )
    candidates = sorted(
        best_by_charge.values(),
        key=lambda d: float(d.get("composite_score", d.get("css", -1.0))),
        reverse=True,
    )

    if accepted_ranked:
        top_candidate = dict(accepted_ranked[0])
        if first_anchor_int is None:
            try:
                first_anchor_int = float(top_candidate.get("obs_anchor_int", 0.0))
            except Exception:
                first_anchor_int = 0.0
        if ambiguity_guard and len(accepted_ranked) >= 2:
            gap = float(accepted_ranked[0].get("composite_score", 0.0)) - float(
                accepted_ranked[1].get("composite_score", 0.0)
            )
            if gap < float(ambiguity_margin):
                top_candidates = [
                    dict(c) for c in accepted_ranked[: min(3, len(accepted_ranked))]
                ]
                ambiguous_result = {
                    "status": "ambiguous",
                    "accepted": False,
                    "ambiguous": True,
                    "iteration": 1,
                    "window_min": float(
                        top_candidates[0].get(
                            "local_window_min",
                            top_candidates[0].get("anchor_theory_mz", 0.0),
                        )
                    ),
                    "window_max": float(
                        top_candidates[0].get(
                            "local_window_max",
                            top_candidates[0].get("anchor_theory_mz", 0.0),
                        )
                    ),
                    "score": float(top_candidates[0].get("composite_score", 0.0)),
                    "css": float(top_candidates[0].get("css", 0.0)),
                    "top_candidates": top_candidates,
                }
                print(
                    "Precursor search ambiguous: "
                    f"top score={float(ambiguous_result.get('score', 0.0)):.3f}; "
                    "stopping without calibration."
                )
            else:
                match_found = True
        else:
            match_found = True

        if match_found:
            best_candidate = top_candidate
            best_z = int(best_candidate["charge"])
            best_state = best_candidate.get("state")
            best_css = float(best_candidate["css"])
            best_composite_score = float(
                best_candidate.get("composite_score", best_css)
            )
            best_coverage = float(best_candidate.get("coverage", 0.0) or 0.0)
            best_ppm_rmse = (
                float(best_candidate.get("ppm_rmse"))
                if best_candidate.get("ppm_rmse") is not None
                else None
            )
            best_obs_mz = float(best_candidate["obs_mz"])
            best_theory_mz = float(best_candidate["anchor_theory_mz"])
            best_theory_dist = best_candidate["dist"]
            shift_ppm = (
                ((best_obs_mz - best_theory_mz) / best_theory_mz) * 1e6
                if best_theory_mz
                else 0.0
            )
            state_label = f" ({best_state})" if best_state and best_state != "0" else ""
            strat = best_candidate.get("strategy")
            strat_label = f" [{strat}]" if strat else ""
            print(
                f"Precursor found: z={best_z}+{state_label}{strat_label} "
                f"m/z={best_obs_mz:.4f} css={best_css:.3f} "
                f"score={float(best_candidate.get('composite_score', best_css)):.3f}"
            )

    calibrated_spectrum = np.array(spectrum, dtype=float, copy=True)
    calibration_applied = False
    calibration_safe = False
    calibration_block_reasons: list[str] = []
    if match_found:
        min_calibration_score = float(
            getattr(cfg, "PRECURSOR_CALIBRATION_MIN_SCORE", 0.70)
        )
        min_calibration_coverage = float(
            getattr(
                cfg,
                "PRECURSOR_CALIBRATION_MIN_COVERAGE",
                getattr(cfg, "PRECURSOR_MIN_COVERAGE", 0.30),
            )
        )
        max_calibration_ppm_rmse = float(
            getattr(
                cfg,
                "PRECURSOR_CALIBRATION_MAX_PPM_RMSE",
                getattr(cfg, "PRECURSOR_MAX_RESIDUAL_RMSE_PPM", precursor_tol_ppm),
            )
        )
        max_calibration_shift_ppm = float(
            getattr(cfg, "PRECURSOR_CALIBRATION_MAX_SHIFT_PPM", 100.0)
        )
        if float(best_composite_score) < float(min_calibration_score):
            calibration_block_reasons.append(f"score<{min_calibration_score:.2f}")
        if float(best_coverage) < float(min_calibration_coverage):
            calibration_block_reasons.append(f"coverage<{min_calibration_coverage:.2f}")
        if (
            best_ppm_rmse is not None
            and np.isfinite(best_ppm_rmse)
            and float(best_ppm_rmse) > float(max_calibration_ppm_rmse)
        ):
            calibration_block_reasons.append(f"ppm_rmse>{max_calibration_ppm_rmse:.1f}")
        if abs(float(shift_ppm)) > float(max_calibration_shift_ppm):
            calibration_block_reasons.append(
                f"abs_shift>{max_calibration_shift_ppm:.1f}ppm"
            )
        calibration_safe = len(calibration_block_reasons) == 0

    if match_found and bool(apply_calibration) and calibration_safe:
        print(f"Applying lock-mass calibration: shift {-shift_ppm:.2f} ppm")
        calibrated_spectrum[:, 0] = calibrated_spectrum[:, 0] / (
            1.0 + (shift_ppm / 1e6)
        )
        calibration_applied = True
    elif match_found and bool(apply_calibration) and not calibration_safe:
        print(
            "Precursor matched, but calibration was skipped because "
            + ", ".join(calibration_block_reasons)
            + "."
        )
    elif not match_found:
        print("Precursor not found; no calibration applied.")

    if not match_found and all_theory_dist is not None:
        if first_anchor_int is not None:
            all_theory_dist = _scale_dist_to_obs(
                all_theory_dist, float(first_anchor_int)
            )
        best_theory_dist = all_theory_dist

    candidates = sorted(
        best_by_charge.values(),
        key=lambda d: float(d.get("composite_score", d.get("css", -1.0))),
        reverse=True,
    )
    plot_window = None
    if (
        match_found
        and best_theory_dist is not None
        and isinstance(best_theory_dist, np.ndarray)
        and best_theory_dist.size
    ):
        plot_window = (
            float(best_theory_dist[0, 0]) - 5.0,
            float(best_theory_dist[-1, 0]) + 5.0,
        )

    return {
        "spectrum": calibrated_spectrum,
        "spectrum_mz": np.asarray(calibrated_spectrum[:, 0], dtype=float),
        "spectrum_int": np.asarray(calibrated_spectrum[:, 1], dtype=float),
        "match_found": bool(match_found),
        "search_status": "matched"
        if match_found
        else ("ambiguous" if ambiguous_result is not None else "not_found"),
        "best_z": best_z,
        "best_state": best_state,
        "best_css": float(best_css),
        "best_composite_score": float(best_composite_score),
        "best_coverage": float(best_coverage),
        "best_ppm_rmse": best_ppm_rmse,
        "shift_ppm": float(shift_ppm),
        "precursor_tol_ppm": float(precursor_tol_ppm),
        "calibration_requested": bool(apply_calibration),
        "calibration_applied": bool(calibration_applied),
        "calibration_safe": bool(calibration_safe),
        "calibration_block_reasons": calibration_block_reasons,
        "best_obs_mz": best_obs_mz,
        "best_theory_mz": best_theory_mz,
        "best_theory_dist": best_theory_dist,
        "plot_window": plot_window,
        "search_window": search_window,
        "ambiguous_window": (
            {
                "iteration": int(ambiguous_result.get("iteration", 0) or 0),
                "min": float(ambiguous_result.get("window_min")),
                "max": float(ambiguous_result.get("window_max")),
                "strategy": ambiguous_result.get("strategy"),
            }
            if ambiguous_result is not None
            else None
        ),
        "ambiguous_candidates": list(ambiguous_result.get("top_candidates") or [])
        if ambiguous_result is not None
        else [],
        "candidates": candidates,
    }


def run_precursor_mode(residues, spectrum, isodec_config) -> np.ndarray:
    result = run_precursor_headless(residues, spectrum, isodec_config)
    calibrated_spectrum = np.asarray(result.get("spectrum"), dtype=float)
    match_found = bool(result.get("match_found"))
    best_theory_dist = result.get("best_theory_dist")
    best_z = result.get("best_z")
    best_state = result.get("best_state")
    best_css = float(result.get("best_css", 0.0) or 0.0)
    shift_ppm = float(result.get("shift_ppm", 0.0) or 0.0)

    if isinstance(best_theory_dist, np.ndarray) and best_theory_dist.size:
        if match_found:
            state_label = f" ({best_state})" if best_state and best_state != "0" else ""
            label = f"Precursor {best_z}+{state_label} (css={best_css:.3f}, shift {-shift_ppm:.2f} ppm)"
            color = "tab:red"
            window = result.get("plot_window")
            mz_min = float(window[0]) if window else float(best_theory_dist[0, 0]) - 5.0
            mz_max = (
                float(window[1]) if window else float(best_theory_dist[-1, 0]) + 5.0
            )
        else:
            label = f"Precursor theories (z={int(cfg.PRECURSOR_MIN_CHARGE)}-{int(cfg.PRECURSOR_MAX_CHARGE)})"
            color = "tab:gray"
            mz_min = None
            mz_max = None
        plot_overlay(
            calibrated_spectrum,
            [(best_theory_dist, color, label)],
            mz_min=mz_min,
            mz_max=mz_max,
        )

    return calibrated_spectrum


def _charge_reduced_state_shifts() -> list[tuple[str, int]]:
    if not bool(getattr(cfg, "ENABLE_H_TRANSFER", False)):
        return [("0", 0)]
    return [
        ("0", 0),
        ("+H", 1),
        ("+2H", 2),
        ("-H", -1),
        ("-2H", -2),
    ]


def _filter_spectrum_by_requested_mz_window(
    spectrum: np.ndarray,
) -> tuple[np.ndarray, dict | None]:
    work_spectrum = np.asarray(spectrum, dtype=float)
    search_mz_min = None if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
    search_mz_max = None if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
    if search_mz_min is None and search_mz_max is None:
        return work_spectrum, None

    lower = -np.inf if search_mz_min is None else float(search_mz_min)
    upper = np.inf if search_mz_max is None else float(search_mz_max)
    keep = (work_spectrum[:, 0] >= lower) & (work_spectrum[:, 0] <= upper)
    return work_spectrum[keep], {
        "min": None if search_mz_min is None else float(search_mz_min),
        "max": None if search_mz_max is None else float(search_mz_max),
    }


def _match_theory_local_monotonic(
    local: np.ndarray,
    dist_shifted: np.ndarray,
    shift_da: float,
    tol_ppm: float,
) -> list[dict]:
    local_mz = np.asarray(local[:, 0], dtype=float)
    local_int = np.asarray(local[:, 1], dtype=float)
    theory_mz = np.asarray(dist_shifted[:, 0], dtype=float)
    theory_int = np.asarray(dist_shifted[:, 1], dtype=float)
    pred_mz = theory_mz + float(shift_da)

    used = np.zeros(local_mz.shape[0], dtype=bool)
    last_local_idx = -1
    rows: list[dict] = []
    for i in range(pred_mz.shape[0]):
        mz_pred = float(pred_mz[i])
        delta = abs(mz_pred) * float(tol_ppm) * 1e-6 if mz_pred > 0 else 0.0
        low = mz_pred - delta
        high = mz_pred + delta
        start_idx = int(np.searchsorted(local_mz, low, side="left"))
        end_idx = int(np.searchsorted(local_mz, high, side="right"))
        if end_idx <= start_idx:
            continue
        start_idx = max(start_idx, last_local_idx + 1)
        if end_idx <= start_idx:
            continue
        cands = np.arange(start_idx, end_idx, dtype=int)
        if cands.size == 0:
            continue
        cands = cands[~used[cands]]
        if cands.size == 0:
            continue
        pick = int(cands[np.argmin(np.abs(local_mz[cands] - mz_pred))])
        used[pick] = True
        last_local_idx = pick
        mz_obs = float(local_mz[pick])
        mz_theory = float(theory_mz[i])
        residual_da = mz_obs - mz_pred
        residual_ppm = (residual_da / mz_theory) * 1e6 if mz_theory else 0.0
        rows.append(
            {
                "theory_idx": int(i),
                "theory_mz": mz_theory,
                "theory_int": float(theory_int[i]),
                "pred_mz": mz_pred,
                "obs_idx": int(pick),
                "obs_mz": mz_obs,
                "obs_int": float(local_int[pick]),
                "residual_da": float(residual_da),
                "residual_ppm": float(residual_ppm),
            }
        )
    return rows


def _composite_match_components(
    *,
    css: float,
    matches: list[dict],
    dist_shifted: np.ndarray,
    anchor_ppm_abs: float,
    anchor_theory_mz: float,
    intensity_ratio: float,
    ppm_sigma: float,
    spacing_sigma_da: float,
    score_w_css: float,
    score_w_cov: float,
    score_w_ppm: float,
    score_w_spacing: float,
    score_w_intensity: float,
) -> dict:
    css_val = float(np.clip(css, 0.0, 1.0))
    theory_int = np.asarray(dist_shifted[:, 1], dtype=float)
    theory_sum = float(np.sum(theory_int)) if theory_int.size else 0.0
    if theory_sum > 0 and matches:
        matched_sum = float(
            sum(max(0.0, float(r.get("theory_int", 0.0))) for r in matches)
        )
        coverage = float(np.clip(matched_sum / theory_sum, 0.0, 1.0))
    else:
        coverage = 0.0

    if matches:
        weights = np.array(
            [max(float(r.get("theory_int", 0.0)), 1e-12) for r in matches], dtype=float
        )
        residual_ppm = np.array(
            [float(r.get("residual_ppm", 0.0)) for r in matches], dtype=float
        )
        ppm_rmse = float(
            np.sqrt(np.average(residual_ppm * residual_ppm, weights=weights))
        )
    else:
        ppm_rmse = float("inf")

    residual_score = (
        0.0 if not np.isfinite(ppm_rmse) else float(np.exp(-ppm_rmse / ppm_sigma))
    )
    anchor_score = float(np.exp(-abs(anchor_ppm_abs) / ppm_sigma))
    ppm_consistency = float(
        np.clip((0.70 * residual_score) + (0.30 * anchor_score), 0.0, 1.0)
    )

    if len(matches) >= 2:
        obs = np.array([float(r.get("obs_mz", 0.0)) for r in matches], dtype=float)
        pred = np.array([float(r.get("pred_mz", 0.0)) for r in matches], dtype=float)
        diff_err = np.diff(obs) - np.diff(pred)
        spacing_rmse_da = (
            float(np.sqrt(np.mean(diff_err * diff_err))) if diff_err.size else 0.0
        )
        spacing_consistency = float(np.exp(-spacing_rmse_da / spacing_sigma_da))
    else:
        spacing_rmse_da = float("inf")
        spacing_consistency = 0.5

    intensity_prior = float(np.sqrt(np.clip(float(intensity_ratio), 0.0, 1.0)))
    composite_score = (
        score_w_css * css_val
        + score_w_cov * coverage
        + score_w_ppm * ppm_consistency
        + score_w_spacing * spacing_consistency
        + score_w_intensity * intensity_prior
    )
    return {
        "coverage": float(coverage),
        "ppm_rmse": float(ppm_rmse),
        "ppm_consistency": float(ppm_consistency),
        "spacing_rmse_da": float(spacing_rmse_da),
        "spacing_consistency": float(spacing_consistency),
        "intensity_prior": float(intensity_prior),
        "composite_score": float(composite_score),
    }


def _fragment_noise_core_components(
    *,
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    local: np.ndarray,
    matches: list[dict],
    dist_shifted: np.ndarray,
    core_top_n: int,
    base_score: float,
    ppm_sigma: float,
    anchor_mz: float,
    noise_model: dict | None,
    score_w_fit: float,
    score_w_correlation: float,
    score_w_snr: float,
    s2n_scale: float,
    penalty_unexplained: float,
    penalty_missing_core: float,
    penalty_missing_peaks: float,
    penalty_mass_error_std: float,
) -> dict:
    local_int = (
        np.asarray(local[:, 1], dtype=float)
        if isinstance(local, np.ndarray) and local.size
        else np.asarray([], dtype=float)
    )
    local_total = float(np.sum(local_int)) if local_int.size else 0.0
    matched_obs_indices = sorted(
        {
            int(r.get("obs_idx"))
            for r in matches
            if r.get("obs_idx") is not None and int(r.get("obs_idx")) >= 0
        }
    )
    explained_obs_sum = (
        float(
            sum(
                float(local_int[idx])
                for idx in matched_obs_indices
                if idx < local_int.shape[0]
            )
        )
        if local_total > 0
        else 0.0
    )
    local_explained_fraction = (
        float(np.clip(explained_obs_sum / local_total, 0.0, 1.0))
        if local_total > 0
        else 0.0
    )
    unexplained_fraction = float(np.clip(1.0 - local_explained_fraction, 0.0, 1.0))

    theory_int = (
        np.asarray(dist_shifted[:, 1], dtype=float)
        if isinstance(dist_shifted, np.ndarray) and dist_shifted.size
        else np.asarray([], dtype=float)
    )
    positive_idx = np.where(theory_int > 0)[0]
    core_n = max(1, int(core_top_n))
    if positive_idx.size:
        ranked = positive_idx[np.argsort(theory_int[positive_idx])[::-1]]
        core_idx = ranked[:core_n]
    else:
        core_idx = np.asarray([], dtype=int)
    matched_theory = {
        int(r.get("theory_idx"))
        for r in matches
        if r.get("theory_idx") is not None and int(r.get("theory_idx")) >= 0
    }
    if core_idx.size:
        core_sum = float(np.sum(theory_int[core_idx]))
        matched_core_sum = float(
            sum(float(theory_int[idx]) for idx in core_idx if idx in matched_theory)
        )
        core_coverage = (
            float(np.clip(matched_core_sum / core_sum, 0.0, 1.0))
            if core_sum > 0
            else 0.0
        )
    else:
        core_coverage = 0.0
    missing_core_fraction = float(np.clip(1.0 - core_coverage, 0.0, 1.0))

    total_theory_peaks = int(theory_int.size)
    matched_by_theory_idx = {
        int(r.get("theory_idx")): r
        for r in matches
        if r.get("theory_idx") is not None and int(r.get("theory_idx")) >= 0
    }
    obs_aligned = np.zeros(total_theory_peaks, dtype=float)
    matched_bool = np.zeros(total_theory_peaks, dtype=int)
    for theory_idx, row in matched_by_theory_idx.items():
        if 0 <= theory_idx < total_theory_peaks:
            obs_aligned[theory_idx] = max(float(row.get("obs_int", 0.0)), 0.0)
            matched_bool[theory_idx] = 1

    num_missing_peaks = max(total_theory_peaks - int(np.sum(matched_bool)), 0)
    pc_missing_peaks = (
        float((num_missing_peaks / total_theory_peaks) * 100.0)
        if total_theory_peaks > 0
        else 100.0
    )

    scaled_theory, theory_scale = _scale_theoretical_to_observed(
        theory_int, obs_aligned
    )
    fit_score = _fit_score_from_envelope(obs_aligned, scaled_theory)
    correlation_coefficient = _safe_pearson(obs_aligned, scaled_theory)
    chisq_stat = _chisq_statistic(obs_aligned, scaled_theory)

    if matches:
        residual_ppm = np.asarray(
            [float(r.get("residual_ppm", 0.0)) for r in matches], dtype=float
        )
        mass_error_std = (
            float(np.std(residual_ppm)) if residual_ppm.size else float("inf")
        )
    else:
        mass_error_std = float("inf")

    mz_min = (
        float(np.min(dist_shifted[:, 0]))
        if isinstance(dist_shifted, np.ndarray) and dist_shifted.size
        else float(anchor_mz)
    )
    mz_max = (
        float(np.max(dist_shifted[:, 0]))
        if isinstance(dist_shifted, np.ndarray) and dist_shifted.size
        else float(anchor_mz)
    )
    env_mask = (np.asarray(spectrum_mz, dtype=float) >= mz_min) & (
        np.asarray(spectrum_mz, dtype=float) <= mz_max
    )
    env_total_signal = (
        float(np.sum(np.asarray(spectrum_int, dtype=float)[env_mask]))
        if np.any(env_mask)
        else 0.0
    )
    target_signal = float(np.sum(obs_aligned))
    if env_total_signal > 0.0:
        interference = float(
            np.clip(1.0 - (target_signal / env_total_signal), 0.0, 1.0)
        )
    else:
        interference = float(unexplained_fraction)

    local_noise = _evaluate_noise_level(noise_model, float(anchor_mz))
    if local_int.size:
        try:
            local_q = float(np.quantile(local_int, 0.20))
            if np.isfinite(local_q) and local_q > 0:
                local_noise = max(local_noise, local_q)
        except Exception:
            pass
    max_signal = (
        float(np.max(obs_aligned))
        if obs_aligned.size
        else (float(np.max(local_int)) if local_int.size else 0.0)
    )
    s2n = float(max_signal / local_noise) if local_noise > 0.0 else float("inf")
    log_s2n = (
        float(np.log10(max(s2n, 1e-12)))
        if np.isfinite(s2n) and s2n > 0.0
        else float("-inf")
    )

    fit_bonus = float(score_w_fit) * float(np.clip(fit_score, 0.0, 1.0))
    correlation_bonus = float(score_w_correlation) * float(
        np.clip((correlation_coefficient + 1.0) / 2.0, 0.0, 1.0)
    )
    s2n_scale_val = max(float(s2n_scale), 1e-6)
    s2n_score = (
        1.0
        if not np.isfinite(s2n)
        else float(1.0 - np.exp(-max(s2n, 0.0) / s2n_scale_val))
    )
    s2n_bonus = float(score_w_snr) * float(np.clip(s2n_score, 0.0, 1.0))
    ppm_sigma_val = max(float(ppm_sigma), 1e-6)
    mass_error_penalty = (
        1.0
        if not np.isfinite(mass_error_std)
        else float(1.0 - np.exp(-max(mass_error_std, 0.0) / ppm_sigma_val))
    )

    evidence_score = (
        float(base_score)
        + fit_bonus
        + correlation_bonus
        + s2n_bonus
        - (float(penalty_unexplained) * unexplained_fraction)
        - (float(penalty_missing_core) * missing_core_fraction)
        - (float(penalty_missing_peaks) * (pc_missing_peaks / 100.0))
        - (float(penalty_mass_error_std) * mass_error_penalty)
    )
    return {
        "local_explained_fraction": float(local_explained_fraction),
        "unexplained_fraction": float(unexplained_fraction),
        "core_coverage": float(core_coverage),
        "missing_core_fraction": float(missing_core_fraction),
        "interference": float(interference),
        "num_missing_peaks": int(num_missing_peaks),
        "pc_missing_peaks": float(pc_missing_peaks),
        "fit_score": float(fit_score),
        "correlation_coefficient": float(correlation_coefficient),
        "chisq_stat": float(chisq_stat),
        "mass_error_std": float(mass_error_std),
        "noise_level": float(local_noise),
        "s2n": float(s2n),
        "log_s2n": float(log_s2n),
        "theory_scale": float(theory_scale),
        "evidence_score": float(evidence_score),
    }


def _fragment_isodec_detail(
    *,
    local_centroids: np.ndarray,
    dist_plot: np.ndarray,
    z: int,
    isodec_config,
) -> dict:
    detail = {
        "matched_peaks_n": 0,
        "minpeaks_effective": int(getattr(isodec_config, "minpeaks", 0) or 0)
        if isodec_config is not None
        else 0,
        "areacovered": 0.0,
        "topthree": False,
    }
    if (
        cfg.isodec_find_matches is None
        or not isinstance(local_centroids, np.ndarray)
        or local_centroids.size == 0
        or not isinstance(dist_plot, np.ndarray)
        or dist_plot.size == 0
        or isodec_config is None
    ):
        return detail

    matchedindexes, isomatches = cfg.isodec_find_matches(
        local_centroids,
        dist_plot,
        float(isodec_config.matchtol),
    )
    matched_peaks_n = int(len(matchedindexes))
    minpeaks_eff = int(getattr(isodec_config, "minpeaks", 0) or 0)

    if int(z) == 1 and matched_peaks_n == 2 and len(isomatches) == 2:
        if isomatches[0] == 0 and isomatches[1] == 1:
            int1 = float(local_centroids[matchedindexes[0], 1])
            int2 = float(local_centroids[matchedindexes[1], 1])
            ratio = (int2 / int1) if int1 != 0 else 0.0
            if (
                float(isodec_config.plusoneintwindowlb)
                < ratio
                < float(isodec_config.plusoneintwindowub)
            ):
                minpeaks_eff = 2

    mi = (
        np.array(matchedindexes, dtype=int)
        if len(matchedindexes)
        else np.array([], dtype=int)
    )
    ii = np.array(isomatches, dtype=int) if len(isomatches) else np.array([], dtype=int)
    matchedcentroids = local_centroids[mi] if mi.size else np.empty((0, 2), dtype=float)
    matchediso = dist_plot[ii] if ii.size else np.empty((0, 2), dtype=float)

    areacovered = (
        float(np.sum(matchediso[:, 1])) / float(np.sum(local_centroids[:, 1]))
        if local_centroids.size and np.sum(local_centroids[:, 1]) > 0
        else 0.0
    )
    topn = int(minpeaks_eff)
    topthree = False
    if local_centroids.size and matchedcentroids.size and topn > 0:
        top_iso = np.sort(matchedcentroids[:, 1])[::-1][:topn]
        top_cent = np.sort(local_centroids[:, 1])[::-1][:topn]
        topthree = bool(np.array_equal(top_iso, top_cent))

    detail.update(
        {
            "matched_peaks_n": matched_peaks_n,
            "minpeaks_effective": int(minpeaks_eff),
            "areacovered": float(areacovered),
            "topthree": bool(topthree),
        }
    )
    return detail


def _fragment_truth_score(
    *,
    correlation_coefficient: float,
    pc_missing_peaks: float,
    isodec_css: float,
    top_peaks: bool | float | int,
) -> tuple[float | None, float | None]:
    values = [
        correlation_coefficient,
        pc_missing_peaks,
        isodec_css,
        float(top_peaks) if top_peaks is not None else None,
    ]
    if any(value is None or not np.isfinite(float(value)) for value in values):
        return None, None

    z_corr = (float(correlation_coefficient) - 0.7568201889847942) / 0.24197271755281535
    z_pc_missing = (28.699328693056465 - float(pc_missing_peaks)) / 24.234497548756174
    z_css = (float(isodec_css) - 0.8372419143663081) / 0.14986324501779563
    z_top = (float(top_peaks) - 0.5159235668789809) / 0.5005440559698912

    logit = (
        (0.872 * z_corr)
        + (0.933 * z_pc_missing)
        + (0.781 * z_css)
        + (0.503 * z_top)
        + 0.520
    )
    clipped = float(np.clip(logit, -40.0, 40.0))
    prob = float(1.0 / (1.0 + np.exp(-clipped)))
    return clipped, prob


def _fragment_scoring_settings(match_tol_ppm: float | None = None) -> dict:
    match_tol_ppm = float(
        cfg.MATCH_TOL_PPM if match_tol_ppm is None else match_tol_ppm
    )
    max_anchor_abs_ppm_cfg = getattr(cfg, "FRAG_MAX_ANCHOR_ABS_PPM", None)
    max_anchor_abs_ppm = (
        float(max_anchor_abs_ppm_cfg)
        if max_anchor_abs_ppm_cfg is not None
        else (match_tol_ppm * 1.5)
    )
    ppm_sigma_cfg = getattr(cfg, "FRAG_PPM_SIGMA", None)
    ppm_sigma = (
        float(ppm_sigma_cfg) if ppm_sigma_cfg is not None else float(match_tol_ppm)
    )
    if ppm_sigma <= 0:
        ppm_sigma = max(match_tol_ppm, 1.0)

    score_w_css = float(getattr(cfg, "FRAG_SCORE_W_CSS", 0.40))
    score_w_cov = float(getattr(cfg, "FRAG_SCORE_W_COVERAGE", 0.20))
    score_w_ppm = float(getattr(cfg, "FRAG_SCORE_W_PPM", 0.15))
    score_w_spacing = float(getattr(cfg, "FRAG_SCORE_W_SPACING", 0.10))
    score_w_intensity = float(getattr(cfg, "FRAG_SCORE_W_INTENSITY", 0.15))
    score_w_fit = float(getattr(cfg, "FRAG_SCORE_W_FIT", 0.10))
    score_w_correlation = float(getattr(cfg, "FRAG_SCORE_W_CORRELATION", 0.05))
    score_w_snr = float(getattr(cfg, "FRAG_SCORE_W_SNR", 0.05))
    score_w_sum = (
        score_w_css
        + score_w_cov
        + score_w_ppm
        + score_w_spacing
        + score_w_intensity
        + score_w_fit
        + score_w_correlation
        + score_w_snr
    )
    if score_w_sum <= 0:
        score_w_css, score_w_cov, score_w_ppm, score_w_spacing, score_w_intensity = (
            0.40,
            0.20,
            0.15,
            0.10,
            0.15,
        )
        score_w_fit, score_w_correlation, score_w_snr = 0.10, 0.05, 0.05
    else:
        score_w_css /= score_w_sum
        score_w_cov /= score_w_sum
        score_w_ppm /= score_w_sum
        score_w_spacing /= score_w_sum
        score_w_intensity /= score_w_sum
        score_w_fit /= score_w_sum
        score_w_correlation /= score_w_sum
        score_w_snr /= score_w_sum

    return {
        "min_matched_peaks": max(1, int(getattr(cfg, "FRAG_MIN_MATCHED_PEAKS", 2))),
        "max_anchor_abs_ppm": float(max_anchor_abs_ppm),
        "ppm_sigma": float(ppm_sigma),
        "core_top_n": max(1, int(getattr(cfg, "FRAG_CORE_TOP_N", 3))),
        "min_isodec_css": float(getattr(cfg, "FRAG_MIN_ISODEC_CSS", cfg.MIN_COSINE)),
        "max_pc_missing_peaks": float(
            getattr(cfg, "FRAG_MAX_PC_MISSING_PEAKS", 85.0)
        ),
        "min_fit_score": float(getattr(cfg, "FRAG_MIN_FIT_SCORE", 0.35)),
        "min_correlation": (
            float(getattr(cfg, "FRAG_MIN_CORRELATION", None))
            if getattr(cfg, "FRAG_MIN_CORRELATION", None) is not None
            else None
        ),
        "noise_model_splits": max(4, int(getattr(cfg, "FRAG_NOISE_MODEL_SPLITS", 50))),
        "noise_hist_bins": max(16, int(getattr(cfg, "FRAG_NOISE_HIST_BINS", 128))),
        "s2n_score_scale": max(float(getattr(cfg, "FRAG_S2N_SCORE_SCALE", 4.0)), 1e-6),
        "score_w_css": float(score_w_css),
        "score_w_cov": float(score_w_cov),
        "score_w_ppm": float(score_w_ppm),
        "score_w_spacing": float(score_w_spacing),
        "score_w_intensity": float(score_w_intensity),
        "score_w_fit": float(score_w_fit),
        "score_w_correlation": float(score_w_correlation),
        "score_w_snr": float(score_w_snr),
        "score_penalty_unexplained": float(
            getattr(cfg, "FRAG_SCORE_PENALTY_UNEXPLAINED", 0.35)
        ),
        "score_penalty_missing_core": float(
            getattr(cfg, "FRAG_SCORE_PENALTY_MISSING_CORE", 0.25)
        ),
        "score_penalty_missing_peaks": float(
            getattr(cfg, "FRAG_SCORE_PENALTY_MISSING_PEAKS", 0.10)
        ),
        "score_penalty_mass_error_std": float(
            getattr(cfg, "FRAG_SCORE_PENALTY_MASS_ERROR_STD", 0.10)
        ),
        "truth_score_enabled": bool(getattr(cfg, "FRAG_TRUTH_SCORE_ENABLE", False)),
        "truth_score_threshold": float(
            getattr(cfg, "FRAG_TRUTH_SCORE_THRESHOLD", 0.85)
        ),
        "use_truth_score_for_ranking": bool(
            getattr(cfg, "FRAG_TRUTH_SCORE_USE_FOR_RANKING", True)
        ),
    }


def _fragment_gate_from_pipeline(
    *,
    isodec_css: float,
    ppm: float,
    local_match_count: int,
    quality: dict,
    settings: dict,
) -> dict:
    checks = {
        "isodec_css": {
            "value": float(isodec_css),
            "threshold": f">= {float(settings['min_isodec_css']):.6f}",
            "threshold_value": float(settings["min_isodec_css"]),
            "pass": float(isodec_css) >= float(settings["min_isodec_css"]),
            "description": "Cosine similarity score",
        },
        "ppm": {
            "value": float(abs(ppm)),
            "threshold": f"<= {float(settings['max_anchor_abs_ppm']):.1f}",
            "threshold_value": float(settings["max_anchor_abs_ppm"]),
            "pass": float(abs(ppm)) <= float(settings["max_anchor_abs_ppm"]),
            "description": "Anchor mass error (ppm)",
        },
        "local_matches": {
            "value": int(local_match_count),
            "threshold": f">= {int(settings['min_matched_peaks'])}",
            "threshold_value": int(settings["min_matched_peaks"]),
            "pass": int(local_match_count) >= int(settings["min_matched_peaks"]),
            "description": "Number of matched peaks",
        },
        "pc_missing_peaks": {
            "value": float(quality["pc_missing_peaks"]),
            "threshold": f"<= {float(settings['max_pc_missing_peaks']):.1f}",
            "threshold_value": float(settings["max_pc_missing_peaks"]),
            "pass": float(quality["pc_missing_peaks"])
            <= float(settings["max_pc_missing_peaks"]),
            "description": "Percentage of missing peaks",
        },
        "fit_score": {
            "value": float(quality["fit_score"]),
            "threshold": f">= {float(settings['min_fit_score']):.6f}",
            "threshold_value": float(settings["min_fit_score"]),
            "pass": float(quality["fit_score"]) >= float(settings["min_fit_score"]),
            "description": "Envelope fit score",
        },
    }

    min_correlation = settings.get("min_correlation")
    if min_correlation is not None:
        checks["correlation_coefficient"] = {
            "value": float(quality["correlation_coefficient"]),
            "threshold": f">= {float(min_correlation):.6f}",
            "threshold_value": float(min_correlation),
            "pass": float(quality["correlation_coefficient"])
            >= float(min_correlation),
            "description": "Intensity correlation coefficient",
        }

    legacy_accepted = all(check["pass"] for check in checks.values())
    failed_at = None
    for key, check in checks.items():
        if not check["pass"]:
            failed_at = key
            break

    return {
        "legacy_accepted": bool(legacy_accepted),
        "failed_at": failed_at,
        "checks": checks,
        "quality_metrics": {
            "fit_score": float(quality["fit_score"]),
            "correlation_coefficient": float(quality["correlation_coefficient"]),
            "pc_missing_peaks": float(quality["pc_missing_peaks"]),
            "local_matches_count": int(local_match_count),
        },
    }


def _evaluate_fragment_pipeline(
    *,
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    obs_max: float,
    match_tol_ppm: float,
    dist_model: np.ndarray,
    dist_plot: np.ndarray,
    obs_mz: float,
    obs_int: float,
    anchor_theory_mz: float,
    ppm: float,
    isodec_css: float,
    isodec_detail: dict,
    isodec_config,
    use_centroid_logic: bool,
    noise_model: dict | None,
    settings: dict,
) -> dict:
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
        local_mask = (spectrum_mz >= obs_mz + local_lb) & (
            spectrum_mz <= obs_mz + local_ub
        )
        if np.any(local_mask):
            local_centroids = np.column_stack(
                (spectrum_mz[local_mask], spectrum_int[local_mask])
            )
    if (
        not isinstance(local_centroids, np.ndarray)
        or local_centroids.size == 0
        or local_centroids.ndim != 2
        or local_centroids.shape[1] != 2
    ):
        return {
            "pipeline_ready": False,
            "failure_reason": "local_window_empty",
        }

    local_max_int = (
        float(np.max(local_centroids[:, 1])) if local_centroids.size else 0.0
    )
    if local_max_int <= 0.0:
        return {
            "pipeline_ready": False,
            "failure_reason": "local_window_zero_intensity",
        }

    anchor_rel_int = float(np.clip(obs_int / local_max_int, 0.0, 1.0))
    shift_da_local = float(obs_mz - anchor_theory_mz)
    local_matches = _match_theory_local_monotonic(
        local_centroids, dist_model, shift_da_local, match_tol_ppm
    )
    spacing_sigma_da = abs(float(anchor_theory_mz)) * float(match_tol_ppm) * 1e-6
    if spacing_sigma_da <= 0:
        spacing_sigma_da = max(
            abs(float(anchor_theory_mz)) * float(match_tol_ppm) * 1e-6, 1e-6
        )
    comp = _composite_match_components(
        css=float(isodec_css),
        matches=local_matches,
        dist_shifted=dist_model,
        anchor_ppm_abs=abs(float(ppm)),
        anchor_theory_mz=float(anchor_theory_mz),
        intensity_ratio=float(anchor_rel_int),
        ppm_sigma=float(settings["ppm_sigma"]),
        spacing_sigma_da=float(spacing_sigma_da),
        score_w_css=float(settings["score_w_css"]),
        score_w_cov=float(settings["score_w_cov"]),
        score_w_ppm=float(settings["score_w_ppm"]),
        score_w_spacing=float(settings["score_w_spacing"]),
        score_w_intensity=float(settings["score_w_intensity"]),
    )
    quality = _fragment_noise_core_components(
        spectrum_mz=spectrum_mz,
        spectrum_int=spectrum_int,
        local=local_centroids,
        matches=local_matches,
        dist_shifted=dist_plot,
        core_top_n=int(settings["core_top_n"]),
        base_score=float(comp["composite_score"]),
        ppm_sigma=float(settings["ppm_sigma"]),
        anchor_mz=float(obs_mz),
        noise_model=noise_model,
        score_w_fit=float(settings["score_w_fit"]),
        score_w_correlation=float(settings["score_w_correlation"]),
        score_w_snr=float(settings["score_w_snr"]),
        s2n_scale=float(settings["s2n_score_scale"]),
        penalty_unexplained=float(settings["score_penalty_unexplained"]),
        penalty_missing_core=float(settings["score_penalty_missing_core"]),
        penalty_missing_peaks=float(settings["score_penalty_missing_peaks"]),
        penalty_mass_error_std=float(settings["score_penalty_mass_error_std"]),
    )
    gate = _fragment_gate_from_pipeline(
        isodec_css=float(isodec_css),
        ppm=float(ppm),
        local_match_count=int(len(local_matches)),
        quality=quality,
        settings=settings,
    )
    truth_score_logit, truth_score = _fragment_truth_score(
        correlation_coefficient=float(quality["correlation_coefficient"]),
        pc_missing_peaks=float(quality["pc_missing_peaks"]),
        isodec_css=float(isodec_css),
        top_peaks=bool(isodec_detail.get("topthree", False)),
    )
    truth_score_accepted = bool(
        settings["truth_score_enabled"]
        and truth_score is not None
        and float(truth_score) >= float(settings["truth_score_threshold"])
    )
    accepted = bool(gate["legacy_accepted"] or truth_score_accepted)
    selection_score = (
        float(truth_score)
        if settings["truth_score_enabled"]
        and settings["use_truth_score_for_ranking"]
        and truth_score is not None
        else float(quality["evidence_score"])
    )
    gate["truth_score_accepted"] = bool(truth_score_accepted)
    gate["accepted"] = bool(accepted)
    gate["selection_score"] = float(selection_score)

    return {
        "pipeline_ready": True,
        "local_centroids": local_centroids,
        "local_matches": local_matches,
        "comp": comp,
        "quality": quality,
        "fragments_gate": gate,
        "legacy_accepted": bool(gate["legacy_accepted"]),
        "truth_score_logit": (
            float(truth_score_logit) if truth_score_logit is not None else None
        ),
        "truth_score": float(truth_score) if truth_score is not None else None,
        "truth_score_accepted": bool(truth_score_accepted),
        "accepted": bool(accepted),
        "selection_score": float(selection_score),
    }


def run_charge_reduced_headless(residues, spectrum, isodec_config) -> dict:
    spectrum_copy = np.array(spectrum, dtype=float, copy=True)
    spectrum_mz = spectrum_copy[:, 0]
    spectrum_int = spectrum_copy[:, 1]
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0
    search_spectrum, search_window = _filter_spectrum_by_requested_mz_window(
        spectrum_copy
    )
    search_mz = (
        np.asarray(search_spectrum[:, 0], dtype=float)
        if search_spectrum.size
        else np.asarray([], dtype=float)
    )
    search_int = (
        np.asarray(search_spectrum[:, 1], dtype=float)
        if search_spectrum.size
        else np.asarray([], dtype=float)
    )

    targets: dict[str, tuple[object, str]] = {}
    monomer_comp = get_neutral_monomer_composition(residues)
    targets["Monomer"] = (monomer_comp, "tab:blue")
    if int(cfg.COPIES) > 1:
        complex_comp = get_precursor_composition(residues)
        targets[f"Complex ({int(cfg.COPIES)}x)"] = (complex_comp, "tab:red")

    state_shifts = _charge_reduced_state_shifts()
    match_tol_ppm = float(cfg.MATCH_TOL_PPM)
    anchor_search_da = float(getattr(cfg, "CR_ANCHOR_SEARCH_DA", 2.0))
    local_window_da = float(getattr(cfg, "CR_LOCAL_WINDOW_DA", 3.0))
    min_matched_peaks = max(1, int(getattr(cfg, "CR_MIN_MATCHED_PEAKS", 2)))
    min_coverage = float(getattr(cfg, "CR_MIN_COVERAGE", 0.30))
    max_anchor_abs_ppm_cfg = getattr(cfg, "CR_MAX_ANCHOR_ABS_PPM", None)
    max_anchor_abs_ppm = (
        float(max_anchor_abs_ppm_cfg)
        if max_anchor_abs_ppm_cfg is not None
        else (match_tol_ppm * 1.5)
    )
    max_residual_rmse_cfg = getattr(cfg, "CR_MAX_RESIDUAL_RMSE_PPM", None)
    max_residual_rmse_ppm = (
        float(max_residual_rmse_cfg)
        if max_residual_rmse_cfg is not None
        else float(match_tol_ppm)
    )
    ppm_sigma_cfg = getattr(cfg, "CR_PPM_SIGMA", None)
    ppm_sigma = (
        float(ppm_sigma_cfg) if ppm_sigma_cfg is not None else float(match_tol_ppm)
    )
    if ppm_sigma <= 0:
        ppm_sigma = max(match_tol_ppm, 1.0)
    ambiguity_guard = bool(getattr(cfg, "CR_ENABLE_AMBIGUITY_GUARD", True))
    ambiguity_margin = float(getattr(cfg, "CR_AMBIGUITY_MARGIN", 0.03))
    score_w_css = float(getattr(cfg, "CR_SCORE_W_CSS", 0.45))
    score_w_cov = float(getattr(cfg, "CR_SCORE_W_COVERAGE", 0.25))
    score_w_ppm = float(getattr(cfg, "CR_SCORE_W_PPM", 0.20))
    score_w_spacing = float(getattr(cfg, "CR_SCORE_W_SPACING", 0.10))
    score_w_intensity = float(getattr(cfg, "CR_SCORE_W_INTENSITY", 0.25))
    score_w_sum = (
        score_w_css + score_w_cov + score_w_ppm + score_w_spacing + score_w_intensity
    )
    if score_w_sum <= 0:
        score_w_css, score_w_cov, score_w_ppm, score_w_spacing, score_w_intensity = (
            0.45,
            0.25,
            0.20,
            0.10,
            0.25,
        )
    else:
        score_w_css /= score_w_sum
        score_w_cov /= score_w_sum
        score_w_ppm /= score_w_sum
        score_w_spacing /= score_w_sum
        score_w_intensity /= score_w_sum

    accepted_candidates: list[dict] = []
    ambiguous_candidates: list[dict] = []
    shadowed_candidates: list[dict] = []
    all_candidates: list[dict] = []

    print("--- Starting Charge-Reduced Precursor Search (data-driven anchor) ---")
    if search_window is not None:
        print(
            "Restricting charge-reduced search to requested m/z window "
            f"[{search_window['min'] if search_window['min'] is not None else '-inf'}, "
            f"{search_window['max'] if search_window['max'] is not None else '+inf'}]."
        )

    if search_spectrum.size:
        for label, (target_comp, color) in targets.items():
            for z in range(int(cfg.CR_MIN_CHARGE), int(cfg.CR_MAX_CHARGE) + 1):
                try:
                    dist0 = theoretical_isodist_from_comp(target_comp, z)
                except ValueError:
                    continue
                if dist0.size == 0:
                    continue

                for state, h_shift in state_shifts:
                    dist_state = dist0.copy()
                    shift_mz = (float(h_shift) * float(cfg.H_TRANSFER_MASS)) / float(z)
                    if h_shift:
                        dist_state[:, 0] += shift_mz
                    anchor_idx = get_anchor_idx(dist_state)
                    anchor_theory_mz = float(dist_state[anchor_idx, 0])
                    anchor_window_mask = (
                        search_mz >= anchor_theory_mz - anchor_search_da
                    ) & (search_mz <= anchor_theory_mz + anchor_search_da)
                    if not np.any(anchor_window_mask):
                        continue
                    window_idx = np.where(anchor_window_mask)[0]
                    obs_anchor_mz = float(
                        search_mz[window_idx[np.argmax(search_int[window_idx])]]
                    )

                    spacing_sigma_cfg = getattr(cfg, "CR_SPACING_SIGMA_DA", None)
                    if spacing_sigma_cfg is None:
                        spacing_sigma_da = abs(anchor_theory_mz) * match_tol_ppm * 1e-6
                    else:
                        spacing_sigma_da = float(spacing_sigma_cfg)
                    if spacing_sigma_da <= 0:
                        spacing_sigma_da = max(
                            abs(anchor_theory_mz) * match_tol_ppm * 1e-6, 1e-6
                        )

                    def _score_cr(*, use_centroid_logic: bool):
                        local = get_local_centroids_window(
                            search_mz,
                            search_int,
                            center_mz=obs_anchor_mz,
                            lb=-local_window_da,
                            ub=local_window_da,
                            force_hill=bool(use_centroid_logic),
                        )
                        if not isinstance(local, np.ndarray) or local.size == 0:
                            local_mask = (
                                search_mz >= obs_anchor_mz - local_window_da
                            ) & (search_mz <= obs_anchor_mz + local_window_da)
                            if not np.any(local_mask):
                                return None
                            local = np.column_stack(
                                (search_mz[local_mask], search_int[local_mask])
                            )
                        if local.ndim != 2 or local.shape[1] != 2:
                            return None

                        cent_mz = np.asarray(local[:, 0], dtype=float)
                        cent_int = np.asarray(local[:, 1], dtype=float)
                        local_max_int = (
                            float(np.max(cent_int)) if cent_int.size else 0.0
                        )
                        if local_max_int <= 0:
                            return None

                        if cfg.ENABLE_ISODEC_RULES and isodec_config is not None:
                            accepted_model, css, shifted_peak = isodec_css_and_accept(
                                local,
                                dist_state,
                                z=z,
                                peakmz=obs_anchor_mz,
                                config=isodec_config,
                            )
                        else:
                            y_obs = observed_intensities_isodec(
                                cent_mz,
                                cent_int,
                                dist_state[:, 0],
                                z=int(z),
                                match_tol_ppm=match_tol_ppm,
                                peak_mz=obs_anchor_mz,
                            )
                            css = css_similarity(y_obs, dist_state[:, 1])
                            accepted_model = css >= float(cfg.MIN_COSINE)
                            shifted_peak = None

                        if not accepted_model or float(css) < float(cfg.MIN_COSINE):
                            return None

                        obs_mz = (
                            float(shifted_peak)
                            if shifted_peak is not None
                            else float(obs_anchor_mz)
                        )
                        obs_idx = nearest_peak_index(cent_mz, obs_mz)
                        if obs_idx < 0:
                            return None
                        obs_int = float(cent_int[obs_idx])
                        anchor_rel_int = (
                            float(obs_int / local_max_int) if local_max_int > 0 else 0.0
                        )
                        anchor_ppm = (
                            ((obs_mz - anchor_theory_mz) / anchor_theory_mz) * 1e6
                            if anchor_theory_mz
                            else 0.0
                        )
                        anchor_ppm_abs = abs(float(anchor_ppm))
                        shift_da_local = float(obs_mz - anchor_theory_mz)
                        matches = _match_theory_local_monotonic(
                            local, dist_state, shift_da_local, match_tol_ppm
                        )
                        comp = _composite_match_components(
                            css=float(css),
                            matches=matches,
                            dist_shifted=dist_state,
                            anchor_ppm_abs=float(anchor_ppm_abs),
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

                        accepted = bool(
                            bool(accepted_model)
                            and float(css) >= float(cfg.MIN_COSINE)
                            and anchor_ppm_abs <= float(max_anchor_abs_ppm)
                            and len(matches) >= int(min_matched_peaks)
                            and float(comp["coverage"]) >= float(min_coverage)
                        )
                        if len(matches) >= 2 and np.isfinite(comp["ppm_rmse"]):
                            accepted = accepted and float(comp["ppm_rmse"]) <= float(
                                max_residual_rmse_ppm
                            )
                        if not accepted:
                            return None

                        dist_full = dist_state.copy()
                        dist_plot = dist_state.copy()
                        max_plot = (
                            float(np.max(dist_plot[:, 1])) if dist_plot.size else 0.0
                        )
                        if max_plot > 0 and obs_int > 0:
                            dist_plot[:, 1] *= obs_int / max_plot
                        keep = (
                            dist_plot[:, 1]
                            >= max_plot * float(cfg.REL_INTENSITY_CUTOFF)
                            if max_plot > 0
                            else dist_plot[:, 1] > 0
                        )
                        dist_plot = dist_plot[keep] if dist_plot.size else dist_plot
                        theory_matches = match_theory_peaks(
                            cent_mz,
                            cent_int,
                            dist_full[:, 0],
                            tol_ppm=match_tol_ppm,
                            theory_int=dist_full[:, 1],
                        )
                        search_obs_idx = nearest_peak_index(search_mz, obs_mz)
                        short_label = f"{label}^{z}+"
                        if state != "0":
                            short_label += f" ({state})"

                        return {
                            "label": f"{short_label} | score={float(comp['composite_score']):.3f} css={float(css):.3f}",
                            "short_label": short_label,
                            "target": label,
                            "z": int(z),
                            "state": state,
                            "css": float(css),
                            "score": float(comp["composite_score"]),
                            "coverage": float(comp["coverage"]),
                            "ppm_rmse": float(comp["ppm_rmse"]),
                            "ppm_consistency": float(comp["ppm_consistency"]),
                            "spacing_consistency": float(comp["spacing_consistency"]),
                            "spacing_rmse_da": float(comp["spacing_rmse_da"]),
                            "match_count": int(len(matches)),
                            "obs_mz": obs_mz,
                            "obs_int": obs_int,
                            "anchor_theory_mz": float(anchor_theory_mz),
                            "anchor_ppm": float(anchor_ppm),
                            "anchor_rel_int": float(anchor_rel_int),
                            "dist_full": dist_full,
                            "dist": dist_plot,
                            "theory_matches": theory_matches,
                            "color": color,
                            "search_obs_idx": int(search_obs_idx),
                            "accepted": True,
                        }

                    best_candidate = execute_hybrid_strategy(_score_cr)
                    if best_candidate is not None:
                        all_candidates.append(best_candidate)

    all_candidates.sort(
        key=lambda x: (
            float(x.get("score", x.get("css", 0.0))),
            float(x.get("css", 0.0)),
            float(x.get("obs_int", 0.0)),
        ),
        reverse=True,
    )

    grouped_candidates: dict[int, list[dict]] = {}
    for cand in all_candidates:
        grouped_candidates.setdefault(int(cand.get("search_obs_idx", -1)), []).append(
            cand
        )

    for group in grouped_candidates.values():
        group.sort(
            key=lambda x: (
                float(x.get("score", x.get("css", 0.0))),
                float(x.get("css", 0.0)),
                float(x.get("obs_int", 0.0)),
            ),
            reverse=True,
        )
        if len(group) == 1:
            accepted = dict(group[0])
            accepted["status"] = "accepted"
            accepted["accepted"] = True
            accepted_candidates.append(accepted)
            continue

        top = group[0]
        runner_up = group[1]
        gap = float(top.get("score", 0.0)) - float(runner_up.get("score", 0.0))
        if ambiguity_guard and gap < float(ambiguity_margin):
            for idx, cand in enumerate(group):
                item = dict(cand)
                if idx < 2:
                    item["status"] = "ambiguous"
                    item["accepted"] = False
                    ambiguous_candidates.append(item)
                else:
                    item["status"] = "shadowed"
                    item["accepted"] = False
                    item["shadowed_by"] = str(
                        top.get("short_label", top.get("label", ""))
                    )
                    shadowed_candidates.append(item)
            continue

        accepted = dict(top)
        accepted["status"] = "accepted"
        accepted["accepted"] = True
        accepted_candidates.append(accepted)
        for cand in group[1:]:
            item = dict(cand)
            item["status"] = "shadowed"
            item["accepted"] = False
            item["shadowed_by"] = str(top.get("short_label", top.get("label", "")))
            shadowed_candidates.append(item)

    accepted_candidates.sort(
        key=lambda x: (
            float(x.get("score", x.get("css", 0.0))),
            float(x.get("obs_int", 0.0)),
        ),
        reverse=True,
    )
    ambiguous_candidates.sort(
        key=lambda x: (
            float(x.get("score", x.get("css", 0.0))),
            float(x.get("obs_int", 0.0)),
        ),
        reverse=True,
    )
    shadowed_candidates.sort(
        key=lambda x: (
            float(x.get("score", x.get("css", 0.0))),
            float(x.get("obs_int", 0.0)),
        ),
        reverse=True,
    )

    print(
        "Charge-reduced summary: "
        f"{len(accepted_candidates)} accepted, "
        f"{len(ambiguous_candidates)} ambiguous, "
        f"{len(shadowed_candidates)} shadowed."
    )
    for m in accepted_candidates:
        print(
            f"  ACCEPT {m['label']} at m/z {m['obs_mz']:.3f} "
            f"(score={float(m.get('score', 0.0)):.3f}, ppm={float(m.get('anchor_ppm', 0.0)):+.1f})"
        )
    for m in ambiguous_candidates[:3]:
        print(
            f"  AMBIG {m['label']} at m/z {m['obs_mz']:.3f} "
            f"(score={float(m.get('score', 0.0)):.3f}, ppm={float(m.get('anchor_ppm', 0.0)):+.1f})"
        )

    return {
        "spectrum": spectrum_copy,
        "spectrum_mz": np.asarray(spectrum_mz, dtype=float),
        "spectrum_int": np.asarray(spectrum_int, dtype=float),
        "obs_max": obs_max,
        "search_window": search_window,
        "search_status": "matched"
        if accepted_candidates
        else ("ambiguous" if ambiguous_candidates else "not_found"),
        "matches": accepted_candidates,
        "accepted_matches": accepted_candidates,
        "ambiguous_matches": ambiguous_candidates,
        "shadowed_matches": shadowed_candidates,
        "all_matches": accepted_candidates + ambiguous_candidates + shadowed_candidates,
    }


def run_charge_reduced_mode(residues, spectrum, isodec_config) -> None:
    result = run_charge_reduced_headless(residues, spectrum, isodec_config)
    spectrum_view = np.asarray(result.get("spectrum"), dtype=float)
    obs_max = float(result.get("obs_max", 0.0) or 0.0)
    matches = list(result.get("accepted_matches", []) or [])
    ambiguous_matches = list(result.get("ambiguous_matches", []) or [])
    shadowed_matches = list(result.get("shadowed_matches", []) or [])
    overlays_source = matches if matches else ambiguous_matches
    overlays = [(m["dist"], m["color"], m["label"]) for m in overlays_source]
    plot_overlay(
        spectrum_view,
        overlays,
        mz_min=cfg.MZ_MIN,
        mz_max=cfg.MZ_MAX,
        noise_cutoff=(obs_max * float(cfg.MIN_OBS_REL_INT))
        if float(cfg.MIN_OBS_REL_INT) > 0
        else None,
    )

    if bool(cfg.CHARGE_REDUCED_EXPORT_CSV):
        out_dir = Path(__file__).parent / "charge_reduced_outputs"
        file_tag = sanitize_filename(Path(str(cfg.filepath)).stem)
        base = sanitize_filename(f"charge_reduced_scan{int(cfg.SCAN)}_{file_tag}")

        summary_path = (
            Path(cfg.CHARGE_REDUCED_CSV_SUMMARY_PATH)
            if cfg.CHARGE_REDUCED_CSV_SUMMARY_PATH
            else out_dir / f"{base}_summary.csv"
        )
        peaks_path = (
            Path(cfg.CHARGE_REDUCED_CSV_PEAKS_PATH)
            if cfg.CHARGE_REDUCED_CSV_PEAKS_PATH
            else out_dir / f"{base}_peaks.csv"
        )

        summary_rows = []
        peaks_rows = []
        csv_candidates = matches + ambiguous_matches + shadowed_matches
        if csv_candidates:
            for m in csv_candidates:
                summary_rows.append(
                    {
                        "label": m.get("label", ""),
                        "status": m.get("status", "accepted"),
                        "target": m.get("target", ""),
                        "z": m.get("z", ""),
                        "state": m.get("state", ""),
                        "score": m.get("score", ""),
                        "css": m.get("css", ""),
                        "coverage": m.get("coverage", ""),
                        "ppm_rmse": m.get("ppm_rmse", ""),
                        "anchor_theory_mz": m.get("anchor_theory_mz", ""),
                        "anchor_ppm": m.get("anchor_ppm", ""),
                        "match_count": m.get("match_count", ""),
                        "obs_mz": m.get("obs_mz", ""),
                        "obs_int": m.get("obs_int", ""),
                        "strategy": m.get("strategy", ""),
                    }
                )

                for p in (
                    m.get("theory_matches", [])
                    if isinstance(m.get("theory_matches"), list)
                    else []
                ):
                    peaks_rows.append(
                        {
                            "label": m.get("label", ""),
                            "status": m.get("status", "accepted"),
                            "target": m.get("target", ""),
                            "z": m.get("z", ""),
                            "state": m.get("state", ""),
                            "score": m.get("score", ""),
                            "css": m.get("css", ""),
                            "theory_mz": p.get("theory_mz", ""),
                            "theory_int": p.get("theory_int", ""),
                            "obs_mz": p.get("obs_mz", ""),
                            "ppm": p.get("ppm", ""),
                            "obs_int": p.get("obs_int", ""),
                            "within": p.get("within", ""),
                            "obs_idx": p.get("obs_idx", ""),
                        }
                    )
        else:
            targets: dict[str, tuple[object, str]] = {}
            monomer_comp = get_neutral_monomer_composition(residues)
            targets["Monomer"] = (monomer_comp, "tab:blue")
            if int(cfg.COPIES) > 1:
                complex_comp = get_precursor_composition(residues)
                targets[f"Complex ({int(cfg.COPIES)}x)"] = (complex_comp, "tab:red")
            for label, (target_comp, _) in targets.items():
                for z in range(int(cfg.CR_MIN_CHARGE), int(cfg.CR_MAX_CHARGE) + 1):
                    try:
                        dist0 = theoretical_isodist_from_comp(target_comp, z)
                    except ValueError:
                        continue
                    if dist0.size == 0:
                        continue

                    for state, h_shift in _charge_reduced_state_shifts():
                        dist = dist0.copy()
                        dist[:, 0] += (
                            float(h_shift) * float(cfg.H_TRANSFER_MASS)
                        ) / float(z)
                        summary_rows.append(
                            {
                                "label": f"{label}^{z}+",
                                "status": "unmatched",
                                "target": label,
                                "z": int(z),
                                "state": state,
                                "score": "",
                                "css": "",
                                "coverage": "",
                                "ppm_rmse": "",
                                "anchor_theory_mz": "",
                                "anchor_ppm": "",
                                "match_count": "",
                                "obs_mz": "",
                                "obs_int": "",
                                "strategy": "",
                            }
                        )
                        for mz_val, int_val in dist:
                            peaks_rows.append(
                                {
                                    "label": f"{label}^{z}+",
                                    "status": "unmatched",
                                    "target": label,
                                    "z": int(z),
                                    "state": state,
                                    "score": "",
                                    "css": "",
                                    "theory_mz": float(mz_val),
                                    "theory_int": float(int_val),
                                    "obs_mz": "",
                                    "ppm": "",
                                    "obs_int": "",
                                    "within": False,
                                    "obs_idx": "",
                                }
                            )

        write_csv(
            summary_path,
            [
                "label",
                "status",
                "target",
                "z",
                "state",
                "score",
                "css",
                "coverage",
                "ppm_rmse",
                "anchor_theory_mz",
                "anchor_ppm",
                "match_count",
                "obs_mz",
                "obs_int",
                "strategy",
            ],
            summary_rows,
        )
        write_csv(
            peaks_path,
            [
                "label",
                "status",
                "target",
                "z",
                "state",
                "score",
                "css",
                "theory_mz",
                "theory_int",
                "obs_mz",
                "ppm",
                "obs_int",
                "within",
                "obs_idx",
            ],
            peaks_rows,
        )
        print(f"Wrote CSV: {summary_path}")
        print(f"Wrote CSV: {peaks_path}")


def run_fragments_mode(
    residues, spectrum, isodec_config, emit_outputs: bool = True
) -> dict:
    if (
        cfg.FRAG_MIN_CHARGE <= 0
        or cfg.FRAG_MAX_CHARGE <= 0
        or cfg.FRAG_MIN_CHARGE > cfg.FRAG_MAX_CHARGE
    ):
        raise ValueError(
            "Set FRAG_MIN_CHARGE/FRAG_MAX_CHARGE to a valid positive range."
        )

    # Get current mode from config
    current_mode = str(cfg.PLOT_MODE).lower()

    if bool(cfg.ENABLE_FRAGMENT_INTENSITY_CAP):
        tol_ppm = (
            float(cfg.MATCH_TOL_PPM)
            if cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM)
        )
        mz_min_cap = (
            None
            if cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN)
        )
        mz_max_cap = (
            None
            if cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX)
        )
        min_hits = int(max(0, cfg.FRAGMENT_INTENSITY_CAP_MIN_HITS))
        cap, hits = compute_fragment_intensity_cap(
            residues,
            spectrum[:, 0],
            spectrum[:, 1],
            tol_ppm=float(tol_ppm),
            mode=current_mode,
            mz_min=mz_min_cap,
            mz_max=mz_max_cap,
        )
        if hits >= min_hits and cap > 0:
            n_before = int(len(spectrum))
            spectrum = strip_peaks_above_intensity_cap(spectrum, cap=float(cap))
            removed = max(0, n_before - int(len(spectrum)))
            if removed > 0 or bool(cfg.FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(
                    f"Fragment intensity-cap strip: cap={cap:.3g} ({hits} windows), removed {removed} peaks"
                )
        else:
            if bool(cfg.FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(
                    f"Fragment intensity-cap strip: skipped (cap={cap:.3g}, hits={hits} < {min_hits})"
                )

    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0
    match_tol_ppm = float(cfg.MATCH_TOL_PPM)
    scoring_settings = _fragment_scoring_settings(match_tol_ppm=float(match_tol_ppm))
    min_matched_peaks = int(scoring_settings["min_matched_peaks"])
    max_anchor_abs_ppm = float(scoring_settings["max_anchor_abs_ppm"])
    ppm_sigma = float(scoring_settings["ppm_sigma"])
    spacing_sigma_cfg = getattr(cfg, "FRAG_SPACING_SIGMA_DA", None)
    min_isodec_css = float(scoring_settings["min_isodec_css"])
    noise_model_splits = int(scoring_settings["noise_model_splits"])
    noise_hist_bins = int(scoring_settings["noise_hist_bins"])
    noise_model = _build_noise_level_model(
        spectrum_mz,
        spectrum_int,
        num_splits=int(noise_model_splits),
        hist_bins=int(noise_hist_bins),
    )

    ion_colors = {
        "b": "tab:blue",
        "y": "tab:orange",
        "c": "tab:green",
        "z": "tab:red",
        "z-dot": "tab:red",
        "c-dot": "tab:green",
    }

    matches: list[dict] = []
    n = len(residues)

    # Import get_interchain_fragment_composition for complex_fragments mode
    from personalized_sequence import get_interchain_fragment_composition

    for ion_type in cfg.ION_TYPES:
        series = ion_series(ion_type)
        allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (
            series in set(cfg.H_TRANSFER_ION_TYPES_1H)
        )
        allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (
            series in set(cfg.H_TRANSFER_ION_TYPES_2H)
        )
        for frag_len in range(1, n):
            # Get appropriate composition based on mode
            if current_mode == "complex_fragments":
                # Use interchain fragment composition
                frag_name, target_comp = get_interchain_fragment_composition(
                    residues, ion_type, frag_len, amidated=cfg.AMIDATED
                )
            else:
                # Use regular fragment composition
                frag_name, target_comp = ion_composition_from_sequence(
                    residues, ion_type, frag_len, amidated=cfg.AMIDATED
                )

            cys_variants = get_disulfide_logic(ion_type, frag_len, n)
            if not cys_variants:
                cys_variants = [("", None)]

            def evaluate_candidate(
                frag_id_base: str,
                loss_suffix: str,
                loss_comp,
                z: int,
                variant_suffix: str,
                variant_type: str,
                use_centroid_logic: bool = True,
            ):
                try:
                    dist0 = theoretical_isodist_from_comp(loss_comp, z)
                except ValueError:
                    return None
                if dist0.size == 0:
                    return None

                shift_1 = (
                    float(cfg.H_TRANSFER_MASS) / float(z)
                    if (allow_1h or allow_2h)
                    else 0.0
                )
                shift_2 = (
                    2.0 * float(cfg.H_TRANSFER_MASS) / float(z) if allow_2h else 0.0
                )

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
                    return None

                peak_mz = float(dist0[get_anchor_idx(dist0), 0])
                y_obs = observed_intensities_isodec(
                    spectrum_mz,
                    spectrum_int,
                    sample_mzs,
                    z=int(z),
                    match_tol_ppm=float(cfg.MATCH_TOL_PPM),
                    peak_mz=peak_mz,
                )
                y0 = vectorize_dist(
                    dist0, sample_keys, scale, mz_min=cfg.MZ_MIN, mz_max=cfg.MZ_MAX
                )

                neutral_score_union = css_similarity(y_obs, y0)
                neutral_score = neutral_score_union
                dist0_neutral = dist0
                if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
                    mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
                    mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
                    dist0_neutral = dist0[
                        (dist0[:, 0] >= mz_min) & (dist0[:, 0] <= mz_max)
                    ]
                if dist0_neutral.size:
                    y_obs_neutral = observed_intensities_isodec(
                        spectrum_mz,
                        spectrum_int,
                        dist0_neutral[:, 0],
                        z=int(z),
                        match_tol_ppm=float(cfg.MATCH_TOL_PPM),
                        peak_mz=peak_mz,
                    )
                    neutral_score = css_similarity(y_obs_neutral, dist0_neutral[:, 1])
                best_model = "neutral"
                best_score = neutral_score_union
                best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}
                best_pred = y0

                if allow_1h or allow_2h:
                    yp1 = (
                        vectorize_dist(
                            dist_p1,
                            sample_keys,
                            scale,
                            mz_min=cfg.MZ_MIN,
                            mz_max=cfg.MZ_MAX,
                        )
                        if allow_1h
                        else None
                    )
                    ym1 = (
                        vectorize_dist(
                            dist_m1,
                            sample_keys,
                            scale,
                            mz_min=cfg.MZ_MIN,
                            mz_max=cfg.MZ_MAX,
                        )
                        if allow_1h
                        else None
                    )
                    yp2 = (
                        vectorize_dist(
                            dist_p2,
                            sample_keys,
                            scale,
                            mz_min=cfg.MZ_MIN,
                            mz_max=cfg.MZ_MAX,
                        )
                        if allow_2h
                        else None
                    )
                    ym2 = (
                        vectorize_dist(
                            dist_m2,
                            sample_keys,
                            scale,
                            mz_min=cfg.MZ_MIN,
                            mz_max=cfg.MZ_MAX,
                        )
                        if allow_2h
                        else None
                    )

                    comps_plus = [("0", y0)]
                    comps_minus = [("0", y0)]
                    if allow_1h:
                        comps_plus.append(("+H", yp1))
                        comps_minus.append(("-H", ym1))
                    if allow_2h:
                        comps_plus.append(("+2H", yp2))
                        comps_minus.append(("-2H", ym2))

                    names_plus, vecs_plus = zip(*comps_plus)
                    w_plus, y_plus, score_plus = fit_simplex_mixture(
                        y_obs, list(vecs_plus)
                    )
                    weights_plus = dict(zip(names_plus, w_plus))

                    names_minus, vecs_minus = zip(*comps_minus)
                    w_minus, y_minus, score_minus = fit_simplex_mixture(
                        y_obs, list(vecs_minus)
                    )
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

                    rel_improve = (best_score - neutral_score_union) / max(
                        neutral_score_union, 1e-12
                    )
                    if best_model != "neutral" and rel_improve < float(
                        cfg.H_TRANSFER_MIN_REL_IMPROVEMENT
                    ):
                        best_model = "neutral"
                        best_score = neutral_score_union
                        best_pred = y0
                        best_weights = {
                            "0": 1.0,
                            "+H": 0.0,
                            "+2H": 0.0,
                            "-H": 0.0,
                            "-2H": 0.0,
                        }
                if best_model == "neutral":
                    best_score = neutral_score

                if float(np.max(best_pred)) <= 0.0:
                    return None

                dist_model = np.column_stack([sample_mzs.copy(), best_pred.copy()])
                if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
                    mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
                    mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
                    dist_model = dist_model[
                        (dist_model[:, 0] >= mz_min) & (dist_model[:, 0] <= mz_max)
                    ]
                if dist_model.size == 0:
                    return None
                max_model = float(np.max(dist_model[:, 1]))
                if max_model <= 0.0:
                    return None
                keep_model = dist_model[:, 1] >= max_model * float(
                    cfg.REL_INTENSITY_CUTOFF
                )
                dist_model = dist_model[keep_model]
                if dist_model.size == 0:
                    return None

                anchor_theory_mz = None
                obs_idx = None
                obs_mz = None
                obs_int = None
                anchor_hits = 0
                if bool(getattr(cfg, "FRAG_ANCHOR_USE_HYPOTHESIS_SCORING", False)):
                    anchor_choice, anchor_hits = _choose_fragment_anchor_hypothesis(
                        spectrum_mz,
                        spectrum_int,
                        sample_mzs=sample_mzs,
                        best_pred=best_pred,
                        dist_model=dist_model,
                        z=int(z),
                        obs_max=float(obs_max),
                        match_tol_ppm=float(match_tol_ppm),
                        use_centroid_logic=bool(use_centroid_logic),
                    )
                    if anchor_choice is not None:
                        anchor_theory_mz = float(anchor_choice["theory_mz"])
                        obs_idx = int(anchor_choice["obs_idx"])
                        obs_mz = float(anchor_choice["obs_mz"])
                        obs_int = float(anchor_choice["obs_int"])
                elif bool(getattr(cfg, "FRAG_ANCHOR_USE_INTENSITY_FALLBACK", False)):
                    sorted_idx = np.argsort(best_pred)[::-1][: int(cfg.ANCHOR_TOP_N)]
                    for idx in sorted_idx:
                        mz_candidate = float(sample_mzs[int(idx)])
                        candidate_rows = _iter_fragment_anchor_candidates_by_intensity(
                            spectrum_mz,
                            spectrum_int,
                            mz_candidate=mz_candidate,
                            use_centroid_logic=bool(use_centroid_logic),
                        )
                        for obs_mz_c, obs_int_c, obs_idx_c in candidate_rows:
                            if not within_ppm(
                                obs_mz_c, mz_candidate, float(match_tol_ppm)
                            ):
                                continue
                            if float(
                                cfg.MIN_OBS_REL_INT
                            ) > 0 and obs_int_c < obs_max * float(cfg.MIN_OBS_REL_INT):
                                continue
                            anchor_hits += 1
                            if anchor_theory_mz is None:
                                anchor_theory_mz = mz_candidate
                                obs_idx = int(obs_idx_c)
                                obs_mz = obs_mz_c
                                obs_int = obs_int_c
                            break
                else:
                    anchor_window = float(
                        getattr(cfg, "FRAG_ANCHOR_CENTROID_WINDOW_DA", 0.2)
                    )
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
                        if (
                            isinstance(local_centroids, np.ndarray)
                            and local_centroids.size
                        ):
                            best_local_idx = int(np.argmax(local_centroids[:, 1]))
                            obs_mz_c = float(local_centroids[best_local_idx, 0])
                            obs_int_c = float(local_centroids[best_local_idx, 1])
                            obs_idx_c = nearest_peak_index(spectrum_mz, obs_mz_c)
                        else:
                            obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
                            obs_mz_c = float(spectrum_mz[obs_idx_c])
                            obs_int_c = float(spectrum_int[obs_idx_c])
                        if not within_ppm(obs_mz_c, mz_candidate, float(match_tol_ppm)):
                            continue
                        if float(
                            cfg.MIN_OBS_REL_INT
                        ) > 0 and obs_int_c < obs_max * float(cfg.MIN_OBS_REL_INT):
                            continue
                        anchor_hits += 1
                        if anchor_theory_mz is None:
                            anchor_theory_mz = mz_candidate
                            obs_idx = int(obs_idx_c)
                            obs_mz = obs_mz_c
                            obs_int = obs_int_c
                if (
                    anchor_hits < int(cfg.ANCHOR_MIN_MATCHES)
                    or anchor_theory_mz is None
                ):
                    return None
                ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6

                dist_plot = dist_model.copy()
                dist_plot[:, 0] += obs_mz - anchor_theory_mz
                dist_plot[:, 1] *= obs_int / float(np.max(dist_plot[:, 1]))

                isodec_css = float(best_score)
                isodec_accepted = True
                isodec_detail = {
                    "matched_peaks_n": 0,
                    "minpeaks_effective": 0,
                    "areacovered": 0.0,
                    "topthree": False,
                }
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
                    if (not isodec_accepted) and (
                        not bool(scoring_settings["truth_score_enabled"])
                    ):
                        return None
                    if shifted_peak is not None:
                        old_obs_mz = obs_mz
                        obs_mz_new = float(shifted_peak)
                        obs_idx = nearest_peak_index(spectrum_mz, obs_mz_new)
                        obs_mz = float(spectrum_mz[obs_idx])
                        obs_int = float(spectrum_int[obs_idx])
                        ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6
                        dist_plot[:, 0] += obs_mz - old_obs_mz

                pipeline = _evaluate_fragment_pipeline(
                    spectrum_mz=spectrum_mz,
                    spectrum_int=spectrum_int,
                    obs_max=float(obs_max),
                    match_tol_ppm=float(match_tol_ppm),
                    dist_model=dist_model,
                    dist_plot=dist_plot,
                    obs_mz=float(obs_mz),
                    obs_int=float(obs_int),
                    anchor_theory_mz=float(anchor_theory_mz),
                    ppm=float(ppm),
                    isodec_css=float(isodec_css),
                    isodec_detail=isodec_detail,
                    isodec_config=isodec_config,
                    use_centroid_logic=bool(use_centroid_logic),
                    noise_model=noise_model,
                    settings=scoring_settings,
                )
                if not pipeline.get("pipeline_ready"):
                    return None
                if not pipeline.get("accepted"):
                    return None

                obs_rel_int = float(obs_int / obs_max) if obs_max > 0 else 0.0
                comp = pipeline["comp"]
                quality = pipeline["quality"]
                local_matches = pipeline["local_matches"]
                legacy_accepted = bool(pipeline["legacy_accepted"])
                truth_score_logit = pipeline.get("truth_score_logit")
                truth_score = pipeline.get("truth_score")
                truth_score_accepted = bool(pipeline.get("truth_score_accepted"))

                frag_id = f"{frag_id_base}{loss_suffix}"
                evidence_score = float(quality["evidence_score"])
                selection_score = float(pipeline["selection_score"])
                label_parts = [
                    f"{frag_id}^{z}+",
                    f"{ppm:.1f} ppm",
                    f"score={selection_score:.3f}",
                    f"css={isodec_css:.3f}",
                ]
                if bool(scoring_settings["truth_score_enabled"]) and truth_score is not None:
                    label_parts.append(f"truth={truth_score:.3f}")
                if best_model != "neutral":
                    h_pct = 100.0 * float(
                        best_weights.get("+H", 0.0) + best_weights.get("-H", 0.0)
                    )
                    h2_pct = 100.0 * float(
                        best_weights.get("+2H", 0.0) + best_weights.get("-2H", 0.0)
                    )
                    label_parts.append(f"%H={h_pct:.0f}")
                    if allow_2h:
                        label_parts.append(f"%2H={h2_pct:.0f}")
                    label_parts.append(best_model)
                label = " | ".join(label_parts)
                return {
                    "frag_id": frag_id,
                    "ion_type": ion_type,
                    "series": series,
                    "frag_len": int(frag_len),
                    "charge": int(z),
                    "loss_suffix": loss_suffix,
                    "variant_suffix": variant_suffix,
                    "variant_type": variant_type,
                    "formula": composition_to_formula(loss_comp, proton_count=int(z)),
                    "best_model": best_model,
                    "obs_idx": int(obs_idx),
                    "obs_mz": float(obs_mz),
                    "obs_int": obs_int,
                    "obs_rel_int": obs_rel_int,
                    "anchor_theory_mz": float(anchor_theory_mz),
                    "ppm": float(ppm),
                    "score": float(evidence_score),
                    "selection_score": float(selection_score),
                    "truth_score": float(truth_score)
                    if truth_score is not None
                    else None,
                    "truth_score_logit": float(truth_score_logit)
                    if truth_score_logit is not None
                    else None,
                    "truth_score_accepted": bool(truth_score_accepted),
                    "legacy_accepted": bool(legacy_accepted),
                    "css": float(isodec_css),
                    "composite_score": float(evidence_score),
                    "raw_score": float(best_score),
                    "neutral_score": float(neutral_score),
                    "isodec_accepted": bool(isodec_accepted),
                    "isodec_top_peaks": bool(isodec_detail.get("topthree", False)),
                    "coverage": float(comp["coverage"]),
                    "ppm_rmse": float(comp["ppm_rmse"]),
                    "ppm_consistency": float(comp["ppm_consistency"]),
                    "spacing_consistency": float(comp["spacing_consistency"]),
                    "spacing_rmse_da": float(comp["spacing_rmse_da"]),
                    "match_count": int(len(local_matches)),
                    "local_explained_fraction": float(
                        quality["local_explained_fraction"]
                    ),
                    "unexplained_fraction": float(quality["unexplained_fraction"]),
                    "core_coverage": float(quality["core_coverage"]),
                    "missing_core_fraction": float(quality["missing_core_fraction"]),
                    "interference": float(quality["interference"]),
                    "num_missing_peaks": int(quality["num_missing_peaks"]),
                    "pc_missing_peaks": float(quality["pc_missing_peaks"]),
                    "fit_score": float(quality["fit_score"]),
                    "correlation_coefficient": float(
                        quality["correlation_coefficient"]
                    ),
                    "chisq_stat": float(quality["chisq_stat"]),
                    "mass_error_std": float(quality["mass_error_std"]),
                    "noise_level": float(quality["noise_level"]),
                    "s2n": float(quality["s2n"]),
                    "log_s2n": float(quality["log_s2n"]),
                    "isodec_detail": isodec_detail,
                    "h_weights": best_weights,
                    "dist": dist_plot,
                    "label": label,
                    "color": ion_colors.get(ion_type, "tab:purple"),
                }

            variant_shift_map = {suffix: shift for suffix, shift in cys_variants}

            for z in range(int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1):
                neutral_candidates = []
                for variant_suffix, shift_comp in cys_variants:
                    if shift_comp is not None:
                        try:
                            variant_comp = target_comp + shift_comp
                        except Exception:
                            continue
                        frag_id_base = f"{frag_name}{variant_suffix}"
                    else:
                        variant_comp = target_comp
                        frag_id_base = frag_name

                    variant_type = variant_type_from_suffix(variant_suffix)
                    neutral_match = evaluate_candidate(
                        frag_id_base,
                        "",
                        variant_comp,
                        z,
                        variant_suffix,
                        variant_type,
                    )
                    neutral_match = execute_hybrid_strategy(
                        evaluate_candidate,
                        frag_id_base,
                        "",
                        variant_comp,
                        z,
                        variant_suffix,
                        variant_type,
                    )
                    if neutral_match is not None:
                        neutral_candidates.append(neutral_match)

                if not neutral_candidates:
                    continue

                pass_count = 0
                for candidate in neutral_candidates:
                    try:
                        score_val = float(candidate.get("score", float("-inf")))
                    except Exception:
                        score_val = float("-inf")
                    if score_val >= float(cfg.ISODEC_CSS_THRESH):
                        pass_count += 1

                best_neutral = max(neutral_candidates, key=variant_rank_key_from_result)
                best_neutral["variant_pass_count"] = int(pass_count)
                matches.append(best_neutral)

                # Gate B: only consider loss variants if neutral passes final CSS filter.
                if best_neutral["score"] < float(cfg.ISODEC_CSS_THRESH):
                    continue

                best_suffix = best_neutral.get("variant_suffix", "")
                best_shift = variant_shift_map.get(best_suffix)
                if best_shift:
                    try:
                        variant_comp = target_comp + best_shift
                    except Exception:
                        continue
                    frag_id_base = f"{frag_name}{best_suffix}"
                else:
                    variant_comp = target_comp
                    frag_id_base = frag_name

                variant_type = best_neutral.get(
                    "variant_type"
                ) or variant_type_from_suffix(best_suffix)

                # Loss variants only after neutral passes final filter.
                for loss_suffix, loss_comp in neutral_loss_variants(
                    variant_comp, ion_series_letter=series
                ):
                    if not loss_suffix:
                        continue
                    loss_match = evaluate_candidate(
                        frag_id_base,
                        loss_suffix,
                        loss_comp,
                        z,
                        best_suffix,
                        variant_type,
                    )
                    loss_match = execute_hybrid_strategy(
                        evaluate_candidate,
                        frag_id_base,
                        loss_suffix,
                        loss_comp,
                        z,
                        best_suffix,
                        variant_type,
                    )
                    if loss_match is not None:
                        loss_match["variant_pass_count"] = int(pass_count)
                        matches.append(loss_match)

    best_by_obs: dict[int, dict] = {}
    for m in matches:
        key = m["obs_idx"]
        if key not in best_by_obs:
            best_by_obs[key] = m
            continue
        cur = best_by_obs[key]
        m_score = float(
            m.get("selection_score", m.get("score", m.get("css", float("-inf"))))
        )
        cur_score = float(
            cur.get("selection_score", cur.get("score", cur.get("css", float("-inf"))))
        )
        if (
            m_score > cur_score
            or (
                m_score == cur_score
                and float(m.get("coverage", 0.0)) > float(cur.get("coverage", 0.0))
            )
            or (
                m_score == cur_score
                and float(m.get("coverage", 0.0)) == float(cur.get("coverage", 0.0))
                and m["obs_int"] > cur["obs_int"]
            )
        ):
            best_by_obs[key] = m

    best = list(best_by_obs.values())
    best.sort(
        key=lambda d: (
            float(d.get("selection_score", d.get("score", d.get("css", 0.0)))),
            float(d.get("coverage", 0.0)),
            float(d.get("css", 0.0)),
            d["obs_int"],
        ),
        reverse=True,
    )
    if cfg.MAX_PLOTTED_FRAGMENTS is not None:
        best = best[: int(cfg.MAX_PLOTTED_FRAGMENTS)]

    if emit_outputs:
        print(f"Matched fragments: {len(best)} (from {len(matches)} raw matches)")
        for m in best:
            print(
                f"{m['label']}\tI={m['obs_int']:.3g}\t"
                f"anchor={m['anchor_theory_mz']:.4f}->{m['obs_mz']:.4f}\t"
                f"sel={float(m.get('selection_score', m.get('score', 0.0))):.3f}\t"
                f"cov={float(m.get('coverage', 0.0)):.3f}\t"
                f"unexp={float(m.get('unexplained_fraction', 0.0)):.3f}\t"
                f"fit={float(m.get('fit_score', 0.0)):.3f}\t"
                f"s2n={float(m.get('s2n', 0.0)):.2f}\t"
                f"rawcos={m['raw_score']:.3f}"
            )

        if bool(cfg.EXPORT_FRAGMENTS_CSV):
            out_dir = Path(__file__).parent / "match_outputs"
            file_tag = sanitize_filename(Path(str(cfg.filepath)).stem)
            mz_tag = f"{'' if cfg.MZ_MIN is None else int(cfg.MZ_MIN)}-{'' if cfg.MZ_MAX is None else int(cfg.MZ_MAX)}"
            base = sanitize_filename(
                f"fragments_scan{int(cfg.SCAN)}_{file_tag}_mz{mz_tag}"
            )

            summary_path = (
                Path(cfg.FRAGMENTS_CSV_SUMMARY_PATH)
                if cfg.FRAGMENTS_CSV_SUMMARY_PATH
                else (out_dir / f"{base}_summary.csv")
            )
            peaks_path = (
                Path(cfg.FRAGMENTS_CSV_PEAKS_PATH)
                if cfg.FRAGMENTS_CSV_PEAKS_PATH
                else (out_dir / f"{base}_peaks.csv")
            )

            summary_rows = []
            peaks_rows = []
            for m in best:
                h = m.get("h_weights") or {}
                pct_h = 100.0 * float(h.get("+H", 0.0) + h.get("-H", 0.0))
                pct_2h = 100.0 * float(h.get("+2H", 0.0) + h.get("-2H", 0.0))
                loss_cols = neutral_loss_columns(m.get("loss_suffix", ""))
                summary_rows.append(
                    {
                        "frag_id": m.get("frag_id", ""),
                        "ion_type": m.get("ion_type", ""),
                        "series": m.get("series", ""),
                        "frag_len": m.get("frag_len", ""),
                        "charge": m.get("charge", ""),
                        "H2O": loss_cols["H2O"],
                        "NH3": loss_cols["NH3"],
                        "CO": loss_cols["CO"],
                        "CO2": loss_cols["CO2"],
                        "2H2O": loss_cols["2H2O"],
                        "2NH3": loss_cols["2NH3"],
                        "variant_type": m.get("variant_type", ""),
                        "variant_suffix": m.get("variant_suffix", ""),
                        "variant_pass_count": m.get("variant_pass_count", ""),
                        "%H": pct_h,
                        "%2H": pct_2h,
                        "obs_idx": m.get("obs_idx", ""),
                        "obs_mz": m.get("obs_mz", ""),
                        "obs_int": m.get("obs_int", ""),
                        "obs_rel_int": m.get("obs_rel_int", ""),
                        "anchor_theory_mz": m.get("anchor_theory_mz", ""),
                        "anchor_ppm": m.get("ppm", ""),
                        "selection_score": m.get("selection_score", ""),
                        "score": m.get("score", ""),
                        "truth_score": m.get("truth_score", ""),
                        "truth_score_logit": m.get("truth_score_logit", ""),
                        "truth_score_accepted": m.get("truth_score_accepted", ""),
                        "legacy_accepted": m.get("legacy_accepted", ""),
                        "css": m.get("css", ""),
                        "rawcos": m.get("raw_score", ""),
                        "cos0": m.get("neutral_score", ""),
                        "coverage": m.get("coverage", ""),
                        "ppm_rmse": m.get("ppm_rmse", ""),
                        "match_count": m.get("match_count", ""),
                        "local_explained_fraction": m.get(
                            "local_explained_fraction", ""
                        ),
                        "unexplained_fraction": m.get("unexplained_fraction", ""),
                        "interference": m.get("interference", ""),
                        "core_coverage": m.get("core_coverage", ""),
                        "missing_core_fraction": m.get("missing_core_fraction", ""),
                        "num_missing_peaks": m.get("num_missing_peaks", ""),
                        "pc_missing_peaks": m.get("pc_missing_peaks", ""),
                        "fit_score": m.get("fit_score", ""),
                        "correlation_coefficient": m.get("correlation_coefficient", ""),
                        "chisq_stat": m.get("chisq_stat", ""),
                        "mass_error_std": m.get("mass_error_std", ""),
                        "noise_level": m.get("noise_level", ""),
                        "s2n": m.get("s2n", ""),
                        "log_s2n": m.get("log_s2n", ""),
                        "w0": h.get("0", ""),
                        "w_plusH": h.get("+H", ""),
                        "w_plus2H": h.get("+2H", ""),
                        "w_minusH": h.get("-H", ""),
                        "w_minus2H": h.get("-2H", ""),
                        "label": m.get("label", ""),
                    }
                )

                dist = m.get("dist")
                if isinstance(dist, np.ndarray) and dist.size:
                    for p in match_theory_peaks(
                        spectrum_mz,
                        spectrum_int,
                        dist[:, 0],
                        tol_ppm=float(cfg.MATCH_TOL_PPM),
                        theory_int=dist[:, 1],
                    ):
                        peaks_rows.append(
                            {
                                "frag_id": m.get("frag_id", ""),
                                "charge": m.get("charge", ""),
                                "H2O": loss_cols["H2O"],
                                "NH3": loss_cols["NH3"],
                                "CO": loss_cols["CO"],
                                "CO2": loss_cols["CO2"],
                                "2H2O": loss_cols["2H2O"],
                                "2NH3": loss_cols["2NH3"],
                                "variant_type": m.get("variant_type", ""),
                                "variant_suffix": m.get("variant_suffix", ""),
                                "variant_pass_count": m.get("variant_pass_count", ""),
                                "%H": pct_h,
                                "%2H": pct_2h,
                                "score": m.get("score", ""),
                                "css": m.get("css", ""),
                                "rawcos": m.get("raw_score", ""),
                                "theory_mz": p.get("theory_mz", ""),
                                "theory_int": p.get("theory_int", ""),
                                "obs_mz": p.get("obs_mz", ""),
                                "ppm": p.get("ppm", ""),
                                "obs_int": p.get("obs_int", ""),
                                "within": p.get("within", ""),
                                "obs_idx": p.get("obs_idx", ""),
                            }
                        )

            write_csv(
                summary_path,
                [
                    "frag_id",
                    "ion_type",
                    "series",
                    "frag_len",
                    "charge",
                    "H2O",
                    "NH3",
                    "CO",
                    "CO2",
                    "2H2O",
                    "2NH3",
                    "variant_type",
                    "variant_suffix",
                    "variant_pass_count",
                    "%H",
                    "%2H",
                    "obs_idx",
                    "obs_mz",
                    "obs_int",
                    "obs_rel_int",
                    "anchor_theory_mz",
                    "anchor_ppm",
                    "selection_score",
                    "score",
                    "truth_score",
                    "truth_score_logit",
                    "truth_score_accepted",
                    "legacy_accepted",
                    "css",
                    "rawcos",
                    "cos0",
                    "coverage",
                    "ppm_rmse",
                    "match_count",
                    "local_explained_fraction",
                    "unexplained_fraction",
                    "interference",
                    "core_coverage",
                    "missing_core_fraction",
                    "num_missing_peaks",
                    "pc_missing_peaks",
                    "fit_score",
                    "correlation_coefficient",
                    "chisq_stat",
                    "mass_error_std",
                    "noise_level",
                    "s2n",
                    "log_s2n",
                    "w0",
                    "w_plusH",
                    "w_plus2H",
                    "w_minusH",
                    "w_minus2H",
                    "label",
                ],
                summary_rows,
            )
            write_csv(
                peaks_path,
                [
                    "frag_id",
                    "charge",
                    "H2O",
                    "NH3",
                    "CO",
                    "CO2",
                    "2H2O",
                    "2NH3",
                    "variant_type",
                    "variant_suffix",
                    "variant_pass_count",
                    "%H",
                    "%2H",
                    "score",
                    "css",
                    "rawcos",
                    "theory_mz",
                    "theory_int",
                    "obs_mz",
                    "ppm",
                    "obs_int",
                    "within",
                    "obs_idx",
                ],
                peaks_rows,
            )
            print(f"Wrote CSV: {summary_path}")
            print(f"Wrote CSV: {peaks_path}")

        overlays = [(m["dist"], m["color"], m["label"]) for m in best]
        plot_overlay(
            spectrum,
            overlays,
            mz_min=None if cfg.MZ_MIN is None else float(cfg.MZ_MIN),
            mz_max=None if cfg.MZ_MAX is None else float(cfg.MZ_MAX),
            noise_cutoff=(obs_max * float(cfg.MIN_OBS_REL_INT))
            if float(cfg.MIN_OBS_REL_INT) > 0
            else None,
        )

    return {
        "spectrum": spectrum,
        "spectrum_mz": spectrum_mz,
        "spectrum_int": spectrum_int,
        "obs_max": obs_max,
        "matches": matches,
        "best": best,
    }


def run_fragments_headless(residues, spectrum, isodec_config) -> dict:
    return run_fragments_mode(residues, spectrum, isodec_config, emit_outputs=False)


def run_diagnose_headless(
    residues, spectrum, isodec_config, ion_spec: str = None, h_transfer: int = 0
) -> dict:
    """Run diagnose mode and return results without plotting (for API use)."""
    spec = ion_spec or cfg.DIAGNOSE_ION_SPEC
    if not spec:
        return {
            "error": "No ion spec provided",
            "results": [],
            "spectrum_mz": spectrum[:, 0].tolist() if spectrum.size else [],
            "spectrum_int": spectrum[:, 1].tolist() if spectrum.size else [],
        }

    raw_spectrum = np.array(spectrum, dtype=float, copy=True)
    if bool(cfg.ENABLE_FRAGMENT_INTENSITY_CAP):
        tol_ppm = (
            float(cfg.MATCH_TOL_PPM)
            if cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM)
        )
        mz_min_cap = (
            None
            if cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN)
        )
        mz_max_cap = (
            None
            if cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX)
        )
        min_hits = int(max(0, cfg.FRAGMENT_INTENSITY_CAP_MIN_HITS))
        cap, hits = compute_fragment_intensity_cap(
            residues,
            spectrum[:, 0],
            spectrum[:, 1],
            tol_ppm=float(tol_ppm),
            mz_min=mz_min_cap,
            mz_max=mz_max_cap,
        )
        if hits >= min_hits and cap > 0:
            spectrum = strip_peaks_above_intensity_cap(spectrum, cap=float(cap))

    ion_type, frag_len, loss_formula, loss_count, charge = parse_fragment_spec(spec)
    charges = (
        [int(charge)]
        if charge is not None
        else list(range(int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1))
    )

    if h_transfer not in (-2, -1, 0, 1, 2):
        h_transfer = 0

    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]

    results = []
    for z in charges:
        r = diagnose_candidate(
            residues=residues,
            ion_type=ion_type,
            frag_len=frag_len,
            z=int(z),
            loss_formula=loss_formula,
            loss_count=int(loss_count),
            h_transfer=h_transfer,
            spectrum_mz=spectrum_mz,
            spectrum_int=spectrum_int,
            match_tol_ppm=float(cfg.MATCH_TOL_PPM),
            min_obs_rel_int=float(cfg.MIN_OBS_REL_INT),
            rel_intensity_cutoff=float(cfg.REL_INTENSITY_CUTOFF),
            mz_min=cfg.MZ_MIN,
            mz_max=cfg.MZ_MAX,
            isodec_config=isodec_config,
        )
        results.append(r)

    def rank_key(d: dict):
        try:
            css = float(d.get("isodec_css", float("nan")))
        except Exception:
            css = float("nan")
        raw = d.get("raw_cosine_preanchor", d.get("raw_cosine", 0.0))
        try:
            raw_val = float(raw)
        except Exception:
            raw_val = 0.0
        ok = 1 if d.get("ok") else 0
        css_val = css if np.isfinite(css) else -1.0
        return (ok, css_val, raw_val)

    results.sort(key=rank_key, reverse=True)

    # Format results for API response. Show all disulfide variants in parallel.
    formatted_results = []
    for r in results:
        variants = r.get("all_variants")
        rows = variants if isinstance(variants, list) and variants else [r]
        for row in rows:
            z = row["z"]
            frag_name = row.get("frag_name", f"{ion_type}{frag_len}")
            row_loss_formula = row.get("loss_formula", "")
            row_loss_count = int(row.get("loss_count", 0) or 0)
            loss = (
                neutral_loss_label(row_loss_count, row_loss_formula)
                if row_loss_formula and row_loss_count
                else ""
            )
            label = f"{frag_name}{loss}^{z}+"

            dist = row.get("dist_plot")
            theory_mz = (
                dist[:, 0].tolist()
                if isinstance(dist, np.ndarray) and dist.size
                else []
            )
            theory_int = (
                dist[:, 1].tolist()
                if isinstance(dist, np.ndarray) and dist.size
                else []
            )

            # Use anchor_theory_mz if available, otherwise use expected_theory_mz
            theory_anchor_mz = row.get("anchor_theory_mz") or row.get(
                "expected_theory_mz"
            )

            formatted_results.append(
                {
                    "label": label,
                    "ion_type": row.get("ion_type", ""),
                    "frag_name": frag_name,
                    "frag_len": row.get("frag_len"),
                    "charge": int(z),
                    "loss_formula": row.get("loss_formula", ""),
                    "loss_count": row.get("loss_count", 0),
                    "h_transfer": row.get("h_transfer", 0),
                    "variant_suffix": row.get("variant_suffix", ""),
                    "variant_type": row.get("variant_type", ""),
                    "ok": bool(row.get("ok", False)),
                    "reason": row.get("reason", ""),
                    "formula": row.get("formula", ""),
                    "mono_mass": row.get("mono_mass"),
                    "avg_mass": row.get("avg_mass"),
                    "raw_cosine": row.get("raw_cosine_preanchor"),
                    "final_cosine": row.get("final_cosine"),
                    "isodec_css": row.get("isodec_css"),
                    "isodec_accepted": bool(row.get("isodec_accepted", False)),
                    "isodec_detail": row.get("isodec_detail"),
                    "fragments_gate": row.get("fragments_gate"),
                    "anchor_theory_mz": theory_anchor_mz,
                    "anchor_obs_mz": row.get("anchor_obs_mz"),
                    "anchor_ppm": row.get("anchor_ppm"),
                    "anchor_within_ppm": bool(row.get("anchor_within_ppm", False)),
                    "obs_int": row.get("obs_int"),
                    "obs_rel_int": row.get("obs_rel_int"),
                    "theory_mz": theory_mz,
                    "theory_int": theory_int,
                    "diagnostic_steps": row.get("diagnostic_steps", []),
                    "theory_matches": row.get("theory_matches", []),
                }
            )

    formatted_results.sort(key=rank_key, reverse=True)

    best_formatted = formatted_results[0] if formatted_results else None

    return {
        "ion_spec": spec,
        "parsed": {
            "ion_type": ion_type,
            "frag_len": frag_len,
            "loss_formula": loss_formula,
            "loss_count": loss_count,
            "charge": charge,
        },
        "h_transfer": h_transfer,
        "charges_scanned": charges,
        "results": formatted_results,
        "best": best_formatted,
        "spectrum_mz": raw_spectrum[:, 0].tolist() if raw_spectrum.size else [],
        "spectrum_int": raw_spectrum[:, 1].tolist() if raw_spectrum.size else [],
        "theory_mz": best_formatted.get("theory_mz", [])
        if isinstance(best_formatted, dict)
        else [],
        "theory_int": best_formatted.get("theory_int", [])
        if isinstance(best_formatted, dict)
        else [],
    }


def run_diagnose_mode(residues, spectrum, isodec_config) -> None:
    if not cfg.DIAGNOSE_ION_SPEC:
        raise ValueError(
            'Set DIAGNOSE_ION_SPEC (e.g., "c7^2+" or "z12-2H2O^3+") when using PLOT_MODE="diagnose".'
        )

    raw_spectrum = np.array(spectrum, dtype=float, copy=True)
    if bool(cfg.ENABLE_FRAGMENT_INTENSITY_CAP):
        tol_ppm = (
            float(cfg.MATCH_TOL_PPM)
            if cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM)
        )
        mz_min_cap = (
            None
            if cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN)
        )
        mz_max_cap = (
            None
            if cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX is None
            else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX)
        )
        min_hits = int(max(0, cfg.FRAGMENT_INTENSITY_CAP_MIN_HITS))
        cap, hits = compute_fragment_intensity_cap(
            residues,
            spectrum[:, 0],
            spectrum[:, 1],
            tol_ppm=float(tol_ppm),
            mz_min=mz_min_cap,
            mz_max=mz_max_cap,
        )
        if hits >= min_hits and cap > 0:
            n_before = int(len(spectrum))
            spectrum = strip_peaks_above_intensity_cap(spectrum, cap=float(cap))
            removed = max(0, n_before - int(len(spectrum)))
            if removed > 0 or bool(cfg.FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(
                    f"Fragment intensity-cap strip: cap={cap:.3g} ({hits} windows), removed {removed} peaks"
                )
        else:
            if bool(cfg.FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(
                    f"Fragment intensity-cap strip: skipped (cap={cap:.3g}, hits={hits} < {min_hits})"
                )

    ion_type, frag_len, loss_formula, loss_count, charge = parse_fragment_spec(
        cfg.DIAGNOSE_ION_SPEC
    )
    charges = (
        [int(charge)]
        if charge is not None
        else list(range(int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1))
    )
    if charge is None and not bool(cfg.DIAGNOSE_SCAN_CHARGES):
        raise ValueError(
            "Ion spec has no charge; set DIAGNOSE_SCAN_CHARGES=True or include ^z+ (e.g., c7^2+)."
        )

    try:
        h_transfer = int(cfg.DIAGNOSE_H_TRANSFER)
    except Exception as e:
        raise ValueError(
            "DIAGNOSE_H_TRANSFER must be an integer in {-2,-1,0,1,2}."
        ) from e
    if h_transfer not in (-2, -1, 0, 1, 2):
        raise ValueError("DIAGNOSE_H_TRANSFER must be an integer in {-2,-1,0,1,2}.")

    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]

    print("=== Ion diagnostic ===")
    print(f"Spec: {cfg.DIAGNOSE_ION_SPEC}")
    print(
        f"Parsed: ion_type={ion_type}, frag_len={frag_len}, loss={loss_formula or 'none'} x{loss_count or 0}"
    )
    print(f"H transfer: {h_transfer:+d} H+")
    print(f"Charge(s): {charges}")

    diagnose_dir = Path(__file__).parent / "diagnose_outputs"
    spec_safe = sanitize_filename(cfg.DIAGNOSE_ION_SPEC)
    base = f"diagnose_scan{int(cfg.SCAN)}_{spec_safe}_h{h_transfer:+d}"
    base = sanitize_filename(base)
    summary_path = (
        Path(cfg.DIAGNOSE_CSV_SUMMARY_PATH)
        if cfg.DIAGNOSE_CSV_SUMMARY_PATH
        else (diagnose_dir / f"{base}_summary.csv")
    )
    peaks_path = (
        Path(cfg.DIAGNOSE_CSV_PEAKS_PATH)
        if cfg.DIAGNOSE_CSV_PEAKS_PATH
        else (diagnose_dir / f"{base}_peaks.csv")
    )

    results = []
    for z in charges:
        r = diagnose_candidate(
            residues=residues,
            ion_type=ion_type,
            frag_len=frag_len,
            z=int(z),
            loss_formula=loss_formula,
            loss_count=int(loss_count),
            h_transfer=h_transfer,
            spectrum_mz=spectrum_mz,
            spectrum_int=spectrum_int,
            match_tol_ppm=float(cfg.MATCH_TOL_PPM),
            min_obs_rel_int=float(cfg.MIN_OBS_REL_INT),
            rel_intensity_cutoff=float(cfg.REL_INTENSITY_CUTOFF),
            mz_min=cfg.MZ_MIN,
            mz_max=cfg.MZ_MAX,
            isodec_config=isodec_config,
        )

        # Print diagnostic steps for each charge state
        print(f"\n--- Charge {z}+ ---")
        for step in r.get("diagnostic_steps", []):
            status_icon = (
                "PASS"
                if step["status"] == "pass"
                else "FAIL"
                if step["status"] == "fail"
                else "INFO"
            )
            print(f"{status_icon} {step['step']}: {step['status']} - {step['details']}")

        if r.get("ok", False):
            print("PASS Overall: PASS")
        else:
            print(f"FAIL Overall: FAIL - {r.get('reason', 'Unknown reason')}")

        results.append(r)

    def rank_key(d: dict):
        css = d.get("isodec_css", float("nan"))
        raw = d.get("raw_cosine_preanchor", 0.0)
        ok = 1 if d.get("ok") else 0
        css_val = css if np.isfinite(css) else -1.0
        return (ok, css_val, raw)

    results.sort(key=rank_key, reverse=True)

    if bool(cfg.DIAGNOSE_EXPORT_CSV):
        summary_rows = []
        peaks_rows = []
        for r in results:
            z = r["z"]
            frag_name = r.get("frag_name", f"{ion_type}{frag_len}")
            loss = ""
            if loss_formula and loss_count:
                loss = neutral_loss_label(int(loss_count), loss_formula)
            label = f"{frag_name}{loss}^{z}+"
            formula = r.get("formula", "")
            mono_mass = r.get("mono_mass", "")
            avg_mass = r.get("avg_mass", "")

            detail = (
                r.get("isodec_detail")
                if isinstance(r.get("isodec_detail"), dict)
                else {}
            )
            summary_rows.append(
                {
                    "spec": str(cfg.DIAGNOSE_ION_SPEC),
                    "label": label,
                    "formula": formula,
                    "mono_mass": mono_mass,
                    "avg_mass": avg_mass,
                    "ion_type": r.get("ion_type", ""),
                    "frag_name": frag_name,
                    "frag_len": r.get("frag_len", ""),
                    "z": int(z),
                    "loss_formula": r.get("loss_formula", ""),
                    "loss_count": r.get("loss_count", ""),
                    "h_transfer": r.get("h_transfer", ""),
                    "variant_type": r.get("variant_type", ""),
                    "variant_suffix": r.get("variant_suffix", ""),
                    "variant_pass_count": r.get("variant_pass_count", ""),
                    "ok": bool(r.get("ok", False)),
                    "reason": r.get("reason", ""),
                    "raw_cosine_preanchor": r.get("raw_cosine_preanchor", ""),
                    "anchor_theory_mz": r.get("anchor_theory_mz", ""),
                    "anchor_obs_mz": r.get("anchor_obs_mz", ""),
                    "anchor_ppm": r.get("anchor_ppm", ""),
                    "anchor_within_ppm": bool(r.get("anchor_within_ppm", False)),
                    "obs_idx": r.get("obs_idx", ""),
                    "obs_int": r.get("obs_int", ""),
                    "obs_rel_int": r.get("obs_rel_int", ""),
                    "isodec_css": r.get("isodec_css", ""),
                    "isodec_accepted": bool(r.get("isodec_accepted", False)),
                    "isodec_local_centroids_n": detail.get("local_centroids_n", ""),
                    "isodec_matched_peaks_n": detail.get("matched_peaks_n", ""),
                    "isodec_minpeaks_effective": detail.get("minpeaks_effective", ""),
                    "isodec_areacovered": detail.get("areacovered", ""),
                    "isodec_topthree": detail.get("topthree", ""),
                    "match_tol_ppm": float(cfg.MATCH_TOL_PPM),
                    "min_obs_rel_int": float(cfg.MIN_OBS_REL_INT),
                    "rel_intensity_cutoff": float(cfg.REL_INTENSITY_CUTOFF),
                    "mz_min": "" if cfg.MZ_MIN is None else float(cfg.MZ_MIN),
                    "mz_max": "" if cfg.MZ_MAX is None else float(cfg.MZ_MAX),
                }
            )

            for p in (
                r.get("theory_matches", [])
                if isinstance(r.get("theory_matches"), list)
                else []
            ):
                peaks_rows.append(
                    {
                        "spec": str(cfg.DIAGNOSE_ION_SPEC),
                        "label": label,
                        "formula": formula,
                        "mono_mass": mono_mass,
                        "avg_mass": avg_mass,
                        "z": int(z),
                        "h_transfer": r.get("h_transfer", ""),
                        "variant_type": r.get("variant_type", ""),
                        "variant_suffix": r.get("variant_suffix", ""),
                        "variant_pass_count": r.get("variant_pass_count", ""),
                        "theory_mz": p.get("theory_mz", ""),
                        "theory_int": p.get("theory_int", ""),
                        "obs_mz": p.get("obs_mz", ""),
                        "ppm": p.get("ppm", ""),
                        "obs_int": p.get("obs_int", ""),
                        "within": p.get("within", ""),
                        "obs_idx": p.get("obs_idx", ""),
                    }
                )

        write_csv(
            summary_path,
            [
                "spec",
                "label",
                "formula",
                "mono_mass",
                "avg_mass",
                "ion_type",
                "frag_name",
                "frag_len",
                "z",
                "loss_formula",
                "loss_count",
                "h_transfer",
                "variant_type",
                "variant_suffix",
                "variant_pass_count",
                "ok",
                "reason",
                "raw_cosine_preanchor",
                "anchor_theory_mz",
                "anchor_obs_mz",
                "anchor_ppm",
                "anchor_within_ppm",
                "obs_idx",
                "obs_int",
                "obs_rel_int",
                "isodec_css",
                "isodec_accepted",
                "isodec_local_centroids_n",
                "isodec_matched_peaks_n",
                "isodec_minpeaks_effective",
                "isodec_areacovered",
                "isodec_topthree",
                "match_tol_ppm",
                "min_obs_rel_int",
                "rel_intensity_cutoff",
                "mz_min",
                "mz_max",
            ],
            summary_rows,
        )
        write_csv(
            peaks_path,
            [
                "spec",
                "label",
                "formula",
                "mono_mass",
                "avg_mass",
                "z",
                "h_transfer",
                "variant_type",
                "variant_suffix",
                "variant_pass_count",
                "theory_mz",
                "theory_int",
                "obs_mz",
                "ppm",
                "obs_int",
                "within",
                "obs_idx",
            ],
            peaks_rows,
        )
        print(f"Wrote CSV: {summary_path}")
        print(f"Wrote CSV: {peaks_path}")

    for r in results:
        z = r["z"]
        frag_name = r.get("frag_name", f"{ion_type}{frag_len}")
        loss = ""
        if loss_formula and loss_count:
            loss = neutral_loss_label(int(loss_count), loss_formula)
        label = f"{frag_name}{loss}^{z}+"
        formula = r.get("formula", "")
        mono_mass = r.get("mono_mass", "")
        avg_mass = r.get("avg_mass", "")
        raw = r.get("raw_cosine_preanchor", 0.0)
        css = r.get("isodec_css", None)
        css_txt = (
            f"{css:.3f}"
            if isinstance(css, (int, float)) and np.isfinite(css)
            else "n/a"
        )
        mono_txt = (
            f"{mono_mass:.6f}"
            if isinstance(mono_mass, (int, float)) and np.isfinite(mono_mass)
            else str(mono_mass)
        )
        avg_txt = (
            f"{avg_mass:.6f}"
            if isinstance(avg_mass, (int, float)) and np.isfinite(avg_mass)
            else str(avg_mass)
        )
        print(
            f"- {label}\tok={r['ok']}\treason={r['reason']}\tformula={formula}\t"
            f"mono_mass={mono_txt}\tavg_mass={avg_txt}\t"
            f"rawcos={raw:.3f}\tcss={css_txt}"
        )
        if r.get("anchor_within_ppm"):
            print(
                "  anchor: "
                f"theory={r['anchor_theory_mz']:.6f}\t"
                f"obs={r['anchor_obs_mz']:.6f}\t"
                f"ppm={r.get('anchor_ppm', 0.0):.1f}\t"
                f"I={r.get('obs_int', 0.0):.3g}\t"
                f"relI={100.0 * r.get('obs_rel_int', 0.0):.2f}%"
            )
        detail = r.get("isodec_detail")
        if isinstance(detail, dict):
            print(
                "  isodec: "
                f"matched={detail.get('matched_peaks_n')}/{detail.get('local_centroids_n')}\t"
                f"minpeaks={detail.get('minpeaks_effective')}\t"
                f"areacovered={detail.get('areacovered'):.3f}\t"
                f"minarea={detail.get('minareacovered'):.3f}\t"
                f"top={detail.get('topthree')}\t"
                f"css_thresh={detail.get('css_thresh'):.2f}"
            )

        if cfg.DIAGNOSE_MAX_TABLE_ROWS and isinstance(r.get("theory_matches"), list):
            matches = r["theory_matches"]
            matches_sorted = sorted(
                matches, key=lambda x: (x["within"], x["obs_int"]), reverse=True
            )
            print("  peaks (theory_mz -> obs_mz, ppm, I, theory_I):")
            for row in matches_sorted[: int(cfg.DIAGNOSE_MAX_TABLE_ROWS)]:
                flag = "*" if row["within"] else " "
                print(
                    f"   {flag} {row['theory_mz']:.6f} -> {row['obs_mz']:.6f}\tppm={row['ppm']:+.1f}\t"
                    f"I={row['obs_int']:.3g}\tI_theory={row['theory_int']:.3g}"
                )

    best = results[0] if results else None
    if bool(cfg.DIAGNOSE_SHOW_PLOT):
        if isinstance(best, dict) and isinstance(best.get("dist_plot"), np.ndarray):
            z = best["z"]
            frag_name = best.get("frag_name", f"{ion_type}{frag_len}")
            loss = ""
            if loss_formula and loss_count:
                loss = neutral_loss_label(int(loss_count), loss_formula)
            label = f"{frag_name}{loss}^{z}+"
            plot_overlay(
                spectrum,
                [(best["dist_plot"], "tab:purple", f"diagnose {label}")],
                mz_min=None if cfg.MZ_MIN is None else float(cfg.MZ_MIN),
                mz_max=None if cfg.MZ_MAX is None else float(cfg.MZ_MAX),
                noise_cutoff=(
                    float(np.max(spectrum[:, 1])) * float(cfg.MIN_OBS_REL_INT)
                )
                if float(cfg.MIN_OBS_REL_INT) > 0
                else None,
            )
        else:
            try:
                fallback_charge = (
                    best.get("z")
                    if isinstance(best, dict) and best.get("z")
                    else (charges[0] if charges else 1)
                )
                frag_name, base_comp = ion_composition_from_sequence(
                    residues, ion_type, frag_len, amidated=cfg.AMIDATED
                )
                comp = apply_neutral_loss(base_comp, loss_formula, loss_count)
                fallback_suffix = ""
                fallback_comp = comp
                dist_fallback = None
                variants = get_disulfide_logic(ion_type, frag_len, len(residues))
                if not variants:
                    variants = [("", None)]
                for suffix, shift in variants:
                    try:
                        comp_try = comp + shift if shift is not None else comp
                        dist_try = theoretical_isodist_from_comp(
                            comp_try, int(fallback_charge)
                        )
                    except Exception:
                        continue
                    if dist_try.size:
                        fallback_suffix = suffix
                        fallback_comp = comp_try
                        dist_fallback = dist_try
                        break
                if dist_fallback is None:
                    dist_fallback = theoretical_isodist_from_comp(
                        fallback_comp, int(fallback_charge)
                    )
                if int(h_transfer) != 0 and dist_fallback.size:
                    dist_fallback = dist_fallback.copy()
                    dist_fallback[:, 0] += (
                        float(h_transfer) * float(cfg.H_TRANSFER_MASS)
                    ) / float(fallback_charge)
                if dist_fallback.size:
                    dist_plot = dist_fallback.copy()
                    if raw_spectrum.size:
                        max_obs = float(np.max(raw_spectrum[:, 1]))
                        max_theory = (
                            float(np.max(dist_plot[:, 1])) if dist_plot.size else 0.0
                        )
                        if max_theory > 0:
                            dist_plot[:, 1] *= max_obs / max_theory
                    if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
                        mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
                        mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
                        dist_plot = dist_plot[
                            (dist_plot[:, 0] >= mz_min) & (dist_plot[:, 0] <= mz_max)
                        ]
                    loss = (
                        neutral_loss_label(int(loss_count), loss_formula)
                        if loss_formula and loss_count
                        else ""
                    )
                    label = (
                        f"{frag_name}{fallback_suffix}{loss}^{int(fallback_charge)}+"
                    )
                    plot_overlay(
                        raw_spectrum,
                        [(dist_plot, "tab:purple", f"diagnose fallback {label}")],
                        mz_min=None if cfg.MZ_MIN is None else float(cfg.MZ_MIN),
                        mz_max=None if cfg.MZ_MAX is None else float(cfg.MZ_MAX),
                    )
            except Exception as e:
                print(f"Fallback plot skipped: {e}")
