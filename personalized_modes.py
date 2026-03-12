from __future__ import annotations

from copy import copy
from pathlib import Path

import numpy as np
import personalized_config as cfg
from personalized_match import (
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
        precursor_tol_ppm if precursor_tol_ppm is not None else getattr(cfg, "PRECURSOR_MATCH_TOL_PPM", cfg.MATCH_TOL_PPM)
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
    score_w_sum = score_w_css + score_w_cov + score_w_ppm + score_w_spacing + score_w_intensity
    if score_w_sum <= 0:
        score_w_css, score_w_cov, score_w_ppm, score_w_spacing, score_w_intensity = 0.45, 0.25, 0.20, 0.10, 0.25
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
    max_anchor_abs_ppm = float(getattr(cfg, "PRECURSOR_MAX_ANCHOR_ABS_PPM", precursor_tol_ppm * 1.5))
    max_residual_rmse_ppm = float(getattr(cfg, "PRECURSOR_MAX_RESIDUAL_RMSE_PPM", precursor_tol_ppm))
    ppm_sigma = float(getattr(cfg, "PRECURSOR_PPM_SIGMA", precursor_tol_ppm))
    if ppm_sigma <= 0:
        ppm_sigma = max(precursor_tol_ppm, 1.0)
    ambiguity_margin = float(getattr(cfg, "PRECURSOR_AMBIGUITY_MARGIN", 0.03))
    ambiguity_guard = bool(getattr(cfg, "PRECURSOR_ENABLE_AMBIGUITY_GUARD", True))
    ambiguous_result: dict | None = None

    def _build_anchor_indices(local: np.ndarray, target_mz: float, top_k: int, tol_ppm: float) -> list[int]:
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
        tol_da = abs(float(target_mz)) * float(tol_ppm) * 1e-6 if float(target_mz) > 0 else 0.0
        if tol_da > 0:
            in_tol = np.where(np.abs(local_mz - float(target_mz)) <= tol_da)[0]
            if in_tol.size:
                idx_sorted = in_tol[np.argsort(np.abs(local_mz[in_tol] - float(target_mz)))]
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
            matched_sum = float(sum(max(0.0, float(r.get("theory_int", 0.0))) for r in matches))
            coverage = float(np.clip(matched_sum / theory_sum, 0.0, 1.0))
        else:
            coverage = 0.0

        if matches:
            weights = np.array([max(float(r.get("theory_int", 0.0)), 1e-12) for r in matches], dtype=float)
            residual_ppm = np.array([float(r.get("residual_ppm", 0.0)) for r in matches], dtype=float)
            ppm_rmse = float(np.sqrt(np.average(residual_ppm * residual_ppm, weights=weights)))
        else:
            ppm_rmse = float("inf")

        residual_score = 0.0 if not np.isfinite(ppm_rmse) else float(np.exp(-ppm_rmse / ppm_sigma))
        anchor_score = float(np.exp(-abs(anchor_ppm_abs) / ppm_sigma))
        ppm_consistency = float(np.clip((0.70 * residual_score) + (0.30 * anchor_score), 0.0, 1.0))

        if len(matches) >= 2:
            obs = np.array([float(r.get("obs_mz", 0.0)) for r in matches], dtype=float)
            pred = np.array([float(r.get("pred_mz", 0.0)) for r in matches], dtype=float)
            diff_err = np.diff(obs) - np.diff(pred)
            spacing_rmse_da = float(np.sqrt(np.mean(diff_err * diff_err))) if diff_err.size else 0.0
            anchor_tol_da = abs(float(anchor_theory_mz)) * float(precursor_tol_ppm) * 1e-6
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

    search_spectrum, search_window = _filter_spectrum_by_requested_mz_window(np.array(spectrum, dtype=float, copy=True))
    search_mz = np.asarray(search_spectrum[:, 0], dtype=float) if search_spectrum.size else np.asarray([], dtype=float)
    search_int = np.asarray(search_spectrum[:, 1], dtype=float) if search_spectrum.size else np.asarray([], dtype=float)
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
    _use_monoisotopic = str(getattr(cfg, "ANCHOR_MODE", "most_intense")).lower() == "monoisotopic"
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

                theory_mz = float(dist_shifted[0, 0]) if _use_monoisotopic else float(dist_shifted[get_anchor_idx(dist_shifted), 0])
                anchor_window_mask = (
                    (search_mz >= theory_mz - anchor_search_da)
                    & (search_mz <= theory_mz + anchor_search_da)
                )
                if not np.any(anchor_window_mask):
                    continue

                anchor_window_idx = np.where(anchor_window_mask)[0]
                ranked_anchor_idx = anchor_window_idx[np.argsort(search_int[anchor_window_idx])[::-1]]
                nearest_anchor_idx = int(
                    anchor_window_idx[np.argmin(np.abs(search_mz[anchor_window_idx] - theory_mz))]
                )
                selected_anchor_idx: list[int] = []
                for idx in [nearest_anchor_idx, *ranked_anchor_idx[:anchor_top_k].tolist()]:
                    i = int(idx)
                    if i not in selected_anchor_idx:
                        selected_anchor_idx.append(i)

                left_span = max(float(theory_mz - dist_shifted[0, 0]), 0.0)
                right_span = max(float(dist_shifted[-1, 0] - theory_mz), 0.0)
                anchor_pad_da = max(abs(float(theory_mz)) * float(precursor_tol_ppm) * 3e-6, 0.5)
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
                            local_mask = (
                                (search_mz >= obs_anchor_mz + local_lb)
                                & (search_mz <= obs_anchor_mz + local_ub)
                            )
                            if not np.any(local_mask):
                                continue
                            local = np.column_stack((search_mz[local_mask], search_int[local_mask]))
                        if local.ndim != 2 or local.shape[1] != 2:
                            continue

                        local_max_int = float(np.max(local[:, 1])) if local.shape[0] > 0 else 0.0
                        if local_max_int <= 0:
                            continue

                        anchor_indices = _build_anchor_indices(local, theory_mz, anchor_top_k, precursor_tol_ppm)
                        if not anchor_indices:
                            anchor_indices = [int(np.argmin(np.abs(local[:, 0] - obs_anchor_mz)))]

                        for anchor_idx in anchor_indices:
                            anchor_idx = int(anchor_idx)
                            obs_mz_seed = float(local[anchor_idx, 0])
                            obs_anchor_int = float(local[anchor_idx, 1])
                            obs_mz_eval = float(obs_mz_seed)
                            if cfg.ENABLE_ISODEC_RULES and local_isodec_config is not None:
                                accepted_model, isodec_css, shifted_peak = isodec_css_and_accept(
                                    local, dist_shifted, z=int(z), peakmz=obs_mz_seed, config=local_isodec_config
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
                                isodec_css = css_similarity(y_obs_seed, dist_shifted[:, 1])
                                accepted_model = isodec_css >= float(cfg.MIN_COSINE)

                            obs_idx_eval = int(nearest_peak_index(local[:, 0], obs_mz_eval))
                            if obs_idx_eval >= 0:
                                obs_anchor_int = float(local[obs_idx_eval, 1])
                            anchor_rel_int = (float(obs_anchor_int) / local_max_int) if local_max_int > 0 else 0.0

                            shift_da = float(obs_mz_eval - theory_mz)
                            matches = _match_theory_local(local, dist_shifted, shift_da, precursor_tol_ppm)
                            anchor_ppm_abs = abs(((float(obs_mz_eval) - theory_mz) / theory_mz) * 1e6) if theory_mz else 0.0
                            comp = _composite_components(
                                css=float(isodec_css),
                                matches=matches,
                                dist_shifted=dist_shifted,
                                anchor_ppm_abs=float(anchor_ppm_abs),
                                anchor_theory_mz=float(theory_mz),
                                intensity_ratio=float(anchor_rel_int),
                            )
                            ppm = ((float(obs_mz_eval) - theory_mz) / theory_mz) * 1e6 if theory_mz else 0.0
                            accepted = bool(
                                bool(accepted_model)
                                and float(isodec_css) >= float(cfg.MIN_COSINE)
                                and float(anchor_rel_int) >= float(anchor_min_rel_int)
                                and float(comp["coverage"]) >= float(min_coverage)
                                and float(anchor_ppm_abs) <= float(max_anchor_abs_ppm)
                                and len(matches) > 0
                            )
                            if len(matches) >= 2 and np.isfinite(comp["ppm_rmse"]):
                                accepted = accepted and float(comp["ppm_rmse"]) <= float(max_residual_rmse_ppm)
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
                                "spacing_consistency": float(comp["spacing_consistency"]),
                                "spacing_rmse_da": float(comp["spacing_rmse_da"]),
                                "intensity_prior": float(comp["intensity_prior"]),
                                "anchor_rel_int": float(anchor_rel_int),
                                "match_count": int(len(matches)),
                                "anchor_seed_mz": float(obs_mz_seed),
                                "anchor_target_mz": float(theory_mz),
                                "local_window_min": float(obs_anchor_mz + local_lb),
                                "local_window_max": float(obs_anchor_mz + local_ub),
                            }
                            if local_best_for_state is None or float(candidate["composite_score"]) > float(
                                local_best_for_state.get("composite_score", -1.0)
                            ):
                                local_best_for_state = candidate

                    return local_best_for_state

                best_candidate = execute_hybrid_strategy(_score_precursor_candidate)
                if best_candidate is None:
                    continue
                all_candidates.append(best_candidate)
                prev = best_by_charge.get(int(z))
                prev_score = float(prev.get("composite_score", prev.get("css", -1.0))) if prev is not None else -1.0
                if prev is None or float(best_candidate.get("composite_score", 0.0)) > prev_score:
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
                top_candidates = [dict(c) for c in accepted_ranked[: min(3, len(accepted_ranked))]]
                ambiguous_result = {
                    "status": "ambiguous",
                    "accepted": False,
                    "ambiguous": True,
                    "iteration": 1,
                    "window_min": float(top_candidates[0].get("local_window_min", top_candidates[0].get("anchor_theory_mz", 0.0))),
                    "window_max": float(top_candidates[0].get("local_window_max", top_candidates[0].get("anchor_theory_mz", 0.0))),
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
            best_composite_score = float(best_candidate.get("composite_score", best_css))
            best_coverage = float(best_candidate.get("coverage", 0.0) or 0.0)
            best_ppm_rmse = (
                float(best_candidate.get("ppm_rmse"))
                if best_candidate.get("ppm_rmse") is not None
                else None
            )
            best_obs_mz = float(best_candidate["obs_mz"])
            best_theory_mz = float(best_candidate["anchor_theory_mz"])
            best_theory_dist = best_candidate["dist"]
            shift_ppm = ((best_obs_mz - best_theory_mz) / best_theory_mz) * 1e6 if best_theory_mz else 0.0
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
        min_calibration_score = float(getattr(cfg, "PRECURSOR_CALIBRATION_MIN_SCORE", 0.70))
        min_calibration_coverage = float(
            getattr(cfg, "PRECURSOR_CALIBRATION_MIN_COVERAGE", getattr(cfg, "PRECURSOR_MIN_COVERAGE", 0.30))
        )
        max_calibration_ppm_rmse = float(
            getattr(
                cfg,
                "PRECURSOR_CALIBRATION_MAX_PPM_RMSE",
                getattr(cfg, "PRECURSOR_MAX_RESIDUAL_RMSE_PPM", precursor_tol_ppm),
            )
        )
        max_calibration_shift_ppm = float(getattr(cfg, "PRECURSOR_CALIBRATION_MAX_SHIFT_PPM", 100.0))
        if float(best_composite_score) < float(min_calibration_score):
            calibration_block_reasons.append(f"score<{min_calibration_score:.2f}")
        if float(best_coverage) < float(min_calibration_coverage):
            calibration_block_reasons.append(f"coverage<{min_calibration_coverage:.2f}")
        if best_ppm_rmse is not None and np.isfinite(best_ppm_rmse) and float(best_ppm_rmse) > float(max_calibration_ppm_rmse):
            calibration_block_reasons.append(f"ppm_rmse>{max_calibration_ppm_rmse:.1f}")
        if abs(float(shift_ppm)) > float(max_calibration_shift_ppm):
            calibration_block_reasons.append(f"abs_shift>{max_calibration_shift_ppm:.1f}ppm")
        calibration_safe = len(calibration_block_reasons) == 0

    if match_found and bool(apply_calibration) and calibration_safe:
        print(f"Applying lock-mass calibration: shift {-shift_ppm:.2f} ppm")
        calibrated_spectrum[:, 0] = calibrated_spectrum[:, 0] / (1.0 + (shift_ppm / 1e6))
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
            all_theory_dist = _scale_dist_to_obs(all_theory_dist, float(first_anchor_int))
        best_theory_dist = all_theory_dist

    candidates = sorted(
        best_by_charge.values(),
        key=lambda d: float(d.get("composite_score", d.get("css", -1.0))),
        reverse=True,
    )
    plot_window = None
    if match_found and best_theory_dist is not None and isinstance(best_theory_dist, np.ndarray) and best_theory_dist.size:
        plot_window = (float(best_theory_dist[0, 0]) - 5.0, float(best_theory_dist[-1, 0]) + 5.0)

    return {
        "spectrum": calibrated_spectrum,
        "spectrum_mz": np.asarray(calibrated_spectrum[:, 0], dtype=float),
        "spectrum_int": np.asarray(calibrated_spectrum[:, 1], dtype=float),
        "match_found": bool(match_found),
        "search_status": "matched" if match_found else ("ambiguous" if ambiguous_result is not None else "not_found"),
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
        "ambiguous_candidates": list(ambiguous_result.get("top_candidates") or []) if ambiguous_result is not None else [],
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
            mz_max = float(window[1]) if window else float(best_theory_dist[-1, 0]) + 5.0
        else:
            label = (
                f"Precursor theories (z={int(cfg.PRECURSOR_MIN_CHARGE)}-{int(cfg.PRECURSOR_MAX_CHARGE)})"
            )
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


def _filter_spectrum_by_requested_mz_window(spectrum: np.ndarray) -> tuple[np.ndarray, dict | None]:
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
        matched_sum = float(sum(max(0.0, float(r.get("theory_int", 0.0))) for r in matches))
        coverage = float(np.clip(matched_sum / theory_sum, 0.0, 1.0))
    else:
        coverage = 0.0

    if matches:
        weights = np.array([max(float(r.get("theory_int", 0.0)), 1e-12) for r in matches], dtype=float)
        residual_ppm = np.array([float(r.get("residual_ppm", 0.0)) for r in matches], dtype=float)
        ppm_rmse = float(np.sqrt(np.average(residual_ppm * residual_ppm, weights=weights)))
    else:
        ppm_rmse = float("inf")

    residual_score = 0.0 if not np.isfinite(ppm_rmse) else float(np.exp(-ppm_rmse / ppm_sigma))
    anchor_score = float(np.exp(-abs(anchor_ppm_abs) / ppm_sigma))
    ppm_consistency = float(np.clip((0.70 * residual_score) + (0.30 * anchor_score), 0.0, 1.0))

    if len(matches) >= 2:
        obs = np.array([float(r.get("obs_mz", 0.0)) for r in matches], dtype=float)
        pred = np.array([float(r.get("pred_mz", 0.0)) for r in matches], dtype=float)
        diff_err = np.diff(obs) - np.diff(pred)
        spacing_rmse_da = float(np.sqrt(np.mean(diff_err * diff_err))) if diff_err.size else 0.0
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


def run_charge_reduced_headless(residues, spectrum, isodec_config) -> dict:
    spectrum_copy = np.array(spectrum, dtype=float, copy=True)
    spectrum_mz = spectrum_copy[:, 0]
    spectrum_int = spectrum_copy[:, 1]
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0
    search_spectrum, search_window = _filter_spectrum_by_requested_mz_window(spectrum_copy)
    search_mz = np.asarray(search_spectrum[:, 0], dtype=float) if search_spectrum.size else np.asarray([], dtype=float)
    search_int = np.asarray(search_spectrum[:, 1], dtype=float) if search_spectrum.size else np.asarray([], dtype=float)

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
    max_anchor_abs_ppm = float(max_anchor_abs_ppm_cfg) if max_anchor_abs_ppm_cfg is not None else (match_tol_ppm * 1.5)
    max_residual_rmse_cfg = getattr(cfg, "CR_MAX_RESIDUAL_RMSE_PPM", None)
    max_residual_rmse_ppm = (
        float(max_residual_rmse_cfg) if max_residual_rmse_cfg is not None else float(match_tol_ppm)
    )
    ppm_sigma_cfg = getattr(cfg, "CR_PPM_SIGMA", None)
    ppm_sigma = float(ppm_sigma_cfg) if ppm_sigma_cfg is not None else float(match_tol_ppm)
    if ppm_sigma <= 0:
        ppm_sigma = max(match_tol_ppm, 1.0)
    ambiguity_guard = bool(getattr(cfg, "CR_ENABLE_AMBIGUITY_GUARD", True))
    ambiguity_margin = float(getattr(cfg, "CR_AMBIGUITY_MARGIN", 0.03))
    score_w_css = float(getattr(cfg, "CR_SCORE_W_CSS", 0.45))
    score_w_cov = float(getattr(cfg, "CR_SCORE_W_COVERAGE", 0.25))
    score_w_ppm = float(getattr(cfg, "CR_SCORE_W_PPM", 0.20))
    score_w_spacing = float(getattr(cfg, "CR_SCORE_W_SPACING", 0.10))
    score_w_intensity = float(getattr(cfg, "CR_SCORE_W_INTENSITY", 0.25))
    score_w_sum = score_w_css + score_w_cov + score_w_ppm + score_w_spacing + score_w_intensity
    if score_w_sum <= 0:
        score_w_css, score_w_cov, score_w_ppm, score_w_spacing, score_w_intensity = 0.45, 0.25, 0.20, 0.10, 0.25
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
                        (search_mz >= anchor_theory_mz - anchor_search_da)
                        & (search_mz <= anchor_theory_mz + anchor_search_da)
                    )
                    if not np.any(anchor_window_mask):
                        continue
                    window_idx = np.where(anchor_window_mask)[0]
                    obs_anchor_mz = float(search_mz[window_idx[np.argmax(search_int[window_idx])]])

                    spacing_sigma_cfg = getattr(cfg, "CR_SPACING_SIGMA_DA", None)
                    if spacing_sigma_cfg is None:
                        spacing_sigma_da = abs(anchor_theory_mz) * match_tol_ppm * 1e-6
                    else:
                        spacing_sigma_da = float(spacing_sigma_cfg)
                    if spacing_sigma_da <= 0:
                        spacing_sigma_da = max(abs(anchor_theory_mz) * match_tol_ppm * 1e-6, 1e-6)

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
                                (search_mz >= obs_anchor_mz - local_window_da)
                                & (search_mz <= obs_anchor_mz + local_window_da)
                            )
                            if not np.any(local_mask):
                                return None
                            local = np.column_stack((search_mz[local_mask], search_int[local_mask]))
                        if local.ndim != 2 or local.shape[1] != 2:
                            return None

                        cent_mz = np.asarray(local[:, 0], dtype=float)
                        cent_int = np.asarray(local[:, 1], dtype=float)
                        local_max_int = float(np.max(cent_int)) if cent_int.size else 0.0
                        if local_max_int <= 0:
                            return None

                        if cfg.ENABLE_ISODEC_RULES and isodec_config is not None:
                            accepted_model, css, shifted_peak = isodec_css_and_accept(
                                local, dist_state, z=z, peakmz=obs_anchor_mz, config=isodec_config
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

                        obs_mz = float(shifted_peak) if shifted_peak is not None else float(obs_anchor_mz)
                        obs_idx = nearest_peak_index(cent_mz, obs_mz)
                        if obs_idx < 0:
                            return None
                        obs_int = float(cent_int[obs_idx])
                        anchor_rel_int = float(obs_int / local_max_int) if local_max_int > 0 else 0.0
                        anchor_ppm = ((obs_mz - anchor_theory_mz) / anchor_theory_mz) * 1e6 if anchor_theory_mz else 0.0
                        anchor_ppm_abs = abs(float(anchor_ppm))
                        shift_da_local = float(obs_mz - anchor_theory_mz)
                        matches = _match_theory_local_monotonic(local, dist_state, shift_da_local, match_tol_ppm)
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
                            accepted = accepted and float(comp["ppm_rmse"]) <= float(max_residual_rmse_ppm)
                        if not accepted:
                            return None

                        dist_full = dist_state.copy()
                        dist_plot = dist_state.copy()
                        max_plot = float(np.max(dist_plot[:, 1])) if dist_plot.size else 0.0
                        if max_plot > 0 and obs_int > 0:
                            dist_plot[:, 1] *= obs_int / max_plot
                        keep = (
                            dist_plot[:, 1] >= max_plot * float(cfg.REL_INTENSITY_CUTOFF)
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
        grouped_candidates.setdefault(int(cand.get("search_obs_idx", -1)), []).append(cand)

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
                    item["shadowed_by"] = str(top.get("short_label", top.get("label", "")))
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
        key=lambda x: (float(x.get("score", x.get("css", 0.0))), float(x.get("obs_int", 0.0))),
        reverse=True,
    )
    ambiguous_candidates.sort(
        key=lambda x: (float(x.get("score", x.get("css", 0.0))), float(x.get("obs_int", 0.0))),
        reverse=True,
    )
    shadowed_candidates.sort(
        key=lambda x: (float(x.get("score", x.get("css", 0.0))), float(x.get("obs_int", 0.0))),
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
        "search_status": "matched" if accepted_candidates else ("ambiguous" if ambiguous_candidates else "not_found"),
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
        noise_cutoff=(obs_max * float(cfg.MIN_OBS_REL_INT)) if float(cfg.MIN_OBS_REL_INT) > 0 else None,
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

                for p in m.get("theory_matches", []) if isinstance(m.get("theory_matches"), list) else []:
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
                        dist[:, 0] += (float(h_shift) * float(cfg.H_TRANSFER_MASS)) / float(z)
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


def run_fragments_mode(residues, spectrum, isodec_config, emit_outputs: bool = True) -> dict:
    if cfg.FRAG_MIN_CHARGE <= 0 or cfg.FRAG_MAX_CHARGE <= 0 or cfg.FRAG_MIN_CHARGE > cfg.FRAG_MAX_CHARGE:
        raise ValueError("Set FRAG_MIN_CHARGE/FRAG_MAX_CHARGE to a valid positive range.")

    # Get current mode from config
    current_mode = str(cfg.PLOT_MODE).lower()

    if bool(cfg.ENABLE_FRAGMENT_INTENSITY_CAP):
        tol_ppm = float(cfg.MATCH_TOL_PPM) if cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM is None else float(
            cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM
        )
        mz_min_cap = None if cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN is None else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN)
        mz_max_cap = None if cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX is None else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX)
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
                print(f"Fragment intensity-cap strip: cap={cap:.3g} ({hits} windows), removed {removed} peaks")
        else:
            if bool(cfg.FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(f"Fragment intensity-cap strip: skipped (cap={cap:.3g}, hits={hits} < {min_hits})")

    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0

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
        allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_1H))
        allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_2H))
        for frag_len in range(1, n):
            # Get appropriate composition based on mode
            if current_mode == "complex_fragments":
                # Use interchain fragment composition
                frag_name, target_comp = get_interchain_fragment_composition(residues, ion_type, frag_len, amidated=cfg.AMIDATED)
            else:
                # Use regular fragment composition
                frag_name, target_comp = ion_composition_from_sequence(residues, ion_type, frag_len, amidated=cfg.AMIDATED)
            
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
                        match_tol_ppm=float(cfg.MATCH_TOL_PPM),
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

                if float(np.max(best_pred)) <= 0.0:
                    return None

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
                    if not within_ppm(obs_mz_c, mz_candidate, float(cfg.MATCH_TOL_PPM)):
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
                    return None
                ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6

                dist_plot = np.column_stack([sample_mzs.copy(), best_pred.copy()])
                dist_plot[:, 0] += obs_mz - anchor_theory_mz
                dist_plot[:, 1] *= obs_int / float(np.max(dist_plot[:, 1]))

                if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
                    mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
                    mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
                    dist_plot = dist_plot[(dist_plot[:, 0] >= mz_min) & (dist_plot[:, 0] <= mz_max)]
                    if dist_plot.size == 0:
                        return None

                max_plot = float(np.max(dist_plot[:, 1]))
                keep = dist_plot[:, 1] >= max_plot * float(cfg.REL_INTENSITY_CUTOFF)
                dist_plot = dist_plot[keep]
                if dist_plot.size == 0:
                    return None

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
                    accepted, isodec_css, shifted_peak = isodec_css_and_accept(
                        local_centroids, dist_plot, z=z, peakmz=obs_mz, config=isodec_config
                    )
                    if not accepted:
                        return None
                    if shifted_peak is not None:
                        old_obs_mz = obs_mz
                        obs_mz_new = float(shifted_peak)
                        obs_idx = nearest_peak_index(spectrum_mz, obs_mz_new)
                        obs_mz = float(spectrum_mz[obs_idx])
                        obs_int = float(spectrum_int[obs_idx])
                        obs_rel_int = float(obs_int / obs_max) if obs_max > 0 else 0.0
                        ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6
                        dist_plot[:, 0] += obs_mz - old_obs_mz

                frag_id = f"{frag_id_base}{loss_suffix}"
                obs_rel_int = float(obs_int / obs_max) if obs_max > 0 else 0.0
                label_parts = [f"{frag_id}^{z}+", f"{ppm:.1f} ppm", f"css={isodec_css:.3f}"]
                if best_model != "neutral":
                    h_pct = 100.0 * float(best_weights.get("+H", 0.0) + best_weights.get("-H", 0.0))
                    h2_pct = 100.0 * float(best_weights.get("+2H", 0.0) + best_weights.get("-2H", 0.0))
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
                    "best_model": best_model,
                    "obs_idx": int(obs_idx),
                    "obs_mz": float(obs_mz),
                    "obs_int": obs_int,
                    "obs_rel_int": obs_rel_int,
                    "anchor_theory_mz": float(anchor_theory_mz),
                    "ppm": float(ppm),
                    "score": float(isodec_css),
                    "raw_score": float(best_score),
                    "neutral_score": float(neutral_score),
                    "h_weights": best_weights,
                    "dist": dist_plot,
                    "label": label,
                    "color": ion_colors.get(ion_type, "tab:purple"),
                }

            variant_shift_map = {suffix: shift for suffix, shift in cys_variants}

            for z in range(int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1):
                neutral_candidates = []
                for variant_suffix, shift_comp in cys_variants:
                    if shift_comp:
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

                variant_type = best_neutral.get("variant_type") or variant_type_from_suffix(best_suffix)

                # Loss variants only after neutral passes final filter.
                for loss_suffix, loss_comp in neutral_loss_variants(variant_comp, ion_series_letter=series):
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
        if m["score"] > cur["score"] or (m["score"] == cur["score"] and m["obs_int"] > cur["obs_int"]):
            best_by_obs[key] = m

    best = list(best_by_obs.values())
    best.sort(key=lambda d: (d["score"], d["obs_int"]), reverse=True)
    if cfg.MAX_PLOTTED_FRAGMENTS is not None:
        best = best[: int(cfg.MAX_PLOTTED_FRAGMENTS)]

    if emit_outputs:
        print(f"Matched fragments: {len(best)} (from {len(matches)} raw matches)")
        for m in best:
            print(
                f"{m['label']}\tI={m['obs_int']:.3g}\t"
                f"anchor={m['anchor_theory_mz']:.4f}->{m['obs_mz']:.4f}\t"
                f"cos0={m['neutral_score']:.3f}\trawcos={m['raw_score']:.3f}"
            )

        if bool(cfg.EXPORT_FRAGMENTS_CSV):
            out_dir = Path(__file__).parent / "match_outputs"
            file_tag = sanitize_filename(Path(str(cfg.filepath)).stem)
            mz_tag = f"{'' if cfg.MZ_MIN is None else int(cfg.MZ_MIN)}-{'' if cfg.MZ_MAX is None else int(cfg.MZ_MAX)}"
            base = sanitize_filename(f"fragments_scan{int(cfg.SCAN)}_{file_tag}_mz{mz_tag}")

            summary_path = Path(cfg.FRAGMENTS_CSV_SUMMARY_PATH) if cfg.FRAGMENTS_CSV_SUMMARY_PATH else (
                out_dir / f"{base}_summary.csv"
            )
            peaks_path = Path(cfg.FRAGMENTS_CSV_PEAKS_PATH) if cfg.FRAGMENTS_CSV_PEAKS_PATH else (
                out_dir / f"{base}_peaks.csv"
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
                        "css": m.get("score", ""),
                        "rawcos": m.get("raw_score", ""),
                        "cos0": m.get("neutral_score", ""),
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
                                "css": m.get("score", ""),
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
                    "css",
                    "rawcos",
                    "cos0",
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
            noise_cutoff=(obs_max * float(cfg.MIN_OBS_REL_INT)) if float(cfg.MIN_OBS_REL_INT) > 0 else None,
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


def run_diagnose_headless(residues, spectrum, isodec_config, ion_spec: str = None, h_transfer: int = 0) -> dict:
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
        tol_ppm = float(cfg.MATCH_TOL_PPM) if cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM is None else float(
            cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM
        )
        mz_min_cap = None if cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN is None else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN)
        mz_max_cap = None if cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX is None else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX)
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
    charges = [int(charge)] if charge is not None else list(range(int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1))

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
            loss = neutral_loss_label(row_loss_count, row_loss_formula) if row_loss_formula and row_loss_count else ""
            label = f"{frag_name}{loss}^{z}+"

            dist = row.get("dist_plot")
            theory_mz = dist[:, 0].tolist() if isinstance(dist, np.ndarray) and dist.size else []
            theory_int = dist[:, 1].tolist() if isinstance(dist, np.ndarray) and dist.size else []

            # Use anchor_theory_mz if available, otherwise use expected_theory_mz
            theory_anchor_mz = row.get("anchor_theory_mz") or row.get("expected_theory_mz")

            formatted_results.append({
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
                "isodec_css": row.get("isodec_css"),
                "isodec_accepted": bool(row.get("isodec_accepted", False)),
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
            })

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
        "theory_mz": best_formatted.get("theory_mz", []) if isinstance(best_formatted, dict) else [],
        "theory_int": best_formatted.get("theory_int", []) if isinstance(best_formatted, dict) else [],
    }


def run_diagnose_mode(residues, spectrum, isodec_config) -> None:
    if not cfg.DIAGNOSE_ION_SPEC:
        raise ValueError('Set DIAGNOSE_ION_SPEC (e.g., "c7^2+" or "z12-2H2O^3+") when using PLOT_MODE="diagnose".')

    raw_spectrum = np.array(spectrum, dtype=float, copy=True)
    if bool(cfg.ENABLE_FRAGMENT_INTENSITY_CAP):
        tol_ppm = float(cfg.MATCH_TOL_PPM) if cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM is None else float(
            cfg.FRAGMENT_INTENSITY_CAP_TOL_PPM
        )
        mz_min_cap = None if cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN is None else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MIN)
        mz_max_cap = None if cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX is None else float(cfg.FRAGMENT_INTENSITY_CAP_MZ_MAX)
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
                print(f"Fragment intensity-cap strip: cap={cap:.3g} ({hits} windows), removed {removed} peaks")
        else:
            if bool(cfg.FRAGMENT_INTENSITY_CAP_VERBOSE):
                print(f"Fragment intensity-cap strip: skipped (cap={cap:.3g}, hits={hits} < {min_hits})")

    ion_type, frag_len, loss_formula, loss_count, charge = parse_fragment_spec(cfg.DIAGNOSE_ION_SPEC)
    charges = [int(charge)] if charge is not None else list(range(int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1))
    if charge is None and not bool(cfg.DIAGNOSE_SCAN_CHARGES):
        raise ValueError("Ion spec has no charge; set DIAGNOSE_SCAN_CHARGES=True or include ^z+ (e.g., c7^2+).")

    try:
        h_transfer = int(cfg.DIAGNOSE_H_TRANSFER)
    except Exception as e:
        raise ValueError("DIAGNOSE_H_TRANSFER must be an integer in {-2,-1,0,1,2}.") from e
    if h_transfer not in (-2, -1, 0, 1, 2):
        raise ValueError("DIAGNOSE_H_TRANSFER must be an integer in {-2,-1,0,1,2}.")

    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]

    print("=== Ion diagnostic ===")
    print(f"Spec: {cfg.DIAGNOSE_ION_SPEC}")
    print(f"Parsed: ion_type={ion_type}, frag_len={frag_len}, loss={loss_formula or 'none'} x{loss_count or 0}")
    print(f"H transfer: {h_transfer:+d} H+")
    print(f"Charge(s): {charges}")

    diagnose_dir = Path(__file__).parent / "diagnose_outputs"
    spec_safe = sanitize_filename(cfg.DIAGNOSE_ION_SPEC)
    base = f"diagnose_scan{int(cfg.SCAN)}_{spec_safe}_h{h_transfer:+d}"
    base = sanitize_filename(base)
    summary_path = Path(cfg.DIAGNOSE_CSV_SUMMARY_PATH) if cfg.DIAGNOSE_CSV_SUMMARY_PATH else (
        diagnose_dir / f"{base}_summary.csv"
    )
    peaks_path = Path(cfg.DIAGNOSE_CSV_PEAKS_PATH) if cfg.DIAGNOSE_CSV_PEAKS_PATH else (
        diagnose_dir / f"{base}_peaks.csv"
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
            status_icon = "PASS" if step["status"] == "pass" else "FAIL" if step["status"] == "fail" else "INFO"
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

            detail = r.get("isodec_detail") if isinstance(r.get("isodec_detail"), dict) else {}
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

            for p in r.get("theory_matches", []) if isinstance(r.get("theory_matches"), list) else []:
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
        css_txt = f"{css:.3f}" if isinstance(css, (int, float)) and np.isfinite(css) else "n/a"
        mono_txt = f"{mono_mass:.6f}" if isinstance(mono_mass, (int, float)) and np.isfinite(mono_mass) else str(mono_mass)
        avg_txt = f"{avg_mass:.6f}" if isinstance(avg_mass, (int, float)) and np.isfinite(avg_mass) else str(avg_mass)
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
                f"relI={100.0*r.get('obs_rel_int', 0.0):.2f}%"
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
            matches_sorted = sorted(matches, key=lambda x: (x["within"], x["obs_int"]), reverse=True)
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
                noise_cutoff=(float(np.max(spectrum[:, 1])) * float(cfg.MIN_OBS_REL_INT))
                if float(cfg.MIN_OBS_REL_INT) > 0
                else None,
            )
        else:
            try:
                fallback_charge = best.get("z") if isinstance(best, dict) and best.get("z") else (charges[0] if charges else 1)
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
                        dist_try = theoretical_isodist_from_comp(comp_try, int(fallback_charge))
                    except Exception:
                        continue
                    if dist_try.size:
                        fallback_suffix = suffix
                        fallback_comp = comp_try
                        dist_fallback = dist_try
                        break
                if dist_fallback is None:
                    dist_fallback = theoretical_isodist_from_comp(fallback_comp, int(fallback_charge))
                if int(h_transfer) != 0 and dist_fallback.size:
                    dist_fallback = dist_fallback.copy()
                    dist_fallback[:, 0] += (float(h_transfer) * float(cfg.H_TRANSFER_MASS)) / float(fallback_charge)
                if dist_fallback.size:
                    dist_plot = dist_fallback.copy()
                    if raw_spectrum.size:
                        max_obs = float(np.max(raw_spectrum[:, 1]))
                        max_theory = float(np.max(dist_plot[:, 1])) if dist_plot.size else 0.0
                        if max_theory > 0:
                            dist_plot[:, 1] *= max_obs / max_theory
                    if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
                        mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
                        mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
                        dist_plot = dist_plot[(dist_plot[:, 0] >= mz_min) & (dist_plot[:, 0] <= mz_max)]
                    loss = neutral_loss_label(int(loss_count), loss_formula) if loss_formula and loss_count else ""
                    label = f"{frag_name}{fallback_suffix}{loss}^{int(fallback_charge)}+"
                    plot_overlay(
                        raw_spectrum,
                        [(dist_plot, "tab:purple", f"diagnose fallback {label}")],
                        mz_min=None if cfg.MZ_MIN is None else float(cfg.MZ_MIN),
                        mz_max=None if cfg.MZ_MAX is None else float(cfg.MZ_MAX),
                    )
            except Exception as e:
                print(f"Fallback plot skipped: {e}")
