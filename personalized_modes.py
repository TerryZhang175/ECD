from __future__ import annotations

from pathlib import Path

import numpy as np
import personalized_config as cfg
from personalized_match import (
    compute_fragment_intensity_cap,
    diagnose_candidate,
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


def run_precursor_headless(residues, spectrum, isodec_config) -> dict:
    # Override isodec_config.matchtol with PRECURSOR_MATCH_TOL_PPM for precursor mode
    precursor_tol_ppm = float(getattr(cfg, "PRECURSOR_MATCH_TOL_PPM", cfg.MATCH_TOL_PPM))
    if isodec_config is not None and hasattr(isodec_config, "matchtol"):
        isodec_config.matchtol = precursor_tol_ppm

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
        anchor_idx = int(np.argmax(dist[:, 1]))
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
            "best_z": None,
            "best_css": 0.0,
            "shift_ppm": 0.0,
            "best_obs_mz": None,
            "best_theory_mz": None,
            "best_theory_dist": None,
            "plot_window": None,
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

    work_spectrum = np.array(spectrum, dtype=float, copy=True)
    match_found = False
    best_z = None
    best_state = None
    best_css = 0.0
    shift_ppm = 0.0
    best_obs_mz = None
    best_theory_mz = None
    best_theory_dist = None
    first_anchor_int = None
    best_by_charge: dict[int, dict] = {}
    window_da = float(getattr(cfg, "PRECURSOR_WINDOW_DA", 5.1))
    if not np.isfinite(window_da) or window_da <= 0:
        window_da = 5.1

    print(
        f"--- Starting Precursor Search (Max iterations: {int(cfg.PRECURSOR_SEARCH_ITERATIONS)}, "
        f"window={window_da:.2f} Da) ---"
    )

    for iteration in range(1, int(cfg.PRECURSOR_SEARCH_ITERATIONS) + 1):
        if work_spectrum.size == 0:
            break

        window = _find_most_intense_window(work_spectrum[:, 0], work_spectrum[:, 1], window_da)
        if window is None:
            break
        start_idx = int(window["start_idx"])
        end_idx = int(window["end_idx"])
        window_min = float(window["mz_min"])
        window_max = float(window["mz_max"])
        if end_idx <= start_idx:
            keep_mask = (work_spectrum[:, 0] < window_min) | (work_spectrum[:, 0] > window_max)
            work_spectrum = work_spectrum[keep_mask]
            print(
                f"Iteration {iteration}: empty window [{window_min:.4f}, {window_max:.4f}] removed; retrying."
            )
            continue

        windowed = work_spectrum[start_idx:end_idx]
        raw_obs_mz = float(windowed[np.argmax(windowed[:, 1]), 0])

        local_centroids = get_local_centroids_window(
            work_spectrum[:, 0],
            work_spectrum[:, 1],
            raw_obs_mz,
            lb=window_min - raw_obs_mz,
            ub=window_max - raw_obs_mz,
        )
        if isinstance(local_centroids, np.ndarray) and local_centroids.size:
            centroid_idx = int(np.argmax(local_centroids[:, 1]))
            obs_mz = float(local_centroids[centroid_idx, 0])
            obs_anchor_int = float(local_centroids[centroid_idx, 1])
        else:
            obs_mz = raw_obs_mz
            local_centroids = windowed
            obs_anchor_int = float(windowed[np.argmax(windowed[:, 1]), 1]) if windowed.size else 0.0
        if first_anchor_int is None:
            first_anchor_int = obs_anchor_int

        candidate_states = []
        for z, theory in precursor_theories.items():
            mz_pad = precursor_tol_ppm * 1e-6 * float(obs_mz) if obs_mz > 0 else 0.0
            base_min = float(theory["mz_min"])
            base_max = float(theory["mz_max"])
            for state, h_shift in state_shifts:
                shift_mz = (float(h_shift) * float(cfg.H_TRANSFER_MASS)) / float(z)
                mz_min = base_min + shift_mz - mz_pad
                mz_max = base_max + shift_mz + mz_pad
                if mz_min <= obs_mz <= mz_max:
                    candidate_states.append((int(z), str(state), int(h_shift), float(shift_mz)))

        best_candidate = None
        for z, state, h_shift, shift_mz in candidate_states:
            dist = precursor_theories[z]["dist"]
            if h_shift:
                dist_shifted = dist.copy()
                dist_shifted[:, 0] += shift_mz
            else:
                dist_shifted = dist
            theory_mz = float(precursor_theories[z]["anchor_mz"]) + float(shift_mz)
            obs_mz_eval = float(obs_mz)
            if cfg.ENABLE_ISODEC_RULES and isodec_config is not None:
                accepted, isodec_css, shifted_peak = isodec_css_and_accept(
                    local_centroids, dist_shifted, z=z, peakmz=obs_mz, config=isodec_config
                )
                if shifted_peak is not None:
                    obs_mz_eval = float(shifted_peak)
            else:
                y_obs = observed_intensities_isodec(
                    local_centroids[:, 0],
                    local_centroids[:, 1],
                    dist_shifted[:, 0],
                    z=int(z),
                    match_tol_ppm=precursor_tol_ppm,
                    peak_mz=obs_mz,
                )
                isodec_css = css_similarity(y_obs, dist_shifted[:, 1])
                accepted = isodec_css >= float(cfg.MIN_COSINE)

            ppm = ((float(obs_mz_eval) - theory_mz) / theory_mz) * 1e6 if theory_mz else 0.0
            dist_plot = _scale_dist_to_obs(dist_shifted, obs_anchor_int)
            candidate = {
                "charge": int(z),
                "state": str(state),
                "obs_mz": float(obs_mz_eval),
                "anchor_theory_mz": theory_mz,
                "ppm": float(ppm),
                "css": float(isodec_css),
                "accepted": bool(accepted),
                "iteration": int(iteration),
                "dist": dist_plot,
            }
            prev = best_by_charge.get(int(z))
            if prev is None or float(candidate["css"]) > float(prev.get("css", -1.0)):
                best_by_charge[int(z)] = candidate

            if accepted and isodec_css >= float(cfg.MIN_COSINE):
                if best_candidate is None or float(candidate["css"]) > float(best_candidate.get("css", -1.0)):
                    best_candidate = candidate

        if best_candidate is not None:
            match_found = True
            best_z = int(best_candidate["charge"])
            best_state = best_candidate.get("state")
            best_css = float(best_candidate["css"])
            best_obs_mz = float(best_candidate["obs_mz"])
            best_theory_mz = float(best_candidate["anchor_theory_mz"])
            best_theory_dist = best_candidate["dist"]
            shift_ppm = ((best_obs_mz - best_theory_mz) / best_theory_mz) * 1e6 if best_theory_mz else 0.0
            state_label = f" ({best_state})" if best_state and best_state != "0" else ""
            print(
                f"Precursor found in iteration {iteration}: z={best_z}+{state_label} "
                f"m/z={best_obs_mz:.4f} css={best_css:.3f}"
            )
            break

        keep_mask = (work_spectrum[:, 0] < window_min) | (work_spectrum[:, 0] > window_max)
        work_spectrum = work_spectrum[keep_mask]
        print(
            f"Iteration {iteration}: window [{window_min:.4f}, {window_max:.4f}] failed CSS; blanking and retrying."
        )

    calibrated_spectrum = np.array(spectrum, dtype=float, copy=True)
    if match_found and bool(cfg.ENABLE_LOCK_MASS):
        print(f"Applying lock-mass calibration: shift {-shift_ppm:.2f} ppm")
        calibrated_spectrum[:, 0] = calibrated_spectrum[:, 0] / (1.0 + (shift_ppm / 1e6))
    elif not match_found:
        print("Precursor not found; no calibration applied.")

    if not match_found and all_theory_dist is not None:
        if first_anchor_int is not None:
            all_theory_dist = _scale_dist_to_obs(all_theory_dist, float(first_anchor_int))
        best_theory_dist = all_theory_dist

    candidates = sorted(best_by_charge.values(), key=lambda d: float(d.get("css", -1.0)), reverse=True)
    plot_window = None
    if match_found and best_theory_dist is not None and isinstance(best_theory_dist, np.ndarray) and best_theory_dist.size:
        plot_window = (float(best_theory_dist[0, 0]) - 5.0, float(best_theory_dist[-1, 0]) + 5.0)

    return {
        "spectrum": calibrated_spectrum,
        "spectrum_mz": np.asarray(calibrated_spectrum[:, 0], dtype=float),
        "spectrum_int": np.asarray(calibrated_spectrum[:, 1], dtype=float),
        "match_found": bool(match_found),
        "best_z": best_z,
        "best_state": best_state,
        "best_css": float(best_css),
        "shift_ppm": float(shift_ppm),
        "best_obs_mz": best_obs_mz,
        "best_theory_mz": best_theory_mz,
        "best_theory_dist": best_theory_dist,
        "plot_window": plot_window,
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


def run_charge_reduced_headless(residues, spectrum, isodec_config) -> dict:
    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0

    targets: dict[str, tuple[object, str]] = {}
    monomer_comp = get_neutral_monomer_composition(residues)
    targets["Monomer"] = (monomer_comp, "tab:blue")
    if int(cfg.COPIES) > 1:
        complex_comp = get_precursor_composition(residues)
        targets[f"Complex ({int(cfg.COPIES)}x)"] = (complex_comp, "tab:red")

    matches: list[dict] = []
    print("--- Starting Charge-Reduced Precursor Search (data-driven anchor) ---")

    state_shifts = [
        ("0", 0),
        ("+H", 1),
        ("+2H", 2),
        ("-H", -1),
        ("-2H", -2),
    ]

    for label, (target_comp, color) in targets.items():
        for z in range(int(cfg.CR_MIN_CHARGE), int(cfg.CR_MAX_CHARGE) + 1):
            try:
                dist0 = theoretical_isodist_from_comp(target_comp, z)
            except ValueError:
                continue
            if dist0.size == 0:
                continue

            theory_base_mz = float(dist0[np.argmax(dist0[:, 1]), 0])
            window_mask = (spectrum_mz >= theory_base_mz - 2.0) & (spectrum_mz <= theory_base_mz + 2.0)
            if not np.any(window_mask):
                continue
            window_idx = np.where(window_mask)[0]
            obs_anchor_mz = float(spectrum_mz[window_idx[np.argmax(spectrum_int[window_idx])]])

            local_centroids = get_local_centroids_window(
                spectrum_mz, spectrum_int, center_mz=obs_anchor_mz, lb=-3.0, ub=3.0
            )
            if not isinstance(local_centroids, np.ndarray) or local_centroids.size == 0:
                continue
            cent_mz = local_centroids[:, 0]
            cent_int = local_centroids[:, 1]

            best_candidate = None
            for state, h_shift in state_shifts:
                dist = dist0.copy()
                dist[:, 0] += (float(h_shift) * float(cfg.H_TRANSFER_MASS)) / float(z)

                if cfg.ENABLE_ISODEC_RULES and isodec_config is not None:
                    accepted, css, shifted_peak = isodec_css_and_accept(
                        local_centroids, dist, z=z, peakmz=obs_anchor_mz, config=isodec_config
                    )
                else:
                    y_obs = observed_intensities_isodec(
                        cent_mz,
                        cent_int,
                        dist[:, 0],
                        z=int(z),
                        match_tol_ppm=float(cfg.MATCH_TOL_PPM),
                        peak_mz=obs_anchor_mz,
                    )
                    css = css_similarity(y_obs, dist[:, 1])
                    accepted = css >= float(cfg.MIN_COSINE)
                    shifted_peak = None

                if not accepted or css < float(cfg.MIN_COSINE):
                    continue

                obs_mz = float(shifted_peak) if shifted_peak is not None else float(obs_anchor_mz)
                obs_idx = nearest_peak_index(cent_mz, obs_mz)
                obs_int = float(cent_int[obs_idx]) if len(cent_int) else 0.0

                dist_full = dist.copy()
                dist_plot = dist.copy()
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
                    tol_ppm=float(cfg.MATCH_TOL_PPM),
                    theory_int=dist_full[:, 1],
                )

                candidate = {
                    "label": f"{label}^{z}+ ({state}) | css={float(css):.3f}",
                    "target": label,
                    "z": int(z),
                    "state": state,
                    "css": float(css),
                    "obs_mz": obs_mz,
                    "obs_int": obs_int,
                    "dist_full": dist_full,
                    "dist": dist_plot,
                    "theory_matches": theory_matches,
                    "color": color,
                }

                if best_candidate is None or candidate["css"] > best_candidate["css"]:
                    best_candidate = candidate

            if best_candidate is not None:
                matches.append(best_candidate)

    matches.sort(key=lambda x: (x["css"], x["obs_int"]), reverse=True)
    print(f"Identified {len(matches)} charge-reduced forms:")
    for m in matches:
        print(f"  {m['label']} at m/z {m['obs_mz']:.3f} (I={m['obs_int']:.1f})")

    return {
        "spectrum": np.array(spectrum, dtype=float, copy=True),
        "spectrum_mz": np.asarray(spectrum_mz, dtype=float),
        "spectrum_int": np.asarray(spectrum_int, dtype=float),
        "obs_max": obs_max,
        "matches": matches,
    }


def run_charge_reduced_mode(residues, spectrum, isodec_config) -> None:
    spectrum_mz = spectrum[:, 0]
    spectrum_int = spectrum[:, 1]
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0

    targets: dict[str, tuple[object, str]] = {}
    monomer_comp = get_neutral_monomer_composition(residues)
    targets["Monomer"] = (monomer_comp, "tab:blue")
    if int(cfg.COPIES) > 1:
        complex_comp = get_precursor_composition(residues)
        targets[f"Complex ({int(cfg.COPIES)}x)"] = (complex_comp, "tab:red")

    matches: list[dict] = []
    print("--- Starting Charge-Reduced Precursor Search (data-driven anchor) ---")

    state_shifts = [
        ("0", 0),
        ("+H", 1),
        ("+2H", 2),
        ("-H", -1),
        ("-2H", -2),
    ]

    for label, (target_comp, color) in targets.items():
        for z in range(int(cfg.CR_MIN_CHARGE), int(cfg.CR_MAX_CHARGE) + 1):
            try:
                dist0 = theoretical_isodist_from_comp(target_comp, z)
            except ValueError:
                continue
            if dist0.size == 0:
                continue

            theory_base_mz = float(dist0[np.argmax(dist0[:, 1]), 0])
            window_mask = (spectrum_mz >= theory_base_mz - 2.0) & (spectrum_mz <= theory_base_mz + 2.0)
            if not np.any(window_mask):
                continue
            window_idx = np.where(window_mask)[0]
            obs_anchor_mz = float(spectrum_mz[window_idx[np.argmax(spectrum_int[window_idx])]])

            local_centroids = get_local_centroids_window(
                spectrum_mz, spectrum_int, center_mz=obs_anchor_mz, lb=-3.0, ub=3.0
            )
            if not isinstance(local_centroids, np.ndarray) or local_centroids.size == 0:
                continue
            cent_mz = local_centroids[:, 0]
            cent_int = local_centroids[:, 1]

            best_candidate = None
            for state, h_shift in state_shifts:
                dist = dist0.copy()
                dist[:, 0] += (float(h_shift) * float(cfg.H_TRANSFER_MASS)) / float(z)

                if cfg.ENABLE_ISODEC_RULES and isodec_config is not None:
                    accepted, css, shifted_peak = isodec_css_and_accept(
                        local_centroids, dist, z=z, peakmz=obs_anchor_mz, config=isodec_config
                    )
                else:
                    y_obs = observed_intensities_isodec(
                        cent_mz,
                        cent_int,
                        dist[:, 0],
                        z=int(z),
                        match_tol_ppm=float(cfg.MATCH_TOL_PPM),
                        peak_mz=obs_anchor_mz,
                    )
                    css = css_similarity(y_obs, dist[:, 1])
                    accepted = css >= float(cfg.MIN_COSINE)
                    shifted_peak = None

                if not accepted or css < float(cfg.MIN_COSINE):
                    continue

                obs_mz = float(shifted_peak) if shifted_peak is not None else float(obs_anchor_mz)
                obs_idx = nearest_peak_index(cent_mz, obs_mz)
                obs_int = float(cent_int[obs_idx]) if len(cent_int) else 0.0

                dist_full = dist.copy()
                dist_plot = dist.copy()
                max_plot = float(np.max(dist_plot[:, 1])) if dist_plot.size else 0.0
                if max_plot > 0 and obs_int > 0:
                    dist_plot[:, 1] *= obs_int / max_plot
                keep = dist_plot[:, 1] >= max_plot * float(cfg.REL_INTENSITY_CUTOFF) if max_plot > 0 else dist_plot[:, 1] > 0
                dist_plot = dist_plot[keep] if dist_plot.size else dist_plot

                theory_matches = match_theory_peaks(
                    cent_mz,
                    cent_int,
                    dist_full[:, 0],
                    tol_ppm=float(cfg.MATCH_TOL_PPM),
                    theory_int=dist_full[:, 1],
                )

                candidate = {
                    "label": f"{label}^{z}+ ({state}) | css={float(css):.3f}",
                    "target": label,
                    "z": int(z),
                    "state": state,
                    "css": float(css),
                    "obs_mz": obs_mz,
                    "obs_int": obs_int,
                    "dist_full": dist_full,
                    "dist": dist_plot,
                    "theory_matches": theory_matches,
                    "color": color,
                }

                if best_candidate is None or candidate["css"] > best_candidate["css"]:
                    best_candidate = candidate

            if best_candidate is not None:
                matches.append(best_candidate)

    matches.sort(key=lambda x: (x["css"], x["obs_int"]), reverse=True)
    print(f"Identified {len(matches)} charge-reduced forms:")
    for m in matches:
        print(f"  {m['label']} at m/z {m['obs_mz']:.3f} (I={m['obs_int']:.1f})")

    overlays = [(m["dist"], m["color"], m["label"]) for m in matches]
    plot_overlay(
        spectrum,
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
        if matches:
            for m in matches:
                summary_rows.append(
                    {
                        "label": m.get("label", ""),
                        "target": m.get("target", ""),
                        "z": m.get("z", ""),
                        "state": m.get("state", ""),
                        "css": m.get("css", ""),
                        "obs_mz": m.get("obs_mz", ""),
                        "obs_int": m.get("obs_int", ""),
                    }
                )

                for p in m.get("theory_matches", []) if isinstance(m.get("theory_matches"), list) else []:
                    peaks_rows.append(
                        {
                            "label": m.get("label", ""),
                            "target": m.get("target", ""),
                            "z": m.get("z", ""),
                            "state": m.get("state", ""),
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
            for label, (target_comp, _) in targets.items():
                for z in range(int(cfg.CR_MIN_CHARGE), int(cfg.CR_MAX_CHARGE) + 1):
                    try:
                        dist0 = theoretical_isodist_from_comp(target_comp, z)
                    except ValueError:
                        continue
                    if dist0.size == 0:
                        continue

                    for state, h_shift in state_shifts:
                        dist = dist0.copy()
                        dist[:, 0] += (float(h_shift) * float(cfg.H_TRANSFER_MASS)) / float(z)
                        summary_rows.append(
                            {
                                "label": f"{label}^{z}+",
                                "target": label,
                                "z": int(z),
                                "state": state,
                                "css": "",
                                "obs_mz": "",
                                "obs_int": "",
                            }
                        )
                        for mz_val, int_val in dist:
                            peaks_rows.append(
                                {
                                    "label": f"{label}^{z}+",
                                    "target": label,
                                    "z": int(z),
                                    "state": state,
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
            ["label", "target", "z", "state", "css", "obs_mz", "obs_int"],
            summary_rows,
        )
        write_csv(
            peaks_path,
            [
                "label",
                "target",
                "z",
                "state",
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

                peak_mz = float(dist0[np.argmax(dist0[:, 1]), 0])
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
