from __future__ import annotations

from pathlib import Path

import numpy as np
import pyteomics.mass as ms

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
    ion_composition_from_sequence,
    ion_series,
    neutral_loss_columns,
    neutral_loss_label,
    neutral_loss_variants,
    residue_range_composition,
)
from personalized_theory import (
    build_sample_axis,
    css_similarity,
    fit_simplex_mixture,
    observed_intensities_isodec,
    theoretical_isodist_from_comp,
    vectorize_dist,
)


def run_precursor_mode(residues, spectrum, isodec_config) -> None:
    base_comp = residue_range_composition(residues, 0, len(residues)) + ms.Composition("H2O")
    if cfg.AMIDATED:
        base_comp += ms.Composition(cfg.AMIDATION_FORMULA)

    comp = base_comp * int(cfg.COPIES)
    comp += ms.Composition("H-2") * int(cfg.DISULFIDE_BONDS)

    theory_plot = theoretical_isodist_from_comp(comp, cfg.CHARGE)
    if theory_plot.size == 0:
        raise ValueError("Theoretical distribution is empty after filtering; lower REL_INTENSITY_CUTOFF.")

    peak_theory_mz = float(theory_plot[np.argmax(theory_plot[:, 1]), 0])
    peak_obs_mz = None
    if cfg.ALIGN_TO_DATA:
        spectrum_mz = spectrum[:, 0]
        spectrum_int = spectrum[:, 1]
        in_win = (spectrum_mz >= peak_theory_mz - cfg.ALIGN_WINDOW_MZ) & (
            spectrum_mz <= peak_theory_mz + cfg.ALIGN_WINDOW_MZ
        )
        if np.any(in_win):
            window = spectrum[in_win]
            best_score = -1.0
            best_shift = 0.0
            best_peak = None
            for obs_mz in window[:, 0]:
                shift = float(obs_mz) - peak_theory_mz
                shifted_mz = theory_plot[:, 0] + shift
                y_obs = observed_intensities_isodec(
                    spectrum_mz,
                    spectrum_int,
                    shifted_mz,
                    z=int(cfg.CHARGE),
                    match_tol_ppm=float(cfg.MATCH_TOL_PPM),
                    peak_mz=float(obs_mz),
                )
                score = css_similarity(y_obs, theory_plot[:, 1])
                if score > best_score:
                    best_score = score
                    best_shift = shift
                    best_peak = float(obs_mz)
            if best_peak is not None:
                peak_obs_mz = best_peak
                theory_plot[:, 0] += best_shift

    if peak_obs_mz is not None:
        idx = int(np.argmin(np.abs(spectrum[:, 0] - peak_obs_mz)))
        scale = float(spectrum[idx, 1]) / float(np.max(theory_plot[:, 1]))
    else:
        scale = float(np.max(spectrum[:, 1])) / float(np.max(theory_plot[:, 1]))
    theory_plot[:, 1] *= scale

    if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
        mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
        mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
        theory_plot = theory_plot[(theory_plot[:, 0] >= mz_min) & (theory_plot[:, 0] <= mz_max)]
        if theory_plot.size == 0:
            raise ValueError("Theoretical distribution is empty inside the selected m/z window.")

    plot_overlay(
        spectrum,
        [(theory_plot, "tab:red", f"Precursor (z={cfg.CHARGE})")],
        mz_min=None if cfg.MZ_MIN is None else float(cfg.MZ_MIN),
        mz_max=None if cfg.MZ_MAX is None else float(cfg.MZ_MAX),
    )


def run_fragments_mode(residues, spectrum, isodec_config) -> None:
    if cfg.FRAG_MIN_CHARGE <= 0 or cfg.FRAG_MAX_CHARGE <= 0 or cfg.FRAG_MIN_CHARGE > cfg.FRAG_MAX_CHARGE:
        raise ValueError("Set FRAG_MIN_CHARGE/FRAG_MAX_CHARGE to a valid positive range.")

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
    for ion_type in cfg.ION_TYPES:
        series = ion_series(ion_type)
        allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_1H))
        allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_2H))
        for frag_len in range(1, n):
            frag_name, frag_comp = ion_composition_from_sequence(residues, ion_type, frag_len, amidated=cfg.AMIDATED)
            def evaluate_candidate(loss_suffix: str, loss_comp, z: int):
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

                neutral_score = css_similarity(y_obs, y0)
                best_model = "neutral"
                best_score = neutral_score
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

                    rel_improve = (best_score - neutral_score) / max(neutral_score, 1e-12)
                    if best_model != "neutral" and rel_improve < float(cfg.H_TRANSFER_MIN_REL_IMPROVEMENT):
                        best_model = "neutral"
                        best_score = neutral_score
                        best_pred = y0
                        best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}

                if float(np.max(best_pred)) <= 0.0:
                    return None

                anchor_theory_mz = None
                obs_idx = None
                obs_mz = None
                obs_int = None
                anchor_hits = 0
                sorted_idx = np.argsort(best_pred)[::-1][: int(cfg.ANCHOR_TOP_N)]
                for idx in sorted_idx:
                    mz_candidate = float(sample_mzs[int(idx)])
                    obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
                    obs_mz_c = float(spectrum_mz[obs_idx_c])
                    if not within_ppm(obs_mz_c, mz_candidate, float(cfg.MATCH_TOL_PPM)):
                        continue
                    obs_int_c = float(spectrum_int[obs_idx_c])
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

                frag_id = f"{frag_name}{loss_suffix}"
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

            for z in range(int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1):
                neutral_match = evaluate_candidate("", frag_comp, z)
                
                # 判据 A: 如果根本没找到匹配 (evaluate_candidate 返回 None)，直接跳过后续 Loss
                if neutral_match is None:
                    continue

                # 判据 B: 只有当中性匹配通过最终输出筛选时才考虑中性丢失变体
                # 这使用与最终片段选择相同的 IsoDec CSS 阈值
                if neutral_match["score"] < float(cfg.ISODEC_CSS_THRESH):
                    # 如果中性分数低于最终输出阈值，保留它但不尝试匹配 Loss
                    matches.append(neutral_match)
                    continue  # 跳过下方的 Loss 循环

                matches.append(neutral_match)
                
                # 只有当中性匹配通过最终输出筛选时才进入此循环
                for loss_suffix, loss_comp in neutral_loss_variants(frag_comp, ion_series_letter=series):
                    if not loss_suffix:
                        continue
                    loss_match = evaluate_candidate(loss_suffix, loss_comp, z)
                    if loss_match is not None:
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

    print(f"Matched fragments: {len(best)} (from {len(matches)} raw matches)")
    for m in best:
        print(f"{m['label']}\tI={m['obs_int']:.3g}\tcos0={m['neutral_score']:.3f}\trawcos={m['raw_score']:.3f}")

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


def run_diagnose_mode(residues, spectrum, isodec_config) -> None:
    if not cfg.DIAGNOSE_ION_SPEC:
        raise ValueError('Set DIAGNOSE_ION_SPEC (e.g., "c7^2+" or "z12-2H2O^3+") when using PLOT_MODE="diagnose".')

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
            status_icon = "✅" if step["status"] == "pass" else "❌" if step["status"] == "fail" else "ℹ️"
            print(f"{status_icon} {step['step']}: {step['status']} - {step['details']}")
        
        if r.get("ok", False):
            print(f"✅ Overall: PASS")
        else:
            print(f"❌ Overall: FAIL - {r.get('reason', 'Unknown reason')}")
        
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
        print(
            f"- {label}\tok={r['ok']}\treason={r['reason']}\tformula={formula}\t"
            f"mono_mass={mono_mass:.6f}\tavg_mass={avg_mass:.6f}\t"
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
    if bool(cfg.DIAGNOSE_SHOW_PLOT) and isinstance(best, dict) and isinstance(best.get("dist_plot"), np.ndarray):
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