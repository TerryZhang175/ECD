from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

import numpy as np

import personalized_config as cfg
from personalized_sequence import (
    apply_neutral_loss,
    ion_composition_from_sequence,
    ion_series,
    neutral_loss_columns,
    neutral_loss_label,
    neutral_loss_variants,
    get_disulfide_logic,
)
from personalized_theory import css_similarity, theoretical_isodist_from_comp, observed_intensities_isodec


def nearest_peak_index(sorted_mzs: np.ndarray, target_mz: float) -> int:
    idx = int(np.searchsorted(sorted_mzs, target_mz))
    if idx <= 0:
        return 0
    if idx >= len(sorted_mzs):
        return len(sorted_mzs) - 1
    left = idx - 1
    right = idx
    if abs(sorted_mzs[left] - target_mz) <= abs(sorted_mzs[right] - target_mz):
        return left
    return right


def within_ppm(mz_obs: float, mz_theory: float, tol_ppm: float) -> bool:
    if mz_theory == 0:
        return False
    return abs(mz_obs - mz_theory) / abs(mz_theory) * 1e6 <= tol_ppm


def sanitize_filename(text: str) -> str:
    s = str(text).strip()
    if not s:
        return "output"
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    return re.sub(r"_+", "_", "".join(out)).strip("_") or "output"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def max_intensity_in_ppm_window(
    spectrum_mz: np.ndarray, spectrum_int: np.ndarray, target_mz: float, tol_ppm: float
) -> float:
    if target_mz <= 0 or tol_ppm <= 0 or spectrum_mz.size == 0:
        return 0.0
    tol = float(tol_ppm) * 1e-6
    lo = float(target_mz) * (1.0 - tol)
    hi = float(target_mz) * (1.0 + tol)
    i0 = int(np.searchsorted(spectrum_mz, lo, side="left"))
    i1 = int(np.searchsorted(spectrum_mz, hi, side="right"))
    if i1 <= i0:
        return 0.0
    return float(np.max(spectrum_int[i0:i1]))


def compute_fragment_intensity_cap(
    residues: list[tuple[str, list[str]]],
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    tol_ppm: float,
    mz_min: Optional[float] = None,
    mz_max: Optional[float] = None,
) -> tuple[float, int]:
    cap = 0.0
    hits = 0
    n = len(residues)
    mz_min_f = None if mz_min is None else float(mz_min)
    mz_max_f = None if mz_max is None else float(mz_max)

    for ion_type in cfg.ION_TYPES:
        series = ion_series(ion_type)
        allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_1H))
        allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_2H))

        for frag_len in range(1, n):
            try:
                _, frag_comp = ion_composition_from_sequence(residues, ion_type, frag_len, amidated=cfg.AMIDATED)
            except Exception:
                continue

            # Get disulfide variants
            cys_variants = get_disulfide_logic(ion_type, frag_len, len(residues))
            
            # If no disulfide variants, use the original composition
            if not cys_variants:
                cys_variants = [("", None)]

            for variant_suffix, shift_comp in cys_variants:
                # Apply disulfide shift if present
                if shift_comp:
                    try:
                        variant_comp = frag_comp + shift_comp
                    except Exception:
                        continue
                else:
                    variant_comp = frag_comp

                for _, loss_comp in neutral_loss_variants(variant_comp, ion_series_letter=series):
                    for z in range(int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1):
                        try:
                            dist0 = theoretical_isodist_from_comp(loss_comp, z)
                        except Exception:
                            continue
                        if dist0.size == 0:
                            continue

                        anchor = float(dist0[np.argmax(dist0[:, 1]), 0])
                        if allow_1h or allow_2h:
                            shift_1 = float(cfg.H_TRANSFER_MASS) / float(z) if (allow_1h or allow_2h) else 0.0
                            shift_2 = 2.0 * float(cfg.H_TRANSFER_MASS) / float(z) if allow_2h else 0.0
                            shifts = [0.0]
                            if allow_1h:
                                shifts.extend([shift_1, -shift_1])
                            if allow_2h:
                                shifts.extend([shift_2, -shift_2])
                        else:
                            shifts = [0.0]

                        for s in shifts:
                            mz0 = anchor + float(s)
                            if mz_min_f is not None and mz0 < mz_min_f:
                                continue
                            if mz_max_f is not None and mz0 > mz_max_f:
                                continue
                            m = max_intensity_in_ppm_window(spectrum_mz, spectrum_int, mz0, tol_ppm=float(tol_ppm))
                            if m > 0.0:
                                hits += 1
                                if m > cap:
                                    cap = m

    return float(cap), int(hits)


def strip_peaks_above_intensity_cap(spectrum: np.ndarray, cap: float) -> np.ndarray:
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.ndim != 2 or spectrum.shape[1] != 2 or len(spectrum) == 0:
        return spectrum
    cap_f = float(cap)
    if not np.isfinite(cap_f) or cap_f <= 0:
        return spectrum
    out = spectrum[spectrum[:, 1] <= cap_f].copy()
    if len(out) > 1:
        out = out[np.argsort(out[:, 0])]
    return out


def match_theory_peaks(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    theory_mz: np.ndarray,
    tol_ppm: float,
    theory_int: Optional[np.ndarray] = None,
) -> list[dict]:
    out: list[dict] = []
    theory_mz = np.asarray(theory_mz, dtype=float)
    
    # 预先找到理论最强峰的索引 (Anchor Peak)
    max_theory_idx = -1
    if theory_int is not None:
        theory_int = np.asarray(theory_int, dtype=float)
        if len(theory_int) > 0:
            max_theory_idx = np.argmax(theory_int)

    for i, mz in enumerate(theory_mz):
        mz_val = float(mz)
        
        # 1. 确定 PPM 窗口的 m/z 范围
        if mz_val == 0:
            delta = 0.0
        else:
            delta = mz_val * float(tol_ppm) * 1e-6
            
        low = mz_val - delta
        high = mz_val + delta
        
        # 2. 在窗口内寻找所有候选峰 (使用 searchsorted 加速范围搜索)
        start_idx = np.searchsorted(spectrum_mz, low, side="left")
        end_idx = np.searchsorted(spectrum_mz, high, side="right")
        
        best_idx = -1
        
        # 如果窗口内没有峰，回退到寻找最近的峰 (即使在 PPM 外)
        # 这样保持了 logic：如果匹配失败，也会返回一个 'within': False 的结果
        if end_idx <= start_idx:
            best_idx = nearest_peak_index(spectrum_mz, mz_val)
        else:
            # 窗口内有候选峰，开始应用你的新策略
            candidate_indices = np.arange(start_idx, end_idx)
            candidate_ints = spectrum_int[start_idx:end_idx]
            
            # 判断当前是否是理论最强峰
            is_anchor = (i == max_theory_idx) and (max_theory_idx != -1)
            
            if is_anchor:
                # 策略 A: 对于理论最强峰，我们假设它是基准，取窗口内强度最大的峰
                # 这避免了因为基准对齐问题导致我们错过真正的 anchor
                local_max_idx = np.argmax(candidate_ints)
                best_idx = start_idx + local_max_idx
            elif theory_int is not None:
                # 策略 B: 对于其他峰，寻找强度最接近 "预期强度" 的峰
                # 预期强度 = theory_int[i] (假设传入前已经做过归一化/缩放)
                target_int = float(theory_int[i])
                diffs = np.abs(candidate_ints - target_int)
                local_min_idx = np.argmin(diffs)
                best_idx = start_idx + local_min_idx
            else:
                # Fallback: 如果没有提供理论强度，则在窗口内找 m/z 最近的
                candidate_mzs = spectrum_mz[start_idx:end_idx]
                mz_diffs = np.abs(candidate_mzs - mz_val)
                local_min_mz_idx = np.argmin(mz_diffs)
                best_idx = start_idx + local_min_mz_idx

        # 3. 构建结果
        idx = int(best_idx)
        mz_obs = float(spectrum_mz[idx])
        ppm = (mz_obs - mz_val) / mz_val * 1e6 if mz_val != 0 else float("inf")
        within = within_ppm(mz_obs, mz_val, tol_ppm)
        
        row = {
            "theory_mz": mz_val,
            "theory_int": float(theory_int[i]) if theory_int is not None and i < len(theory_int) else "",
            "obs_mz": mz_obs,
            "ppm": float(ppm),
            "obs_int": float(spectrum_int[idx]),
            "within": bool(within),
            "obs_idx": idx,
        }
        out.append(row)
        
    return out


def get_local_centroids_window(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    center_mz: float,
    lb: float,
    ub: float,
) -> np.ndarray:
    start = int(np.searchsorted(spectrum_mz, center_mz + float(lb), side="left"))
    end = int(np.searchsorted(spectrum_mz, center_mz + float(ub), side="right"))
    if end <= start:
        return np.empty((0, 2), dtype=float)
    out = np.empty((end - start, 2), dtype=float)
    out[:, 0] = spectrum_mz[start:end]
    out[:, 1] = spectrum_int[start:end]
    return out


def isodec_css_and_accept(
    centroids: np.ndarray,
    isodist: np.ndarray,
    z: int,
    peakmz: float,
    config: cfg.IsoDecConfig,
) -> tuple[bool, float, Optional[float]]:
    if centroids is None or centroids.size == 0 or isodist is None or isodist.size == 0:
        return False, 0.0, None
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        return False, 0.0, None
    if isodist.ndim != 2 or isodist.shape[1] != 2:
        return False, 0.0, None
    if (
        cfg.isodec_find_matched_intensities is None
        or cfg.isodec_calculate_cosinesimilarity is None
        or cfg.isodec_make_shifted_peak is None
    ):
        return False, 0.0, None

    max_shift = 1 if bool(config.minusoneaszero) else 0
    cent_intensities = cfg.isodec_find_matched_intensities(
        centroids[:, 0],
        centroids[:, 1],
        isodist[:, 0],
        int(max_shift),
        tolerance=float(config.matchtol),
        z=int(z),
        peakmz=float(peakmz),
    )

    max_theory = float(np.max(isodist[:, 1]))
    if max_theory <= 0.0:
        return False, 0.0, None
    max_obs = float(np.max(cent_intensities)) if len(cent_intensities) else 0.0
    norm_factor = max_obs / max_theory if max_obs > 0 else 1.0
    if not np.isfinite(norm_factor) or norm_factor <= 0:
        norm_factor = 1.0

    isodist_scaled = isodist.copy()
    isodist_scaled[:, 1] *= norm_factor
    css = float(
        cfg.isodec_calculate_cosinesimilarity(
            cent_intensities,
            isodist_scaled[:, 1],
            0,
            int(max_shift),
            minusoneaszero=bool(config.minusoneaszero),
        )
    )

    massdist = np.column_stack(
        [
            (isodist_scaled[:, 0] * float(z)) - (float(config.adductmass) * float(z)),
            isodist_scaled[:, 1],
        ]
    )
    monoiso = float(massdist[0, 0]) if massdist.size else 0.0
    peakmz_new, *_ = cfg.isodec_make_shifted_peak(
        0,
        float(css),
        float(monoiso),
        massdist,
        isodist_scaled,
        float(peakmz),
        int(z),
        centroids,
        float(config.matchtol),
        int(config.minpeaks),
        float(config.plusoneintwindowlb),
        float(config.plusoneintwindowub),
        float(config.css_thresh),
        float(config.minareacovered),
        bool(config.verbose),
    )
    return peakmz_new is not None, css, peakmz_new


def diagnose_candidate(
    residues: list[tuple[str, list[str]]],
    ion_type: str,
    frag_len: int,
    z: int,
    loss_formula: str,
    loss_count: int,
    h_transfer: int,
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    match_tol_ppm: float,
    min_obs_rel_int: float,
    rel_intensity_cutoff: float,
    mz_min,
    mz_max,
    isodec_config: Optional[cfg.IsoDecConfig],
) -> dict:
    # Ensure css_similarity is available in all branches
    from personalized_theory import css_similarity
    
    result = {
        "ion_type": ion_type,
        "frag_len": int(frag_len),
        "z": int(z),
        "loss_formula": loss_formula,
        "loss_count": int(loss_count),
        "h_transfer": int(h_transfer),
        "ok": False,
        "reason": "",
        "diagnostic_steps": [],
    }

    # Get base ion composition
    frag_name, base_frag_comp = ion_composition_from_sequence(residues, ion_type, frag_len, amidated=cfg.AMIDATED)
    
    # Get disulfide variants
    cys_variants = get_disulfide_logic(ion_type, frag_len, len(residues))
    
    # If no disulfide variants, use the original composition
    if not cys_variants:
        cys_variants = [("", None)]
    
    # Process each disulfide variant
    best_result = None
    best_score = -1
    
    for variant_suffix, shift_comp in cys_variants:
        # Create a copy of the result dictionary for this variant
        variant_result = result.copy()
        variant_result["diagnostic_steps"] = []
        
        # Apply disulfide shift if present
        if shift_comp:
            try:
                frag_comp = base_frag_comp + shift_comp
                variant_name = frag_name + variant_suffix
                variant_result["frag_name"] = variant_name
                variant_result["diagnostic_steps"].append({"step": "1. Ion composition", "status": "pass", "details": f"Generated {variant_name} (with disulfide shift)"})
            except Exception as e:
                variant_result["reason"] = f"disulfide_shift_failed: {e}"
                variant_result["diagnostic_steps"].append({"step": "1. Ion composition", "status": "fail", "details": str(e)})
                continue
        else:
            frag_comp = base_frag_comp
            variant_result["frag_name"] = frag_name
            variant_result["diagnostic_steps"].append({"step": "1. Ion composition", "status": "pass", "details": f"Generated {frag_name}"})

        try:
            loss_comp = apply_neutral_loss(frag_comp, loss_formula, loss_count)
            variant_result["diagnostic_steps"].append({"step": "2. Neutral loss", "status": "pass", "details": f"Applied {loss_count}x{loss_formula}"})
        except Exception as e:
            variant_result["reason"] = f"neutral_loss_failed: {e}"
            variant_result["diagnostic_steps"].append({"step": "2. Neutral loss", "status": "fail", "details": str(e)})
            continue
        
        formula = getattr(loss_comp, "formula", None)
        variant_result["formula"] = str(formula) if formula else str(loss_comp)
        
        try:
            dist0 = theoretical_isodist_from_comp(loss_comp, z)
            variant_result["diagnostic_steps"].append({"step": "3. Theoretical spectrum", "status": "pass", "details": f"Generated {len(dist0)} peaks"})
        except Exception as e:
            variant_result["reason"] = f"theory_failed: {e}"
            variant_result["diagnostic_steps"].append({"step": "3. Theoretical spectrum", "status": "fail", "details": str(e)})
            continue
        
        if dist0.size == 0:
            variant_result["reason"] = "theory_empty"
            variant_result["diagnostic_steps"].append({"step": "3. Theoretical spectrum", "status": "fail", "details": "Empty theoretical spectrum"})
            continue
        
        # Apply fragments mode's H-transfer mixture model
        obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0
        series = ion_series(ion_type)
        allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_1H))
        allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_2H))
        
        # For backward compatibility, if user specifies a specific H-transfer, use it directly
        # Import necessary functions from personalized_theory for H-transfer model
        from personalized_theory import observed_intensities_isodec
        
        if h_transfer != 0:
            variant_result["diagnostic_steps"].append({"step": "4. H-transfer mode", "status": "info", "details": f"Using specified H-transfer: {h_transfer:+d} H+"})
            # Apply the specified H-transfer
            hz = float(cfg.H_TRANSFER_MASS) / float(z)
            dist = dist0.copy()
            dist[:, 0] += float(h_transfer) * hz
            
            # Calculate observed intensities
            peak_mz = float(dist[np.argmax(dist[:, 1]), 0])
            y_obs = observed_intensities_isodec(
                spectrum_mz,
                spectrum_int,
                dist[:, 0],
                z=int(z),
                match_tol_ppm=float(match_tol_ppm),
                peak_mz=peak_mz,
            )
            
            best_model = f"{h_transfer:+d}H"
            best_score_variant = css_similarity(y_obs, dist[:, 1])
            best_pred = dist[:, 1]
            best_weights = {f"{h_transfer:+d}H": 1.0}
        else:
            # Use fragments mode's H-transfer mixture model
            variant_result["diagnostic_steps"].append({"step": "4. H-transfer mixture model", "status": "info", "details": "Applying fragments mode mixture model"})
            
            shift_1 = float(cfg.H_TRANSFER_MASS) / float(z)
            shift_2 = 2.0 * float(cfg.H_TRANSFER_MASS) / float(z)

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

            from personalized_theory import build_sample_axis, observed_intensities_isodec, vectorize_dist, css_similarity, fit_simplex_mixture
            
            sample_keys, sample_mzs, scale = build_sample_axis(
                dists_for_union,
                decimals=6,
                mz_min=mz_min,
                mz_max=mz_max,
            )
            
            if len(sample_mzs) == 0:
                variant_result["reason"] = "sample_axis_empty"
                variant_result["diagnostic_steps"].append({"step": "4. H-transfer mixture model", "status": "fail", "details": "Empty sample axis"})
                continue

            peak_mz = float(dist0[np.argmax(dist0[:, 1]), 0])
            y_obs = observed_intensities_isodec(
                spectrum_mz,
                spectrum_int,
                sample_mzs,
                z=int(z),
                match_tol_ppm=float(match_tol_ppm),
                peak_mz=peak_mz,
            )
            
            y0 = vectorize_dist(dist0, sample_keys, scale, mz_min=mz_min, mz_max=mz_max)
            neutral_score = css_similarity(y_obs, y0)
            
            best_model = "neutral"
            best_score_variant = neutral_score
            best_pred = y0
            best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}

            if allow_1h or allow_2h:
                yp1 = vectorize_dist(dist_p1, sample_keys, scale, mz_min=mz_min, mz_max=mz_max) if allow_1h else None
                ym1 = vectorize_dist(dist_m1, sample_keys, scale, mz_min=mz_min, mz_max=mz_max) if allow_1h else None
                yp2 = vectorize_dist(dist_p2, sample_keys, scale, mz_min=mz_min, mz_max=mz_max) if allow_2h else None
                ym2 = vectorize_dist(dist_m2, sample_keys, scale, mz_min=mz_min, mz_max=mz_max) if allow_2h else None

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

                if score_plus > best_score_variant:
                    best_model = "+mix"
                    best_score_variant = score_plus
                    best_pred = y_plus
                    best_weights = {
                        "0": weights_plus.get("0", 0.0),
                        "+H": weights_plus.get("+H", 0.0),
                        "+2H": weights_plus.get("+2H", 0.0),
                        "-H": 0.0,
                        "-2H": 0.0,
                    }

                if score_minus > best_score_variant:
                    best_model = "-mix"
                    best_score_variant = score_minus
                    best_pred = y_minus
                    best_weights = {
                        "0": weights_minus.get("0", 0.0),
                        "+H": 0.0,
                        "+2H": 0.0,
                        "-H": weights_minus.get("-H", 0.0),
                        "-2H": weights_minus.get("-2H", 0.0),
                    }

            rel_improve = (best_score_variant - neutral_score) / max(neutral_score, 1e-12)
            if best_model != "neutral" and rel_improve < float(cfg.H_TRANSFER_MIN_REL_IMPROVEMENT):
                best_model = "neutral"
                best_score_variant = neutral_score
                best_pred = y0
                best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}
        
        variant_result["diagnostic_steps"].append({"step": "4. H-transfer mixture model", "status": "pass", "details": f"Selected model: {best_model}, CSS: {best_score_variant:.4f}"})
        
        # Update h_transfer based on selected model
        if best_model == "neutral":
            variant_result["h_transfer"] = 0
        elif best_model == "+mix":
            # Select the highest weight H-transfer value from +mix model
            if best_weights["+2H"] > best_weights["+H"]:
                variant_result["h_transfer"] = 2
            elif best_weights["+H"] > 0:
                variant_result["h_transfer"] = 1
            else:
                variant_result["h_transfer"] = 0
        elif best_model == "-mix":
            # Select the highest weight H-transfer value from -mix model
            if best_weights["-2H"] > best_weights["-H"]:
                variant_result["h_transfer"] = -2
            elif best_weights["-H"] > 0:
                variant_result["h_transfer"] = -1
            else:
                variant_result["h_transfer"] = 0
        
        # Create dist_plot for anchor selection
        dist_plot = np.column_stack([sample_mzs.copy(), best_pred.copy()]) if h_transfer == 0 else None
    
        variant_result["raw_cosine_preanchor"] = float(best_score_variant)
    
        # Calculate masses with best H-transfer
        if h_transfer != 0:
            mono_mass = float(loss_comp.mass()) + (float(h_transfer) * float(cfg.H_TRANSFER_MASS))
            avg_mass = float(loss_comp.mass(average=True)) + (float(h_transfer) * float(cfg.H_TRANSFER_MASS))
        else:
            # For mixture model, use neutral mass as reference
            mono_mass = float(loss_comp.mass())
            avg_mass = float(loss_comp.mass(average=True))
        
        variant_result["mono_mass"] = mono_mass
        variant_result["avg_mass"] = avg_mass
    
        # Anchor peak selection (same as fragments mode)
        anchor_theory_mz = None
        obs_idx = None
        obs_mz = None
        obs_int = None
        anchor_hits = 0
        
        if h_transfer != 0:
            # For specified H-transfer, use the simple approach
            sorted_idx = np.argsort(dist[:, 1])[::-1][: int(cfg.ANCHOR_TOP_N)]
            for idx in sorted_idx:
                mz_candidate = float(dist[int(idx), 0])
                obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
                obs_mz_c = float(spectrum_mz[obs_idx_c])
                if not within_ppm(obs_mz_c, mz_candidate, float(match_tol_ppm)):
                    continue
                obs_int_c = float(spectrum_int[obs_idx_c])
                if float(min_obs_rel_int) > 0 and obs_int_c < obs_max * float(min_obs_rel_int):
                    continue
                anchor_hits += 1
                if anchor_theory_mz is None:
                    anchor_theory_mz = mz_candidate
                    obs_idx = int(obs_idx_c)
                    obs_mz = obs_mz_c
                    obs_int = obs_int_c
        else:
            # For mixture model, use the neutral distribution for anchor selection
            sorted_idx = np.argsort(dist0[:, 1])[::-1][: int(cfg.ANCHOR_TOP_N)]
            for idx in sorted_idx:
                mz_candidate = float(dist0[int(idx), 0])
                obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
                obs_mz_c = float(spectrum_mz[obs_idx_c])
                if not within_ppm(obs_mz_c, mz_candidate, float(match_tol_ppm)):
                    continue
                obs_int_c = float(spectrum_int[obs_idx_c])
                if float(min_obs_rel_int) > 0 and obs_int_c < obs_max * float(min_obs_rel_int):
                    continue
                anchor_hits += 1
                if anchor_theory_mz is None:
                    anchor_theory_mz = mz_candidate
                    obs_idx = int(obs_idx_c)
                    obs_mz = obs_mz_c
                    obs_int = obs_int_c
        
        # Check if anchor selection was successful
        if anchor_theory_mz is None:
            variant_result["reason"] = "anchor_outside_ppm"
            variant_result["diagnostic_steps"].append({"step": "5. Anchor selection", "status": "fail", "details": f"No anchor peak found within {match_tol_ppm} ppm and {min_obs_rel_int} relative intensity"})
            # Continue to next variant
            continue
        
        variant_result["anchor_theory_mz"] = float(anchor_theory_mz)
        variant_result["obs_idx"] = int(obs_idx)
        variant_result["obs_mz"] = float(obs_mz)
        variant_result["obs_int"] = float(obs_int)
        variant_result["anchor_hits"] = int(anchor_hits)
        variant_result["diagnostic_steps"].append({"step": "5. Anchor selection", "status": "pass", "details": f"Found anchor peak at {obs_mz:.4f} (theory: {anchor_theory_mz:.4f})"})
    
        # Calculate ppm offset
        ppm_offset = ((float(obs_mz) - float(anchor_theory_mz)) / float(anchor_theory_mz)) * 1e6
        variant_result["ppm_offset"] = float(ppm_offset)
        variant_result["diagnostic_steps"].append({"step": "6. PPM offset calculation", "status": "pass", "details": f"Calculated ppm offset: {ppm_offset:.2f}"})
    
        # Apply ppm offset to all theoretical peaks
        if h_transfer != 0:
            dist_calibrated = dist.copy()
            dist_calibrated[:, 0] += (float(ppm_offset) / 1e6) * dist_calibrated[:, 0]
        else:
            dist_calibrated = dist0.copy()
            dist_calibrated[:, 0] += (float(ppm_offset) / 1e6) * dist_calibrated[:, 0]
        
        # Calculate calibrated masses
        variant_result["mono_mass_calibrated"] = float(loss_comp.mass()) + (float(ppm_offset) / 1e6) * float(loss_comp.mass())
        variant_result["avg_mass_calibrated"] = float(loss_comp.mass(average=True)) + (float(ppm_offset) / 1e6) * float(loss_comp.mass(average=True))
    
        # Isotope deconvolution and final CSS score
        if isodec_config:
            from personalized_theory import observed_intensities_isodec
            
            # Use the calibrated distribution for isotope deconvolution
            peak_mz_calibrated = float(dist_calibrated[np.argmax(dist_calibrated[:, 1]), 0])
            y_obs_calibrated = observed_intensities_isodec(
                spectrum_mz,
                spectrum_int,
                dist_calibrated[:, 0],
                z=int(z),
                match_tol_ppm=float(match_tol_ppm),
                peak_mz=peak_mz_calibrated,
            )
            
            # Calculate final CSS score with calibrated distribution
            final_score = css_similarity(y_obs_calibrated, dist_calibrated[:, 1])
            variant_result["final_cosine"] = float(final_score)
            variant_result["diagnostic_steps"].append({"step": "7. Final CSS calculation", "status": "pass", "details": f"Calculated final CSS score: {final_score:.4f}"})
        else:
            # Use the pre-anchor score as the final score
            final_score = best_score_variant
            variant_result["final_cosine"] = float(final_score)
            variant_result["diagnostic_steps"].append({"step": "7. Final CSS calculation", "status": "pass", "details": f"Using pre-anchor CSS score: {final_score:.4f}"})
    
        # Determine if the match is acceptable
        if final_score >= float(cfg.MIN_COSINE):
            variant_result["ok"] = True
            variant_result["reason"] = "ok"
            variant_result["diagnostic_steps"].append({"step": "8. Match acceptance", "status": "pass", "details": f"Match accepted with CSS score: {final_score:.4f}"})
        else:
            variant_result["ok"] = False
            variant_result["reason"] = f"cosine_below_threshold: {final_score:.4f} < {cfg.MIN_COSINE}"
            variant_result["diagnostic_steps"].append({"step": "8. Match acceptance", "status": "fail", "details": f"Match rejected: CSS score below threshold"})
        
        # Update best result if this variant is better
        if final_score > best_score:
            best_score = final_score
            best_result = variant_result
    
    # If no variant was successful, return the original result with an error
    if best_result is None:
        result["reason"] = "all_disulfide_variants_failed"
        result["diagnostic_steps"].append({"step": "1. Ion composition", "status": "fail", "details": "All disulfide variants failed"})
        return result
    
    # Return the best result
    return best_result

    # Apply fragments mode's H-transfer mixture model
    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0
    series = ion_series(ion_type)
    allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_1H))
    allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (series in set(cfg.H_TRANSFER_ION_TYPES_2H))
    
    # For backward compatibility, if user specifies a specific H-transfer, use it directly
    # Import necessary functions from personalized_theory for H-transfer model
    from personalized_theory import observed_intensities_isodec
    
    if h_transfer != 0:
        result["diagnostic_steps"].append({"step": "4. H-transfer mode", "status": "info", "details": f"Using specified H-transfer: {h_transfer:+d} H+"})
        # Apply the specified H-transfer
        hz = float(cfg.H_TRANSFER_MASS) / float(z)
        dist = dist0.copy()
        dist[:, 0] += float(h_transfer) * hz
        
        # Calculate observed intensities
        peak_mz = float(dist[np.argmax(dist[:, 1]), 0])
        y_obs = observed_intensities_isodec(
            spectrum_mz,
            spectrum_int,
            dist[:, 0],
            z=int(z),
            match_tol_ppm=float(match_tol_ppm),
            peak_mz=peak_mz,
        )
        
        best_model = f"{h_transfer:+d}H"
        best_score = css_similarity(y_obs, dist[:, 1])
        best_pred = dist[:, 1]
        best_weights = {f"{h_transfer:+d}H": 1.0}
    else:
        # Use fragments mode's H-transfer mixture model
        result["diagnostic_steps"].append({"step": "4. H-transfer mixture model", "status": "info", "details": "Applying fragments mode mixture model"})
        
        shift_1 = float(cfg.H_TRANSFER_MASS) / float(z)
        shift_2 = 2.0 * float(cfg.H_TRANSFER_MASS) / float(z)

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

        from personalized_theory import build_sample_axis, observed_intensities_isodec, vectorize_dist, css_similarity, fit_simplex_mixture
        
        sample_keys, sample_mzs, scale = build_sample_axis(
            dists_for_union,
            decimals=6,
            mz_min=mz_min,
            mz_max=mz_max,
        )
        
        if len(sample_mzs) == 0:
            result["reason"] = "sample_axis_empty"
            result["diagnostic_steps"].append({"step": "4. H-transfer mixture model", "status": "fail", "details": "Empty sample axis"})
            return result

        peak_mz = float(dist0[np.argmax(dist0[:, 1]), 0])
        y_obs = observed_intensities_isodec(
            spectrum_mz,
            spectrum_int,
            sample_mzs,
            z=int(z),
            match_tol_ppm=float(match_tol_ppm),
            peak_mz=peak_mz,
        )
        
        y0 = vectorize_dist(dist0, sample_keys, scale, mz_min=mz_min, mz_max=mz_max)
        neutral_score = css_similarity(y_obs, y0)
        
        best_model = "neutral"
        best_score = neutral_score
        best_pred = y0
        best_weights = {"0": 1.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}

        if allow_1h or allow_2h:
            yp1 = vectorize_dist(dist_p1, sample_keys, scale, mz_min=mz_min, mz_max=mz_max) if allow_1h else None
            ym1 = vectorize_dist(dist_m1, sample_keys, scale, mz_min=mz_min, mz_max=mz_max) if allow_1h else None
            yp2 = vectorize_dist(dist_p2, sample_keys, scale, mz_min=mz_min, mz_max=mz_max) if allow_2h else None
            ym2 = vectorize_dist(dist_m2, sample_keys, scale, mz_min=mz_min, mz_max=mz_max) if allow_2h else None

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
        
        result["diagnostic_steps"].append({"step": "4. H-transfer mixture model", "status": "pass", "details": f"Selected model: {best_model}, CSS: {best_score:.4f}"})
        
        # Update h_transfer based on selected model
        if best_model == "neutral":
            result["h_transfer"] = 0
        elif best_model == "+mix":
            # Select the highest weight H-transfer value from +mix model
            if best_weights["+2H"] > best_weights["+H"]:
                result["h_transfer"] = 2
            elif best_weights["+H"] > 0:
                result["h_transfer"] = 1
            else:
                result["h_transfer"] = 0
        elif best_model == "-mix":
            # Select the highest weight H-transfer value from -mix model
            if best_weights["-2H"] > best_weights["-H"]:
                result["h_transfer"] = -2
            elif best_weights["-H"] > 0:
                result["h_transfer"] = -1
            else:
                result["h_transfer"] = 0
        
        # Create dist_plot for anchor selection
        dist_plot = np.column_stack([sample_mzs.copy(), best_pred.copy()])
    
    result["raw_cosine_preanchor"] = float(best_score)
    
    # Calculate masses with best H-transfer
    if h_transfer != 0:
        mono_mass = float(loss_comp.mass()) + (float(h_transfer) * float(cfg.H_TRANSFER_MASS))
        avg_mass = float(loss_comp.mass(average=True)) + (float(h_transfer) * float(cfg.H_TRANSFER_MASS))
    else:
        # For mixture model, use neutral mass as reference
        mono_mass = float(loss_comp.mass())
        avg_mass = float(loss_comp.mass(average=True))
    
    result["mono_mass"] = mono_mass
    result["avg_mass"] = avg_mass
    
    # Anchor peak selection (same as fragments mode)
    anchor_theory_mz = None
    obs_idx = None
    obs_mz = None
    obs_int = None
    anchor_hits = 0
    
    if h_transfer != 0:
        # For specified H-transfer, use the simple approach
        sorted_idx = np.argsort(dist[:, 1])[::-1][: int(cfg.ANCHOR_TOP_N)]
        for idx in sorted_idx:
            mz_candidate = float(dist[int(idx), 0])
            obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
            obs_mz_c = float(spectrum_mz[obs_idx_c])
            if not within_ppm(obs_mz_c, mz_candidate, float(match_tol_ppm)):
                continue
            obs_int_c = float(spectrum_int[obs_idx_c])
            if float(min_obs_rel_int) > 0 and obs_int_c < obs_max * float(min_obs_rel_int):
                continue
            anchor_hits += 1
            if anchor_theory_mz is None:
                anchor_theory_mz = mz_candidate
                obs_idx = int(obs_idx_c)
                obs_mz = obs_mz_c
                obs_int = obs_int_c
    else:
        # For mixture model, use the fragments mode approach
        sorted_idx = np.argsort(best_pred)[::-1][: int(cfg.ANCHOR_TOP_N)]
        for idx in sorted_idx:
            mz_candidate = float(sample_mzs[int(idx)])
            obs_idx_c = nearest_peak_index(spectrum_mz, mz_candidate)
            obs_mz_c = float(spectrum_mz[obs_idx_c])
            if not within_ppm(obs_mz_c, mz_candidate, float(match_tol_ppm)):
                continue
            obs_int_c = float(spectrum_int[obs_idx_c])
            if float(min_obs_rel_int) > 0 and obs_int_c < obs_max * float(min_obs_rel_int):
                continue
            anchor_hits += 1
            if anchor_theory_mz is None:
                anchor_theory_mz = mz_candidate
                obs_idx = int(obs_idx_c)
                obs_mz = obs_mz_c
                obs_int = obs_int_c
    
    if anchor_hits < int(cfg.ANCHOR_MIN_MATCHES) or anchor_theory_mz is None:
        result["reason"] = "anchor_outside_ppm"
        result["diagnostic_steps"].append({"step": "5. Anchor peak selection", "status": "fail", "details": f"Found {anchor_hits} anchor hits (needs {cfg.ANCHOR_MIN_MATCHES})"})
        result["obs_idx"] = int(obs_idx) if obs_idx is not None else ""
        return result
    
    result["diagnostic_steps"].append({"step": "5. Anchor peak selection", "status": "pass", "details": f"Found {anchor_hits} anchor hits, selected peak at {anchor_theory_mz:.4f} m/z"})
    
    result["anchor_theory_mz"] = float(anchor_theory_mz)
    result["anchor_obs_mz"] = float(obs_mz)
    result["anchor_within_ppm"] = True
    result["obs_idx"] = int(obs_idx)
    result["obs_int"] = float(obs_int)
    result["obs_rel_int"] = float(obs_int / obs_max) if obs_max > 0 else 0.0

    ppm = (obs_mz - anchor_theory_mz) / anchor_theory_mz * 1e6
    result["anchor_ppm"] = float(ppm)
    
    # Adjust spectrum with anchor shift
    if h_transfer != 0:
        dist_plot = dist.copy()
        dist_plot[:, 0] += obs_mz - anchor_theory_mz
        dist_plot[:, 1] *= obs_int / float(np.max(dist_plot[:, 1]))
    else:
        dist_plot[:, 0] += obs_mz - anchor_theory_mz
        dist_plot[:, 1] *= obs_int / float(np.max(dist_plot[:, 1]))
    
    result["diagnostic_steps"].append({"step": "6. Spectrum alignment", "status": "pass", "details": f"Shifted by {ppm:.2f} ppm"})

    # MZ window filter
    if mz_min is not None or mz_max is not None:
        mz_min_f = -np.inf if mz_min is None else float(mz_min)
        mz_max_f = np.inf if mz_max is None else float(mz_max)
        dist_plot = dist_plot[(dist_plot[:, 0] >= mz_min_f) & (dist_plot[:, 0] <= mz_max_f)]
        if dist_plot.size == 0:
            result["reason"] = "outside_mz_window"
            result["diagnostic_steps"].append({"step": "7. MZ window filter", "status": "fail", "details": f"No peaks in [{mz_min_f:.2f}, {mz_max_f:.2f}] m/z"})
            return result
    
    result["diagnostic_steps"].append({"step": "7. MZ window filter", "status": "pass", "details": f"{len(dist_plot)} peaks remaining"})

    # Relative intensity cutoff
    max_plot = float(np.max(dist_plot[:, 1]))
    keep = dist_plot[:, 1] >= max_plot * float(rel_intensity_cutoff)
    dist_plot = dist_plot[keep]
    if dist_plot.size == 0:
        result["reason"] = "below_rel_intensity_cutoff"
        result["diagnostic_steps"].append({"step": "8. Relative intensity cutoff", "status": "fail", "details": f"No peaks above {rel_intensity_cutoff*100:.1f}% relative intensity"})
        return result
    
    result["diagnostic_steps"].append({"step": "8. Relative intensity cutoff", "status": "pass", "details": f"{len(dist_plot)} peaks remaining after cutoff"})

    result["dist_plot"] = dist_plot
    result["theory_matches"] = match_theory_peaks(
        spectrum_mz,
        spectrum_int,
        dist_plot[:, 0],
        theory_int=dist_plot[:, 1],
        tol_ppm=float(match_tol_ppm),
    )

    if cfg.ENABLE_ISODEC_RULES and isodec_config is not None:
        local_centroids = get_local_centroids_window(
            spectrum_mz, spectrum_int, obs_mz, isodec_config.mzwindowlb, isodec_config.mzwindowub
        )
        accepted, css, _ = isodec_css_and_accept(
            local_centroids, dist_plot, z=z, peakmz=obs_mz, config=isodec_config
        )
        result["isodec_css"] = float(css)
        result["isodec_accepted"] = bool(accepted)

        if cfg.isodec_find_matches is not None and local_centroids.size and dist_plot.size:
            matchedindexes, isomatches = cfg.isodec_find_matches(
                local_centroids, dist_plot, float(isodec_config.matchtol)
            )
            mi = np.array(matchedindexes, dtype=int) if len(matchedindexes) else np.array([], dtype=int)
            ii = np.array(isomatches, dtype=int) if len(isomatches) else np.array([], dtype=int)
            matchedcentroids = local_centroids[mi] if mi.size else np.empty((0, 2), dtype=float)
            matchediso = dist_plot[ii] if ii.size else np.empty((0, 2), dtype=float)

            minpeaks_eff = int(isodec_config.minpeaks)
            if int(z) == 1 and len(matchedindexes) == 2 and len(isomatches) == 2:
                if isomatches[0] == 0 and isomatches[1] == 1:
                    int1 = float(local_centroids[matchedindexes[0], 1])
                    int2 = float(local_centroids[matchedindexes[1], 1])
                    ratio = (int2 / int1) if int1 != 0 else 0.0
                    if float(isodec_config.plusoneintwindowlb) < ratio < float(isodec_config.plusoneintwindowub):
                        minpeaks_eff = 2

            areacovered = (
                float(np.sum(matchediso[:, 1])) / float(np.sum(local_centroids[:, 1]))
                if local_centroids.size and np.sum(local_centroids[:, 1]) > 0
                else 0.0
            )
            topn = minpeaks_eff
            topthree = False
            if local_centroids.size and matchedcentroids.size and topn > 0:
                top_iso = np.sort(matchedcentroids[:, 1])[::-1][:topn]
                top_cent = np.sort(local_centroids[:, 1])[::-1][:topn]
                topthree = bool(np.array_equal(top_iso, top_cent))

            result["isodec_detail"] = {
                "local_centroids_n": int(local_centroids.shape[0]),
                "matched_peaks_n": int(len(matchedindexes)),
                "minpeaks_effective": int(minpeaks_eff),
                "css_thresh": float(isodec_config.css_thresh),
                "minareacovered": float(isodec_config.minareacovered),
                "areacovered": float(areacovered),
                "topthree": bool(topthree),
            }

        if not accepted:
            result["reason"] = "failed_isodec_rules"
            return result

    result["ok"] = True
    result["reason"] = "accepted"
    return result


def parse_fragment_spec(spec: str) -> tuple[str, int, str, int, Optional[int]]:
    """
    Parse an ion spec like:
      - "c7"
      - "z12-2H2O^3+"
      - "z-dot12-CO"

    Returns: (ion_type, frag_len, loss_formula, loss_count, charge)
    loss_formula is one of {"", "H2O", "NH3", "CO", "CO2"}.
    """
    s = str(spec).strip().replace(" ", "")
    if not s:
        raise ValueError("Empty DIAGNOSE_ION_SPEC.")

    charge = None
    m = re.search(r"(?:\^)?(\d+)\+$", s)
    if m:
        charge = int(m.group(1))
        s = s[: m.start()]

    m = re.match(r"(.+?)(\d+)(.*)$", s)
    if not m:
        raise ValueError(
            f"Could not parse ion spec '{spec}'. Expected something like 'c7', 'z12-2H2O^3+'."
        )
    ion_type = m.group(1)
    frag_len = int(m.group(2))
    tail = m.group(3) or ""

    loss_formula = ""
    loss_count = 0
    if tail:
        tail = tail.replace("(", "").replace(")", "").replace("×", "").replace("x", "")
        m2 = re.fullmatch(r"-(\d+)?(H2O|NH3|CO|CO2)", tail)
        if not m2:
            raise ValueError(
                f"Unsupported loss suffix '{tail}' in '{spec}'. "
                "Use '', '-H2O', '-2H2O', '-NH3', '-2NH3', '-CO', or '-CO2'."
            )
        loss_count = int(m2.group(1) or 1)
        loss_formula = str(m2.group(2))

    return ion_type, frag_len, loss_formula, loss_count, charge
