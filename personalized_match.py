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


def composition_to_formula(comp, proton_count: int = 0) -> str:
    """Format a pyteomics Composition as an ordered chemical formula string."""
    if comp is None:
        return ""
    order = [
        "C",
        "H",
        "N",
        "O",
        "S",
        "P",
        "Cl",
        "Br",
        "Na",
        "K",
        "Fe",
        "Ca",
        "Mg",
        "Zn",
        "Ni",
    ]
    try:
        items = {str(k): int(v) for k, v in comp.items() if int(v) != 0}
    except Exception:
        return str(comp)
    if proton_count:
        items["H"] = items.get("H", 0) + int(proton_count)
        if items["H"] == 0:
            items.pop("H", None)

    parts: list[str] = []
    used: set[str] = set()
    for el in order:
        n = items.get(el, 0)
        if n:
            parts.append(f"{el}{n}")
            used.add(el)
    for el in sorted(k for k in items.keys() if k not in used):
        parts.append(f"{el}{items[el]}")
    return "".join(parts) if parts else ""


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
    mode: str = "fragments",
    mz_min: Optional[float] = None,
    mz_max: Optional[float] = None,
) -> tuple[float, int]:
    cap = 0.0
    hits = 0
    n = len(residues)
    mz_min_f = None if mz_min is None else float(mz_min)
    mz_max_f = None if mz_max is None else float(mz_max)

    # Get monomer neutral composition (for complex mode)
    from personalized_sequence import get_neutral_monomer_composition

    monomer_comp = get_neutral_monomer_composition(residues)

    for ion_type in cfg.ION_TYPES:
        series = ion_series(ion_type)
        allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (
            series in set(cfg.H_TRANSFER_ION_TYPES_1H)
        )
        allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (
            series in set(cfg.H_TRANSFER_ION_TYPES_2H)
        )

        for frag_len in range(1, n):
            try:
                _, frag_comp = ion_composition_from_sequence(
                    residues, ion_type, frag_len, amidated=cfg.AMIDATED
                )
            except Exception:
                continue

            # Get disulfide variants
            cys_variants = get_disulfide_logic(ion_type, frag_len, len(residues))

            # If no disulfide variants, use the original composition
            if not cys_variants:
                cys_variants = [("", None)]

            for variant_suffix, shift_comp in cys_variants:
                # Apply disulfide shift if present
                if shift_comp is not None:
                    try:
                        variant_comp = frag_comp + shift_comp
                    except Exception:
                        continue
                else:
                    variant_comp = frag_comp

                if mode == "complex_fragments":
                    # For complex_fragments mode, only consider complex fragments (monomer + fragment) without neutral losses
                    target_comp = variant_comp + monomer_comp
                    # Process this target_comp once
                    for z in range(
                        int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1
                    ):
                        try:
                            dist0 = theoretical_isodist_from_comp(target_comp, z)
                        except Exception:
                            continue
                        if dist0.size == 0:
                            continue

                        anchor = float(dist0[get_anchor_idx(dist0), 0])
                        if allow_1h or allow_2h:
                            shift_1 = (
                                float(cfg.H_TRANSFER_MASS) / float(z)
                                if (allow_1h or allow_2h)
                                else 0.0
                            )
                            shift_2 = (
                                2.0 * float(cfg.H_TRANSFER_MASS) / float(z)
                                if allow_2h
                                else 0.0
                            )
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
                            m = max_intensity_in_ppm_window(
                                spectrum_mz, spectrum_int, mz0, tol_ppm=float(tol_ppm)
                            )
                            if m > 0.0:
                                hits += 1
                                if m > cap:
                                    cap = m
                else:
                    # For regular fragments mode, consider neutral losses
                    for _, loss_comp in neutral_loss_variants(
                        variant_comp, ion_series_letter=series
                    ):
                        target_comp = loss_comp
                        for z in range(
                            int(cfg.FRAG_MIN_CHARGE), int(cfg.FRAG_MAX_CHARGE) + 1
                        ):
                            try:
                                dist0 = theoretical_isodist_from_comp(target_comp, z)
                            except Exception:
                                continue
                            if dist0.size == 0:
                                continue

                            anchor = float(dist0[get_anchor_idx(dist0), 0])
                            if allow_1h or allow_2h:
                                shift_1 = (
                                    float(cfg.H_TRANSFER_MASS) / float(z)
                                    if (allow_1h or allow_2h)
                                    else 0.0
                                )
                                shift_2 = (
                                    2.0 * float(cfg.H_TRANSFER_MASS) / float(z)
                                    if allow_2h
                                    else 0.0
                                )
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
                                m = max_intensity_in_ppm_window(
                                    spectrum_mz,
                                    spectrum_int,
                                    mz0,
                                    tol_ppm=float(tol_ppm),
                                )
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
            "theory_int": float(theory_int[i])
            if theory_int is not None and i < len(theory_int)
            else "",
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
    force_hill: bool | None = None,
) -> np.ndarray:
    from personalized_centroid import hill_centroid_window

    return hill_centroid_window(
        spectrum_mz, spectrum_int, center_mz, lb, ub, force_hill=force_hill
    )


def execute_hybrid_strategy(scoring_func, *args, **kwargs):
    """
    Run a scoring function twice (centroid vs raw) and select the better result.

    The scoring function must accept `use_centroid_logic: bool` and return either:
    - a dict-like result (with score in one of: score/css/isodec_css/final_cosine/raw_cosine_preanchor), or
    - None if no match.

    The returned dict is annotated with `strategy` = {"centroid","raw"}.
    """

    def _get_score(d: dict) -> float:
        if not isinstance(d, dict):
            return float("-inf")
        for key in (
            "score",
            "css",
            "isodec_css",
            "final_cosine",
            "raw_cosine_preanchor",
        ):
            v = d.get(key, None)
            try:
                if v is None:
                    continue
                v_f = float(v)
                if np.isfinite(v_f):
                    return v_f
            except Exception:
                continue
        return float("-inf")

    def _is_ok(d: dict) -> bool:
        if not isinstance(d, dict):
            return False
        if "ok" in d:
            return bool(d.get("ok"))
        if "accepted" in d:
            return bool(d.get("accepted"))
        if "isodec_accepted" in d:
            return bool(d.get("isodec_accepted"))
        return False

    res_centroid = scoring_func(*args, **kwargs, use_centroid_logic=True)
    res_raw = scoring_func(*args, **kwargs, use_centroid_logic=False)

    if res_centroid is None and res_raw is None:
        return None
    if res_centroid is not None and res_raw is None:
        if isinstance(res_centroid, dict):
            res_centroid["strategy"] = "centroid"
        return res_centroid
    if res_centroid is None and res_raw is not None:
        if isinstance(res_raw, dict):
            res_raw["strategy"] = "raw"
        return res_raw

    # Both exist: prefer OK results; then higher score; break ties toward centroid.
    ok_c = _is_ok(res_centroid)
    ok_r = _is_ok(res_raw)
    if ok_c and not ok_r:
        res_centroid["strategy"] = "centroid"
        return res_centroid
    if ok_r and not ok_c:
        res_raw["strategy"] = "raw"
        return res_raw

    score_c = _get_score(res_centroid)
    score_r = _get_score(res_raw)
    if score_c >= score_r:
        res_centroid["strategy"] = "centroid"
        return res_centroid
    res_raw["strategy"] = "raw"
    return res_raw


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
            bool(config.minusoneaszero),
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


def _classify_outlier_pattern(is_outlier: np.ndarray, ratios: np.ndarray) -> int:
    """
    Classify outlier distribution pattern.

    Returns:
        0 = none (no outliers)
        1 = low_mass (outliers in first half)
        2 = high_mass (outliers in second half)
        3 = spread (outliers distributed)
        4 = clustered (consecutive outliers)
    """
    if not np.any(is_outlier):
        return 0  # none

    outlier_indices = np.where(is_outlier)[0]
    n = len(is_outlier)

    # All in low mass region (first half)
    if np.all(outlier_indices < n / 2):
        return 1  # low_mass

    # All in high mass region (second half)
    if np.all(outlier_indices >= n / 2):
        return 2  # high_mass

    # Check if clustered (consecutive or near-consecutive)
    if len(outlier_indices) >= 2:
        gaps = np.diff(outlier_indices)
        if np.all(gaps <= 2):
            return 4  # clustered

    # Otherwise spread out
    return 3  # spread


def _detect_outlier_peaks(
    obs_int: np.ndarray,
    theory_int: np.ndarray,
    outlier_threshold: float = 1.5,
    max_peaks: int = 7,
) -> tuple[dict, list]:
    """
    Detect outlier peaks in isotope pattern.

    Args:
        obs_int: Observed intensities (aligned to theory)
        theory_int: Theoretical intensities (scaled)
        outlier_threshold: Ratio threshold for outlier detection
        max_peaks: Maximum number of isotope peaks to track (M+0 to M+6)

    Returns:
        outlier_features: dict with ML-friendly features
        outlier_detail: list of per-peak details
    """
    obs = np.asarray(obs_int, dtype=float)
    theory = np.asarray(theory_int, dtype=float)

    n_peaks = min(len(obs), max_peaks)

    # Calculate ratios (fixed size array for ML)
    ratios = np.zeros(max_peaks, dtype=float)
    for i in range(n_peaks):
        if theory[i] > 0:
            ratios[i] = obs[i] / theory[i]
        else:
            ratios[i] = 0.0

    # Identify outliers
    is_outlier = ratios > outlier_threshold
    outlier_count = int(np.sum(is_outlier[:n_peaks]))
    outlier_pct = outlier_count / n_peaks if n_peaks > 0 else 0.0
    max_ratio = float(np.max(ratios[:n_peaks])) if n_peaks > 0 else 1.0

    # Classify pattern
    pattern = _classify_outlier_pattern(is_outlier[:n_peaks], ratios[:n_peaks])

    # Build feature dict (ML-friendly)
    outlier_features = {
        "outlier_max_ratio": max_ratio,
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct,
        "ratios": ratios.tolist(),  # Fixed length 7
        "pattern": pattern,
    }

    # Build detail list (for debugging/analysis)
    outlier_detail = []
    for i in range(n_peaks):
        outlier_detail.append({
            "pos": f"M+{i}",
            "obs": float(obs[i]) if i < len(obs) else 0.0,
            "exp": float(theory[i]) if i < len(theory) else 0.0,
            "ratio": float(ratios[i]),
            "is_outlier": bool(is_outlier[i]),
        })

    return outlier_features, outlier_detail


def _calculate_fragments_gate(
    *,
    isodec_css: float,
    ppm: float,
    theory_matches: list[dict],
    dist_plot: np.ndarray,
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    local_centroids: np.ndarray | None,
    anchor_mz: float,
) -> dict:
    min_isodec_css = float(getattr(cfg, "FRAG_MIN_ISODEC_CSS", cfg.MIN_COSINE))
    max_anchor_abs_ppm_cfg = getattr(cfg, "FRAG_MAX_ANCHOR_ABS_PPM", None)
    max_anchor_abs_ppm = (
        float(max_anchor_abs_ppm_cfg)
        if max_anchor_abs_ppm_cfg is not None
        else float(cfg.MATCH_TOL_PPM) * 1.5
    )
    min_matched_peaks = max(1, int(getattr(cfg, "FRAG_MIN_MATCHED_PEAKS", 2)))
    max_pc_missing_peaks = float(getattr(cfg, "FRAG_MAX_PC_MISSING_PEAKS", 85.0))
    min_fit_score = float(getattr(cfg, "FRAG_MIN_FIT_SCORE", 0.35))
    min_correlation_cfg = getattr(cfg, "FRAG_MIN_CORRELATION", None)
    min_correlation = (
        float(min_correlation_cfg) if min_correlation_cfg is not None else None
    )

    matched_peaks = [m for m in theory_matches if m.get("within")]
    local_matches_count = len(matched_peaks)

    total_theory_peaks = len(theory_matches)
    num_missing = total_theory_peaks - local_matches_count
    pc_missing_peaks = (
        (num_missing / total_theory_peaks * 100.0) if total_theory_peaks > 0 else 100.0
    )

    if total_theory_peaks > 0 and dist_plot.size:
        theory_int = np.asarray(dist_plot[:, 1], dtype=float)
        obs_aligned = np.zeros(total_theory_peaks, dtype=float)
        for i, m in enumerate(theory_matches):
            if i < len(theory_int):
                obs_idx = m.get("obs_idx")
                if obs_idx is not None and 0 <= obs_idx < len(spectrum_int):
                    obs_aligned[i] = float(spectrum_int[obs_idx])

        scaled_theory, _ = _scale_theoretical_to_observed(theory_int, obs_aligned)
        fit_score = _fit_score_from_envelope(obs_aligned, scaled_theory)
        correlation_coefficient = _safe_pearson(obs_aligned, scaled_theory)
        outlier_features, outlier_detail = _detect_outlier_peaks(obs_aligned, scaled_theory)
    else:
        fit_score = 0.0
        correlation_coefficient = 0.0
        outlier_features, outlier_detail = _detect_outlier_peaks(np.array([]), np.array([]))

    checks = {
        "isodec_css": {
            "value": float(isodec_css),
            "threshold": f">= {min_isodec_css:.6f}",
            "threshold_value": min_isodec_css,
            "pass": float(isodec_css) >= min_isodec_css,
            "description": "Cosine similarity score",
        },
        "ppm": {
            "value": float(abs(ppm)),
            "threshold": f"<= {max_anchor_abs_ppm:.1f}",
            "threshold_value": max_anchor_abs_ppm,
            "pass": float(abs(ppm)) <= max_anchor_abs_ppm,
            "description": "Anchor mass error (ppm)",
        },
        "local_matches": {
            "value": int(local_matches_count),
            "threshold": f">= {min_matched_peaks}",
            "threshold_value": min_matched_peaks,
            "pass": int(local_matches_count) >= min_matched_peaks,
            "description": "Number of matched peaks",
        },
        "pc_missing_peaks": {
            "value": float(pc_missing_peaks),
            "threshold": f"<= {max_pc_missing_peaks:.1f}",
            "threshold_value": max_pc_missing_peaks,
            "pass": float(pc_missing_peaks) <= max_pc_missing_peaks,
            "description": "Percentage of missing peaks",
        },
        "fit_score": {
            "value": float(fit_score),
            "threshold": f">= {min_fit_score:.6f}",
            "threshold_value": min_fit_score,
            "pass": float(fit_score) >= min_fit_score,
            "description": "Envelope fit score",
        },
    }

    if min_correlation is not None:
        checks["correlation_coefficient"] = {
            "value": float(correlation_coefficient),
            "threshold": f">= {min_correlation:.6f}",
            "threshold_value": min_correlation,
            "pass": float(correlation_coefficient) >= min_correlation,
            "description": "Intensity correlation coefficient",
        }

    legacy_accepted = all(check["pass"] for check in checks.values())

    failed_at = None
    for key, check in checks.items():
        if not check["pass"]:
            failed_at = key
            break

    return {
        "legacy_accepted": legacy_accepted,
        "failed_at": failed_at,
        "checks": checks,
        "quality_metrics": {
            "fit_score": float(fit_score),
            "correlation_coefficient": float(correlation_coefficient),
            "pc_missing_peaks": float(pc_missing_peaks),
            "local_matches_count": int(local_matches_count),
        },
        "outlier_features": outlier_features,
        "outlier_detail": outlier_detail,
    }


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
    use_centroid_logic: bool | None = None,
) -> dict:
    if use_centroid_logic is None:
        # Self-managed hybrid strategy wrapper.
        def _score(*, use_centroid_logic: bool):
            return diagnose_candidate(
                residues=residues,
                ion_type=ion_type,
                frag_len=frag_len,
                z=z,
                loss_formula=loss_formula,
                loss_count=loss_count,
                h_transfer=h_transfer,
                spectrum_mz=spectrum_mz,
                spectrum_int=spectrum_int,
                match_tol_ppm=match_tol_ppm,
                min_obs_rel_int=min_obs_rel_int,
                rel_intensity_cutoff=rel_intensity_cutoff,
                mz_min=mz_min,
                mz_max=mz_max,
                isodec_config=isodec_config,
                use_centroid_logic=use_centroid_logic,
            )

        best = execute_hybrid_strategy(_score)
        # Fallback to empty result shape if both strategies failed.
        return (
            best
            if isinstance(best, dict)
            else {
                "ion_type": ion_type,
                "frag_len": int(frag_len),
                "z": int(z),
                "loss_formula": loss_formula,
                "loss_count": int(loss_count),
                "h_transfer": int(h_transfer),
                "ok": False,
                "reason": "no_result",
                "diagnostic_steps": [],
            }
        )

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

    def _scaled_dist_for_display(
        base_mz: np.ndarray, base_int: np.ndarray, target_mz: float
    ) -> np.ndarray:
        """Scale theory intensities to local observed anchor intensity for display, even on failures."""
        dist_disp = np.column_stack(
            [np.asarray(base_mz, dtype=float), np.asarray(base_int, dtype=float)]
        )
        if dist_disp.size == 0:
            return dist_disp
        max_theory = float(np.max(dist_disp[:, 1])) if dist_disp.size else 0.0
        if max_theory <= 0:
            return dist_disp
        anchor_int = 0.0
        try:
            anchor_int = max_intensity_in_ppm_window(
                spectrum_mz,
                spectrum_int,
                float(target_mz),
                float(match_tol_ppm),
            )
        except Exception:
            anchor_int = 0.0
        if anchor_int <= 0 and len(spectrum_mz):
            idx_n = nearest_peak_index(spectrum_mz, float(target_mz))
            anchor_int = float(spectrum_int[idx_n])
        if anchor_int <= 0:
            anchor_int = float(np.max(spectrum_int)) if len(spectrum_int) else 1.0
        dist_disp[:, 1] *= float(anchor_int) / float(max_theory)
        return dist_disp

    frag_name, base_frag_comp = ion_composition_from_sequence(
        residues, ion_type, frag_len, amidated=cfg.AMIDATED
    )
    cys_variants = get_disulfide_logic(ion_type, frag_len, len(residues))
    if not cys_variants:
        cys_variants = [("", None)]

    obs_max = float(np.max(spectrum_int)) if len(spectrum_int) else 0.0
    series = ion_series(ion_type)
    allow_1h = bool(cfg.ENABLE_H_TRANSFER) and (
        series in set(cfg.H_TRANSFER_ION_TYPES_1H)
    )
    allow_2h = bool(cfg.ENABLE_H_TRANSFER) and (
        series in set(cfg.H_TRANSFER_ION_TYPES_2H)
    )
    result["strategy"] = "centroid" if bool(use_centroid_logic) else "raw"

    fragments_scoring_settings = None
    fragments_noise_model = None
    fragments_pipeline_eval = None
    try:
        # Import lazily to avoid a module-import cycle with personalized_modes.
        from personalized_modes import (
            _build_noise_level_model,
            _evaluate_fragment_pipeline,
            _fragment_scoring_settings,
        )

        fragments_scoring_settings = _fragment_scoring_settings(
            float(match_tol_ppm)
        )
        fragments_noise_model = _build_noise_level_model(
            np.asarray(spectrum_mz, dtype=float),
            np.asarray(spectrum_int, dtype=float),
            num_splits=int(fragments_scoring_settings["noise_model_splits"]),
            hist_bins=int(fragments_scoring_settings["noise_hist_bins"]),
        )
        fragments_pipeline_eval = _evaluate_fragment_pipeline
    except Exception:
        fragments_scoring_settings = None
        fragments_noise_model = None
        fragments_pipeline_eval = None

    variant_results: list[dict] = []

    for variant_suffix, shift_comp in cys_variants:
        variant_result = result.copy()
        variant_result["diagnostic_steps"] = []
        variant_result["variant_suffix"] = variant_suffix
        variant_result["variant_type"] = variant_type_from_suffix(variant_suffix)

        if shift_comp is not None:
            try:
                frag_comp = base_frag_comp + shift_comp
                variant_name = frag_name + variant_suffix
                variant_result["frag_name"] = variant_name
                variant_result["diagnostic_steps"].append(
                    {
                        "step": "1. Ion composition",
                        "status": "pass",
                        "details": f"Generated {variant_name} (with disulfide shift)",
                    }
                )
            except Exception as e:
                variant_result["reason"] = f"disulfide_shift_failed: {e}"
                variant_result["diagnostic_steps"].append(
                    {"step": "1. Ion composition", "status": "fail", "details": str(e)}
                )
                continue
        else:
            frag_comp = base_frag_comp
            variant_result["frag_name"] = frag_name
            variant_result["diagnostic_steps"].append(
                {
                    "step": "1. Ion composition",
                    "status": "pass",
                    "details": f"Generated {frag_name}",
                }
            )

        try:
            loss_comp = apply_neutral_loss(frag_comp, loss_formula, loss_count)
            variant_result["diagnostic_steps"].append(
                {
                    "step": "2. Neutral loss",
                    "status": "pass",
                    "details": f"Applied {loss_count}x{loss_formula}",
                }
            )
        except Exception as e:
            variant_result["reason"] = f"neutral_loss_failed: {e}"
            variant_result["diagnostic_steps"].append(
                {"step": "2. Neutral loss", "status": "fail", "details": str(e)}
            )
            continue

        variant_result["formula"] = composition_to_formula(
            loss_comp, proton_count=int(z)
        )

        try:
            dist0 = theoretical_isodist_from_comp(loss_comp, z)
            variant_result["diagnostic_steps"].append(
                {
                    "step": "3. Theoretical spectrum",
                    "status": "pass",
                    "details": f"Generated {len(dist0)} peaks",
                }
            )
        except Exception as e:
            variant_result["reason"] = f"theory_failed: {e}"
            variant_result["diagnostic_steps"].append(
                {"step": "3. Theoretical spectrum", "status": "fail", "details": str(e)}
            )
            continue

        if dist0.size == 0:
            variant_result["reason"] = "theory_empty"
            variant_result["diagnostic_steps"].append(
                {
                    "step": "3. Theoretical spectrum",
                    "status": "fail",
                    "details": "Empty theoretical spectrum",
                }
            )
            continue

        # Save neutral theory m/z early for diagnostics; explicit H-transfer may shift it later.
        dist0_theory_mz = float(dist0[get_anchor_idx(dist0), 0])
        variant_result["expected_theory_mz"] = dist0_theory_mz

        sample_mzs = None
        best_pred = None
        best_weights = None
        best_model = "neutral"
        best_score = 0.0

        # Diagnose mode should evaluate the explicitly requested H-transfer state, not auto-mix.
        variant_result["diagnostic_steps"].append(
            {
                "step": "4. H-transfer mode",
                "status": "info",
                "details": f"Requested H-transfer: {int(h_transfer):+d} H+",
            }
        )

        shift_1 = (
            float(cfg.H_TRANSFER_MASS) / float(z) if (allow_1h or allow_2h) else 0.0
        )
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

        model_dists = {"neutral": dist0}
        if allow_1h:
            model_dists["+H"] = dist_p1
            model_dists["-H"] = dist_m1
        if allow_2h:
            model_dists["+2H"] = dist_p2
            model_dists["-2H"] = dist_m2

        requested_model = {
            -2: "-2H",
            -1: "-H",
            0: "neutral",
            1: "+H",
            2: "+2H",
        }[int(h_transfer)]
        requested_dist = model_dists.get(requested_model)
        if requested_dist is None:
            variant_result["reason"] = "requested_h_transfer_not_supported"
            variant_result["anchor_theory_mz"] = dist0_theory_mz
            fallback_dist = _scaled_dist_for_display(
                dist0[:, 0], dist0[:, 1], dist0_theory_mz
            )
            variant_result["dist_plot"] = fallback_dist
            variant_result["theory_matches"] = match_theory_peaks(
                spectrum_mz,
                spectrum_int,
                fallback_dist[:, 0],
                theory_int=fallback_dist[:, 1],
                tol_ppm=float(match_tol_ppm),
            )
            variant_result["diagnostic_steps"].append(
                {
                    "step": "4. H-transfer mode",
                    "status": "fail",
                    "details": f"Requested state {requested_model} is not enabled for ion series {series}",
                }
            )
            variant_results.append(variant_result)
            continue

        requested_anchor_theory_mz = float(
            requested_dist[get_anchor_idx(requested_dist), 0]
        )
        variant_result["expected_theory_mz"] = requested_anchor_theory_mz

        sample_keys, sample_mzs, scale = build_sample_axis(
            [requested_dist],
            decimals=6,
            mz_min=mz_min,
            mz_max=mz_max,
        )

        if len(sample_mzs) == 0:
            variant_result["reason"] = "sample_axis_empty"
            variant_result["anchor_theory_mz"] = requested_anchor_theory_mz
            fallback_dist = _scaled_dist_for_display(
                requested_dist[:, 0], requested_dist[:, 1], requested_anchor_theory_mz
            )
            variant_result["dist_plot"] = fallback_dist
            variant_result["theory_matches"] = match_theory_peaks(
                spectrum_mz,
                spectrum_int,
                fallback_dist[:, 0],
                theory_int=fallback_dist[:, 1],
                tol_ppm=float(match_tol_ppm),
            )
            variant_result["diagnostic_steps"].append(
                {
                    "step": "4. H-transfer mode",
                    "status": "fail",
                    "details": "Empty sample axis",
                }
            )
            variant_results.append(variant_result)
            continue

        peak_mz = requested_anchor_theory_mz
        y_obs = observed_intensities_isodec(
            spectrum_mz,
            spectrum_int,
            sample_mzs,
            z=int(z),
            match_tol_ppm=float(match_tol_ppm),
            peak_mz=peak_mz,
        )

        y_requested = vectorize_dist(
            requested_dist, sample_keys, scale, mz_min=mz_min, mz_max=mz_max
        )
        requested_score_union = css_similarity(y_obs, y_requested)
        requested_score = requested_score_union

        requested_dist_windowed = requested_dist
        if mz_min is not None or mz_max is not None:
            mz_min_f = -np.inf if mz_min is None else float(mz_min)
            mz_max_f = np.inf if mz_max is None else float(mz_max)
            requested_dist_windowed = requested_dist[
                (requested_dist[:, 0] >= mz_min_f) & (requested_dist[:, 0] <= mz_max_f)
            ]
        if requested_dist_windowed.size:
            y_obs_windowed = observed_intensities_isodec(
                spectrum_mz,
                spectrum_int,
                requested_dist_windowed[:, 0],
                z=int(z),
                match_tol_ppm=float(match_tol_ppm),
                peak_mz=peak_mz,
            )
            requested_score = css_similarity(
                y_obs_windowed, requested_dist_windowed[:, 1]
            )

        best_model = requested_model
        best_score = requested_score
        best_pred = y_requested
        best_weights = {"0": 0.0, "+H": 0.0, "+2H": 0.0, "-H": 0.0, "-2H": 0.0}
        if requested_model == "neutral":
            best_weights["0"] = 1.0
        else:
            best_weights[requested_model] = 1.0

        if sample_mzs is None or best_pred is None or float(np.max(best_pred)) <= 0.0:
            variant_result["reason"] = "theory_empty"
            variant_result["anchor_theory_mz"] = requested_anchor_theory_mz
            variant_result["diagnostic_steps"].append(
                {
                    "step": "4. H-transfer mode",
                    "status": "fail",
                    "details": "Empty theoretical prediction",
                }
            )
            variant_results.append(variant_result)
            continue

        variant_result["raw_cosine_preanchor"] = float(best_score)
        variant_result["diagnostic_steps"].append(
            {
                "step": "4. H-transfer mode",
                "status": "pass",
                "details": f"Selected model: {best_model}, CSS: {best_score:.4f}",
            }
        )

        variant_result["h_transfer"] = int(h_transfer)

        mono_mass = float(loss_comp.mass())
        avg_mass = float(loss_comp.mass(average=True))
        variant_result["mono_mass"] = mono_mass
        variant_result["avg_mass"] = avg_mass

        anchor_theory_mz = None
        obs_idx = None
        obs_mz = None
        obs_int = None
        anchor_hits = 0

        anchor_window = float(getattr(cfg, "FRAG_ANCHOR_CENTROID_WINDOW_DA", 0.2))
        if str(getattr(cfg, "ANCHOR_MODE", "most_intense")).lower() == "monoisotopic":
            nonzero_idx = np.where(best_pred > 0)[0]
            sorted_idx = nonzero_idx[: int(cfg.ANCHOR_TOP_N)]
        else:
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
            if not within_ppm(obs_mz_c, mz_candidate, float(match_tol_ppm)):
                continue
            if float(min_obs_rel_int) > 0 and obs_int_c < obs_max * float(
                min_obs_rel_int
            ):
                continue
            anchor_hits += 1
            if anchor_theory_mz is None:
                anchor_theory_mz = mz_candidate
                obs_idx = int(obs_idx_c)
                obs_mz = obs_mz_c
                obs_int = obs_int_c

        if str(getattr(cfg, "ANCHOR_MODE", "most_intense")).lower() == "monoisotopic":
            _nz = np.where(best_pred > 0)[0]
            expected_theory_mz = (
                float(sample_mzs[int(_nz[0])]) if len(_nz) > 0 else float(sample_mzs[0])
            )
        else:
            expected_theory_mz = float(sample_mzs[int(np.argmax(best_pred))])
        variant_result["expected_theory_mz"] = expected_theory_mz

        if anchor_hits < int(cfg.ANCHOR_MIN_MATCHES) or anchor_theory_mz is None:
            variant_result["reason"] = "anchor_outside_ppm"
            variant_result["anchor_theory_mz"] = (
                expected_theory_mz  # Still provide theory m/z
            )
            # Build dist_plot for visualization even without anchor match
            dist_plot = _scaled_dist_for_display(
                sample_mzs.copy(), best_pred.copy(), expected_theory_mz
            )
            variant_result["dist_plot"] = dist_plot
            variant_result["theory_matches"] = match_theory_peaks(
                spectrum_mz,
                spectrum_int,
                dist_plot[:, 0],
                theory_int=dist_plot[:, 1],
                tol_ppm=float(match_tol_ppm),
            )
            variant_result["diagnostic_steps"].append(
                {
                    "step": "5. Anchor selection",
                    "status": "fail",
                    "details": f"No anchor peak found within {match_tol_ppm} ppm (expected at {expected_theory_mz:.4f})",
                }
            )
            variant_results.append(variant_result)
            continue

        ppm_offset = (
            (float(obs_mz) - float(anchor_theory_mz)) / float(anchor_theory_mz)
        ) * 1e6
        obs_rel_int = float(obs_int / obs_max) if obs_max > 0 else 0.0

        variant_result["anchor_theory_mz"] = float(anchor_theory_mz)
        variant_result["anchor_obs_mz"] = float(obs_mz)
        variant_result["anchor_ppm"] = float(ppm_offset)
        variant_result["anchor_within_ppm"] = True
        variant_result["obs_idx"] = int(obs_idx)
        variant_result["obs_mz"] = float(obs_mz)
        variant_result["obs_int"] = float(obs_int)
        variant_result["obs_rel_int"] = float(obs_rel_int)
        variant_result["ppm_offset"] = float(ppm_offset)
        variant_result["diagnostic_steps"].append(
            {
                "step": "5. Anchor selection",
                "status": "pass",
                "details": f"Found anchor peak at {obs_mz:.4f} (theory: {anchor_theory_mz:.4f})",
            }
        )
        variant_result["diagnostic_steps"].append(
            {
                "step": "6. PPM offset calculation",
                "status": "pass",
                "details": f"Calculated ppm offset: {ppm_offset:.2f}",
            }
        )

        dist_model = np.column_stack([sample_mzs.copy(), best_pred.copy()])
        dist_plot = dist_model.copy()
        dist_plot[:, 0] += obs_mz - anchor_theory_mz
        max_plot = float(np.max(dist_plot[:, 1]))
        if max_plot <= 0.0:
            variant_result["reason"] = "theory_empty"
            variant_result["diagnostic_steps"].append(
                {
                    "step": "6. PPM offset calculation",
                    "status": "fail",
                    "details": "Empty calibrated spectrum",
                }
            )
            continue
        dist_plot[:, 1] *= obs_int / max_plot

        if mz_min is not None or mz_max is not None:
            mz_min_f = -np.inf if mz_min is None else float(mz_min)
            mz_max_f = np.inf if mz_max is None else float(mz_max)
            dist_plot = dist_plot[
                (dist_plot[:, 0] >= mz_min_f) & (dist_plot[:, 0] <= mz_max_f)
            ]
        if dist_plot.size == 0:
            variant_result["reason"] = "outside_mz_window"
            fallback_dist = _scaled_dist_for_display(
                sample_mzs.copy(), best_pred.copy(), anchor_theory_mz
            )
            variant_result["dist_plot"] = fallback_dist
            variant_result["theory_matches"] = match_theory_peaks(
                spectrum_mz,
                spectrum_int,
                fallback_dist[:, 0],
                theory_int=fallback_dist[:, 1],
                tol_ppm=float(match_tol_ppm),
            )
            variant_result["diagnostic_steps"].append(
                {
                    "step": "6. PPM offset calculation",
                    "status": "fail",
                    "details": f"No peaks in [{mz_min_f:.2f}, {mz_max_f:.2f}] m/z",
                }
            )
            variant_results.append(variant_result)
            continue

        max_plot = float(np.max(dist_plot[:, 1]))
        keep = dist_plot[:, 1] >= max_plot * float(rel_intensity_cutoff)
        dist_plot = dist_plot[keep]
        if dist_plot.size == 0:
            variant_result["reason"] = "below_rel_intensity_cutoff"
            fallback_dist = _scaled_dist_for_display(
                sample_mzs.copy(), best_pred.copy(), anchor_theory_mz
            )
            variant_result["dist_plot"] = fallback_dist
            variant_result["theory_matches"] = match_theory_peaks(
                spectrum_mz,
                spectrum_int,
                fallback_dist[:, 0],
                theory_int=fallback_dist[:, 1],
                tol_ppm=float(match_tol_ppm),
            )
            variant_result["diagnostic_steps"].append(
                {
                    "step": "6. PPM offset calculation",
                    "status": "fail",
                    "details": f"No peaks above {rel_intensity_cutoff * 100:.1f}% relative intensity",
                }
            )
            variant_results.append(variant_result)
            continue

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
            variant_result["isodec_css"] = float(isodec_css)
            variant_result["isodec_accepted"] = bool(accepted)
            variant_result["diagnostic_steps"].append(
                {
                    "step": "7. IsoDec rules",
                    "status": "pass" if accepted else "fail",
                    "details": f"CSS={isodec_css:.4f}",
                }
            )

            if (
                cfg.isodec_find_matches is not None
                and local_centroids.size
                and dist_plot.size
            ):
                matchedindexes, isomatches = cfg.isodec_find_matches(
                    local_centroids, dist_plot, float(isodec_config.matchtol)
                )
                mi = (
                    np.array(matchedindexes, dtype=int)
                    if len(matchedindexes)
                    else np.array([], dtype=int)
                )
                ii = (
                    np.array(isomatches, dtype=int)
                    if len(isomatches)
                    else np.array([], dtype=int)
                )
                matchedcentroids = (
                    local_centroids[mi] if mi.size else np.empty((0, 2), dtype=float)
                )
                matchediso = dist_plot[ii] if ii.size else np.empty((0, 2), dtype=float)

                minpeaks_eff = int(isodec_config.minpeaks)
                if int(z) == 1 and len(matchedindexes) == 2 and len(isomatches) == 2:
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

                areacovered = (
                    float(np.sum(matchediso[:, 1]))
                    / float(np.sum(local_centroids[:, 1]))
                    if local_centroids.size and np.sum(local_centroids[:, 1]) > 0
                    else 0.0
                )
                topn = minpeaks_eff
                topthree = False
                if local_centroids.size and matchedcentroids.size and topn > 0:
                    top_iso = np.sort(matchedcentroids[:, 1])[::-1][:topn]
                    top_cent = np.sort(local_centroids[:, 1])[::-1][:topn]
                    topthree = bool(np.array_equal(top_iso, top_cent))

                variant_result["isodec_detail"] = {
                    "local_centroids_n": int(local_centroids.shape[0]),
                    "matched_peaks_n": int(len(matchedindexes)),
                    "minpeaks_effective": int(minpeaks_eff),
                    "css_thresh": float(isodec_config.css_thresh),
                    "minareacovered": float(isodec_config.minareacovered),
                    "areacovered": float(areacovered),
                    "topthree": bool(topthree),
                }

            if not accepted:
                # Diagnose-only high-confidence override for borderline strict-rule failures.
                override_enabled = bool(
                    getattr(cfg, "DIAGNOSE_ISODEC_OVERRIDE_ENABLE", False)
                )
                override_min_css = float(
                    getattr(cfg, "DIAGNOSE_ISODEC_OVERRIDE_MIN_CSS", 0.95)
                )
                override_min_matched = int(
                    getattr(cfg, "DIAGNOSE_ISODEC_OVERRIDE_MIN_MATCHED_PEAKS", 2)
                )
                override_max_abs_ppm_cfg = getattr(
                    cfg, "DIAGNOSE_ISODEC_OVERRIDE_MAX_ABS_PPM", None
                )
                if override_max_abs_ppm_cfg is None:
                    override_max_abs_ppm = float(match_tol_ppm)
                else:
                    override_max_abs_ppm = float(override_max_abs_ppm_cfg)
                matched_n = int(
                    variant_result.get("isodec_detail", {}).get("matched_peaks_n", 0)
                )
                anchor_ppm_abs = abs(
                    float(variant_result.get("anchor_ppm", 0.0) or 0.0)
                )
                if (
                    override_enabled
                    and float(isodec_css) >= override_min_css
                    and matched_n >= override_min_matched
                    and anchor_ppm_abs <= override_max_abs_ppm
                ):
                    accepted = True
                    variant_result["diagnostic_steps"].append(
                        {
                            "step": "7b. Diagnose override",
                            "status": "pass",
                            "details": (
                                f"Override accepted: CSS={isodec_css:.4f}, "
                                f"matched={matched_n}, |ppm|={anchor_ppm_abs:.2f}"
                            ),
                        }
                    )
                else:
                    variant_result["reason"] = "failed_isodec_rules"
                    variant_result["dist_plot"] = dist_plot
                    variant_result["theory_matches"] = match_theory_peaks(
                        spectrum_mz,
                        spectrum_int,
                        dist_plot[:, 0],
                        theory_int=dist_plot[:, 1],
                        tol_ppm=float(match_tol_ppm),
                    )
                    fragments_gate = {
                        "legacy_accepted": False,
                        "failed_at": "isodec_rules",
                        "checks": {
                            "isodec_css": {
                                "value": float(
                                    variant_result.get("isodec_css", best_score)
                                ),
                                "threshold": (
                                    f">= {float(getattr(cfg, 'FRAG_MIN_ISODEC_CSS', cfg.MIN_COSINE)):.6f}"
                                ),
                                "threshold_value": float(
                                    getattr(cfg, "FRAG_MIN_ISODEC_CSS", cfg.MIN_COSINE)
                                ),
                                "pass": float(
                                    variant_result.get("isodec_css", best_score)
                                )
                                >= float(
                                    getattr(cfg, "FRAG_MIN_ISODEC_CSS", cfg.MIN_COSINE)
                                ),
                                "description": "Cosine similarity score",
                            },
                            "isodec_rules": {
                                "value": float(
                                    variant_result.get("isodec_detail", {}).get(
                                        "matched_peaks_n", 0
                                    )
                                ),
                                "threshold": "strict IsoDec acceptance",
                                "threshold_value": 1.0,
                                "pass": False,
                                "description": "IsoDec structural rules",
                            },
                        },
                        "quality_metrics": {
                            "fit_score": 0.0,
                            "correlation_coefficient": 0.0,
                            "pc_missing_peaks": 100.0,
                            "local_matches_count": int(
                                variant_result.get("isodec_detail", {}).get(
                                    "matched_peaks_n", 0
                                )
                            ),
                        },
                    }
                    variant_result["fragments_gate"] = fragments_gate
                    variant_results.append(variant_result)
                    continue

            if shifted_peak is not None:
                old_obs_mz = obs_mz
                obs_mz_new = float(shifted_peak)
                obs_idx = nearest_peak_index(spectrum_mz, obs_mz_new)
                obs_mz = float(spectrum_mz[obs_idx])
                obs_int = float(spectrum_int[obs_idx])
                obs_rel_int = float(obs_int / obs_max) if obs_max > 0 else 0.0
                ppm_offset = (
                    (float(obs_mz) - float(anchor_theory_mz)) / float(anchor_theory_mz)
                ) * 1e6
                dist_plot[:, 0] += obs_mz - old_obs_mz
                variant_result["anchor_obs_mz"] = float(obs_mz)
                variant_result["anchor_ppm"] = float(ppm_offset)
                variant_result["obs_idx"] = int(obs_idx)
                variant_result["obs_mz"] = float(obs_mz)
                variant_result["obs_int"] = float(obs_int)
                variant_result["obs_rel_int"] = float(obs_rel_int)
                variant_result["ppm_offset"] = float(ppm_offset)
        else:
            variant_result["isodec_css"] = float(isodec_css)
            variant_result["isodec_accepted"] = False
            variant_result["diagnostic_steps"].append(
                {
                    "step": "7. IsoDec rules",
                    "status": "info",
                    "details": "IsoDec rules disabled",
                }
            )

        variant_result["dist_plot"] = dist_plot
        variant_result["theory_matches"] = match_theory_peaks(
            spectrum_mz,
            spectrum_int,
            dist_plot[:, 0],
            theory_int=dist_plot[:, 1],
            tol_ppm=float(match_tol_ppm),
        )

        fragments_pipeline = None
        pipeline_allowed = (
            fragments_pipeline_eval is not None
            and fragments_scoring_settings is not None
            and obs_mz is not None
            and anchor_theory_mz is not None
            and (
                bool(variant_result.get("isodec_accepted", False))
                or not bool(cfg.ENABLE_ISODEC_RULES)
                or isodec_config is None
                or bool(fragments_scoring_settings.get("truth_score_enabled"))
            )
        )
        if pipeline_allowed:
            fragments_pipeline = fragments_pipeline_eval(
                spectrum_mz=np.asarray(spectrum_mz, dtype=float),
                spectrum_int=np.asarray(spectrum_int, dtype=float),
                obs_max=float(obs_max),
                match_tol_ppm=float(match_tol_ppm),
                dist_model=np.asarray(dist_model, dtype=float),
                dist_plot=np.asarray(dist_plot, dtype=float),
                obs_mz=float(obs_mz),
                obs_int=float(obs_int),
                anchor_theory_mz=float(anchor_theory_mz),
                ppm=float(variant_result.get("anchor_ppm", 0.0) or 0.0),
                isodec_css=float(variant_result.get("isodec_css", best_score)),
                isodec_detail=variant_result.get("isodec_detail") or {},
                isodec_config=isodec_config,
                use_centroid_logic=bool(use_centroid_logic),
                noise_model=fragments_noise_model,
                settings=fragments_scoring_settings,
            )

        if fragments_pipeline and fragments_pipeline.get("pipeline_ready"):
            fragments_gate = fragments_pipeline["fragments_gate"]
            variant_result["fragments_quality"] = fragments_pipeline.get("quality")
            variant_result["fragments_components"] = fragments_pipeline.get("comp")
        else:
            fragments_gate = _calculate_fragments_gate(
                isodec_css=float(variant_result.get("isodec_css", best_score)),
                ppm=float(variant_result.get("anchor_ppm", 0.0) or 0.0),
                theory_matches=variant_result["theory_matches"],
                dist_plot=dist_plot,
                spectrum_mz=spectrum_mz,
                spectrum_int=spectrum_int,
                local_centroids=None,
                anchor_mz=float(obs_mz) if obs_mz is not None else 0.0,
            )
        variant_result["fragments_gate"] = fragments_gate

        gate_details_lines = []
        for check_name, check_info in fragments_gate["checks"].items():
            status_icon = "✓" if check_info["pass"] else "✗"
            gate_details_lines.append(
                f"{status_icon} {check_info['description']}: "
                f"{check_info['value']:.4f} {check_info['threshold']}"
            )
        gate_details = "; ".join(gate_details_lines)

        if fragments_pipeline and fragments_pipeline.get("pipeline_ready"):
            comp = fragments_pipeline.get("comp") or {}
            quality = fragments_pipeline.get("quality") or {}
            gate_details += (
                f" | coverage={float(comp.get('coverage', 0.0)):.3f}"
                f"; unexplained={float(quality.get('unexplained_fraction', 0.0)):.3f}"
                f"; evidence={float(quality.get('evidence_score', 0.0)):.3f}"
            )

        corr_value = fragments_gate["checks"].get(
            "correlation_coefficient", {}
        ).get("value", 0.0)
        corr_threshold = fragments_gate["checks"].get(
            "correlation_coefficient", {}
        ).get("threshold_value", 0.0)
        if corr_threshold > 0 and abs(corr_value - corr_threshold) < 0.10:
            gate_details += (
                f" [Correlation {corr_value:.4f} is near threshold "
                f"{corr_threshold:.4f}]"
            )

        variant_result["diagnostic_steps"].append(
            {
                "step": "8b. Fragments gate (shared fragments pipeline)",
                "status": "pass" if fragments_gate["legacy_accepted"] else "fail",
                "details": gate_details,
            }
        )

        final_score = float(variant_result.get("isodec_css", best_score))
        variant_result["final_cosine"] = final_score
        variant_result["diagnostic_steps"].append(
            {
                "step": "8. Final CSS calculation",
                "status": "pass",
                "details": f"Final CSS score: {final_score:.4f}",
            }
        )

        if final_score >= float(cfg.MIN_COSINE):
            variant_result["ok"] = True
            if variant_result.get("reason") in ("failed_isodec_rules", ""):
                variant_result["reason"] = "ok"
            if any(
                step.get("step") == "7b. Diagnose override"
                for step in variant_result.get("diagnostic_steps", [])
            ):
                variant_result["reason"] = "ok_diagnose_override"
            variant_result["diagnostic_steps"].append(
                {
                    "step": "9. Match acceptance",
                    "status": "pass",
                    "details": f"Match accepted with CSS score: {final_score:.4f}",
                }
            )
        else:
            variant_result["ok"] = False
            variant_result["reason"] = (
                f"cosine_below_threshold: {final_score:.4f} < {cfg.MIN_COSINE}"
            )
            variant_result["diagnostic_steps"].append(
                {
                    "step": "9. Match acceptance",
                    "status": "fail",
                    "details": "Match rejected: CSS score below threshold",
                }
            )

        variant_results.append(variant_result)

    if not variant_results:
        result["reason"] = "all_disulfide_variants_failed"
        result["diagnostic_steps"].append(
            {
                "step": "1. Ion composition",
                "status": "fail",
                "details": "All disulfide variants failed",
            }
        )
        return result

    pass_count = sum(1 for r in variant_results if r.get("ok"))
    ranked_variants = sorted(
        variant_results, key=variant_rank_key_from_result, reverse=True
    )
    best_result = ranked_variants[0]
    best_result["variant_pass_count"] = int(pass_count)
    # Preserve all evaluated disulfide variants so diagnose mode can display them in parallel.
    best_result["all_variants"] = ranked_variants
    return best_result


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
