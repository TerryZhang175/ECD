from __future__ import annotations

from typing import Optional

import numpy as np
import pyteomics.mass as ms

try:
    from unidec.modules.isotopetools import isojim
except Exception:
    from personalized_isotopes import isojim

import personalized_config as cfg


def theoretical_isodist_from_comp(
    comp: ms.Composition,
    charge: int,
    proton_count: Optional[int] = None,
) -> np.ndarray:
    mono_mass = float(comp.mass())
    isolist = np.array(
        [
            comp.get("C", 0),
            comp.get("H", 0),
            comp.get("N", 0),
            comp.get("O", 0),
            comp.get("S", 0),
            comp.get("Fe", 0),
            comp.get("K", 0),
            comp.get("Ca", 0),
            comp.get("Ni", 0),
            comp.get("Zn", 0),
            comp.get("Mg", 0),
        ],
        dtype=int,
    )
    if np.any(isolist < 0):
        raise ValueError(f"Computed a negative elemental composition: {isolist}")

    intensities = np.asarray(isojim(isolist, length=cfg.ISOLEN), dtype=float)
    isotope_index = np.arange(len(intensities), dtype=float)
    masses = mono_mass + isotope_index * cfg.MASS_DIFF_C
    if int(charge) == 0:
        raise ValueError("charge must be non-zero")
    pcount = int(charge) if proton_count is None else int(proton_count)
    mz = (masses + (pcount * cfg.ADDUCT_MASS)) / abs(int(charge))
    dist = np.column_stack([mz, intensities])

    max_int = float(np.max(dist[:, 1]))
    dist = dist[dist[:, 1] >= max_int * cfg.REL_INTENSITY_CUTOFF].copy()
    return dist


def css_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if cfg.isodec_calculate_cosinesimilarity is None:
        raise ImportError("IsoDec cosine similarity is not available; install UniDec IsoDec deps.")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(cfg.isodec_calculate_cosinesimilarity(a, b, 0, 0, False))


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Project vector v onto the probability simplex: w>=0 and sum(w)=1.
    Reference: Duchi et al. (2008).
    """
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, v.size + 1) > (cssv - 1))[0]
    if rho.size == 0:
        return np.full_like(v, 1.0 / v.size)
    rho = int(rho[-1])
    theta = (cssv[rho] - 1) / float(rho + 1)
    w = v - theta
    w[w < 0] = 0.0
    s = float(np.sum(w))
    if s == 0.0:
        return np.full_like(v, 1.0 / v.size)
    return w / s


def fit_simplex_mixture(y_obs: np.ndarray, components: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    if len(components) == 0:
        return np.empty(0, dtype=float), np.zeros_like(y_obs), 0.0
    A = np.column_stack(components)
    w_ls, *_ = np.linalg.lstsq(A, y_obs, rcond=None)
    w = project_to_simplex(w_ls)
    y_pred = A @ w
    score = css_similarity(y_obs, y_pred)
    return w, y_pred, score


def build_sample_axis(
    dists: list[np.ndarray],
    decimals: int = 6,
    mz_min=None,
    mz_max=None,
) -> tuple[np.ndarray, np.ndarray, float]:
    keys_all = []
    scale = float(10 ** int(decimals))
    for dist in dists:
        if dist is None or len(dist) == 0:
            continue
        mz = dist[:, 0]
        if mz_min is not None:
            mz = mz[mz >= float(mz_min)]
        if mz_max is not None:
            mz = mz[mz <= float(mz_max)]
        if mz.size == 0:
            continue
        keys = np.rint(mz * scale).astype(np.int64)
        keys_all.append(keys)

    if len(keys_all) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float), scale

    keys_union = np.unique(np.concatenate(keys_all))
    mzs_union = keys_union.astype(float) / scale
    return keys_union, mzs_union, scale


def vectorize_dist(
    dist: np.ndarray,
    sample_keys: np.ndarray,
    scale: float,
    mz_min=None,
    mz_max=None,
) -> np.ndarray:
    y = np.zeros(len(sample_keys), dtype=float)
    if dist is None or len(dist) == 0:
        return y
    mz = dist[:, 0]
    inten = dist[:, 1]
    if mz_min is not None:
        keep = mz >= float(mz_min)
        mz = mz[keep]
        inten = inten[keep]
    if mz_max is not None:
        keep = mz <= float(mz_max)
        mz = mz[keep]
        inten = inten[keep]
    if mz.size == 0:
        return y

    keys = np.rint(mz * scale).astype(np.int64)
    idx = np.searchsorted(sample_keys, keys)
    good = (idx >= 0) & (idx < len(sample_keys)) & (sample_keys[idx] == keys)
    if np.any(good):
        np.add.at(y, idx[good], inten[good])
    return y


def sample_observed_intensities(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    target_mzs: np.ndarray,
    tol_ppm: float,
    nearest_index,
    within_ppm,
) -> np.ndarray:
    out = np.zeros(len(target_mzs), dtype=float)
    if len(spectrum_mz) == 0 or len(target_mzs) == 0:
        return out
    for i, mz in enumerate(target_mzs):
        idx = nearest_index(spectrum_mz, float(mz))
        mz_obs = float(spectrum_mz[idx])
        if within_ppm(mz_obs, float(mz), tol_ppm):
            out[i] = float(spectrum_int[idx])
    return out


def observed_intensities_isodec(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    theory_mz: np.ndarray,
    z: int,
    match_tol_ppm: float,
    peak_mz: float,
) -> np.ndarray:
    if cfg.isodec_find_matched_intensities is None:
        raise ImportError("IsoDec matched-intensity function is not available; install UniDec IsoDec deps.")
    return cfg.isodec_find_matched_intensities(
        spectrum_mz,
        spectrum_int,
        theory_mz,
        0,
        tolerance=float(match_tol_ppm),
        z=int(z),
        peakmz=float(peak_mz),
    )
