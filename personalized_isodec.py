from __future__ import annotations

from typing import List, Tuple
import math

import numpy as np

MASS_DIFF_C = 1.0033

try:
    from numba import njit as _njit
except Exception:  # pragma: no cover - optional acceleration
    def _njit(*_args, **_kwargs):
        def _wrap(func):
            return func
        return _wrap


class IsoDecConfig:
    """
    Minimal IsoDec configuration used by the personalized pipeline.
    """

    def __init__(self):
        self.adductmass = 1.007276467
        self.mass_diff_c = MASS_DIFF_C
        self.verbose = False

        self.matchtol = 5
        self.minpeaks = 3
        self.css_thresh = 0.7
        self.maxshift = 3
        self.minareacovered = 0.20
        self.minusoneaszero = True
        self.isotopethreshold = 0.01

        self.mzwindow = [-1.05, 2.05]
        self.plusoneintwindow = [0.1, 0.6]
        self.mzwindowlb = self.mzwindow[0]
        self.mzwindowub = self.mzwindow[1]
        self.plusoneintwindowlb = self.plusoneintwindow[0]
        self.plusoneintwindowub = self.plusoneintwindow[1]


@_njit(fastmath=True)
def calculate_cosinesimilarity(
    cent_intensities: np.ndarray,
    iso_intensities: np.ndarray,
    shift: int,
    max_shift: int,
    minusoneareaszero: bool = True,
) -> float:
    ab = 0.0
    a2 = 0.0
    b2 = 0.0

    if minusoneareaszero:
        a_val = cent_intensities[max_shift + shift - 1]
        b_val = 0.0
        ab += a_val * b_val
        a2 += a_val ** 2
        b2 += b_val ** 2

    for i in range(len(iso_intensities)):
        a_val = cent_intensities[i + max_shift + shift]
        b_val = iso_intensities[i]
        ab += a_val * b_val
        a2 += a_val ** 2
        b2 += b_val ** 2

    if ab == 0 or a2 == 0 or b2 == 0:
        return 0.0
    return float(ab / (math.sqrt(a2) * math.sqrt(b2)))


@_njit(fastmath=True)
def find_matches(spec1: np.ndarray, spec2: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match peak indices between two spectra within ppm tolerance.
    """
    if len(spec1) == 0 or len(spec2) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    lowest_idx = 0
    m1 = np.empty(spec2.shape[0], dtype=np.int64)
    m2 = np.empty(spec2.shape[0], dtype=np.int64)
    count = 0
    diff = float(spec1[0, 0]) * float(tolerance) * 1e-6

    for iso_idx in range(spec2.shape[0]):
        mz = float(spec2[iso_idx, 0])
        low_bound = mz - diff
        high_bound = mz + diff

        topint = 0.0
        topindex = -1

        for peak_idx in range(lowest_idx, spec1.shape[0]):
            mz2 = float(spec1[peak_idx, 0])
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak_idx + 1
            else:
                newint = float(spec1[peak_idx, 1])
                if newint > topint:
                    topint = newint
                    topindex = peak_idx

        if topindex != -1:
            m1[count] = int(topindex)
            m2[count] = int(iso_idx)
            count += 1

    return m1[:count], m2[:count]


@_njit(fastmath=True)
def find_matched_intensities(
    spec1_mz: np.ndarray,
    spec1_intensity: np.ndarray,
    spec2_mz: np.ndarray,
    max_shift: int,
    tolerance: float,
    z: int,
    peakmz: float,
) -> np.ndarray:
    """
    Return matched centroid intensities for theoretical isotope m/z values.
    """
    query_len = len(spec2_mz) + 2 * max_shift
    query_mzs = np.zeros(query_len, dtype=np.float64)
    cent_intensities = np.zeros(query_len, dtype=np.float64)

    mono_idx = max_shift
    for i in range(max_shift + 1):
        if i == 0:
            continue
        query_mzs[max_shift + len(spec2_mz) - 1 + i] = spec2_mz[len(spec2_mz) - 1] + (i * MASS_DIFF_C) / z
        query_mzs[mono_idx - i] = spec2_mz[0] - (i * MASS_DIFF_C) / z

    for i in range(len(spec2_mz)):
        query_mzs[i + max_shift] = spec2_mz[i]

    diff = float(peakmz) * float(tolerance) * 1e-6
    for i in range(len(query_mzs)):
        mz = float(query_mzs[i])
        low_bound = mz - diff
        high_bound = mz + diff
        for j in range(len(spec1_mz)):
            mz2 = float(spec1_mz[j])
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                continue
            value = float(spec1_intensity[j])
            if value > cent_intensities[i]:
                cent_intensities[i] = value

    return cent_intensities


def make_shifted_peak(
    shift: int,
    shiftscore: float,
    monoiso: float,
    massdist: np.ndarray,
    isodist: np.ndarray,
    peakmz: float,
    z: int,
    centroids: np.ndarray,
    matchtol: float,
    minpeaks: int,
    p1low: float,
    p1high: float,
    css_thresh: float,
    minareacovered: float,
    verbose: bool = True,
):
    b1 = isodist[:, 1] > 0
    shiftmass = float(shift) * MASS_DIFF_C
    monoiso_new = float(monoiso) + shiftmass
    massdist_new = massdist[b1].copy()
    massdist_new[:, 0] = massdist_new[:, 0] + shiftmass
    shiftmz = shiftmass / float(z)
    isodist_new = isodist[b1].copy()
    isodist_new[:, 0] = isodist_new[:, 0] + shiftmz
    peakmz_new = float(peakmz)

    matchedindexes, isomatches = find_matches(centroids, isodist_new, matchtol)
    matchediso = (
        isodist_new[np.array(isomatches, dtype=int)]
        if len(isomatches)
        else np.empty((0, 2))
    )

    if z == 1 and len(matchedindexes) == 2:
        if isomatches[0] == 0 and isomatches[1] == 1:
            int1 = centroids[matchedindexes[0], 1]
            int2 = centroids[matchedindexes[1], 1]
            ratio = 0.0 if int1 == 0 else (int2 / int1)
            if p1low < ratio < p1high:
                minpeaks = 2

    areacovered = float(np.sum(matchediso[:, 1]) / np.sum(centroids[:, 1])) if len(centroids) else 0.0
    matchedcentroids = (
        centroids[np.array(matchedindexes, dtype=int)]
        if len(matchedindexes)
        else np.empty((0, 2))
    )
    topthreeiso = np.sort(matchedcentroids[:, 1])[::-1][:minpeaks] if len(matchedcentroids) else np.array([])
    topthreecent = np.sort(centroids[:, 1])[::-1][:minpeaks] if len(centroids) else np.array([])
    topthree = bool(len(topthreeiso) and np.array_equal(topthreeiso, topthreecent))

    if verbose:
        print("Matched Peaks:", len(matchedindexes), "Shift Score:", shiftscore, "Area Covered:",
              areacovered, "Top Three:", topthree)
    if len(matchedindexes) >= minpeaks and (
        shiftscore >= css_thresh
    ) and (areacovered > minareacovered or topthree):
        return peakmz_new, isodist_new, matchedindexes, isomatches, monoiso_new, massdist_new

    if verbose:
        print("Failed Peak:", len(matchedindexes), shiftscore, areacovered, topthree)
        if len(matchedindexes) < minpeaks:
            print("Failed Min Peaks")
        if not shiftscore >= css_thresh:
            print("Failed CSS")
        if not (areacovered > minareacovered or topthree):
            if areacovered < minareacovered:
                print("Failed Min Area Covered")
            if not topthree:
                print("Failed Top Three")

    return None, None, None, None, None, None
