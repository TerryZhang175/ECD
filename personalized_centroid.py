from __future__ import annotations

import numpy as np

import personalized_config as cfg

try:
    from unidec.tools import peakdetect as _peakdetect
except Exception:
    try:
        from personalized_peakdetect import peakdetect as _peakdetect
    except Exception:
        _peakdetect = None


def hill_centroid_window(
    spectrum_mz: np.ndarray,
    spectrum_int: np.ndarray,
    center_mz: float,
    lb: float,
    ub: float,
    *,
    ppm: float | None = None,
    window: int | None = None,
    threshold: float | None = None,
    norm: bool | None = None,
) -> np.ndarray:
    spectrum_mz = np.asarray(spectrum_mz, dtype=float)
    spectrum_int = np.asarray(spectrum_int, dtype=float)
    if spectrum_mz.ndim != 1 or spectrum_int.ndim != 1 or spectrum_mz.size == 0:
        return np.empty((0, 2), dtype=float)

    start = int(np.searchsorted(spectrum_mz, center_mz + float(lb), side="left"))
    end = int(np.searchsorted(spectrum_mz, center_mz + float(ub), side="right"))
    if end <= start:
        return np.empty((0, 2), dtype=float)

    windowed = np.column_stack([spectrum_mz[start:end], spectrum_int[start:end]])
    if not bool(getattr(cfg, "ENABLE_CENTROID", True)) or not bool(cfg.ENABLE_HILL_CENTROID) or _peakdetect is None:
        return windowed

    ppm_val = ppm
    if ppm_val is None:
        ppm_val = cfg.HILL_CENTROID_PPM
    ppm_val = float(ppm_val) if ppm_val is not None else float(cfg.MATCH_TOL_PPM)

    window_val = window
    if window_val is None:
        window_val = cfg.HILL_CENTROID_WINDOW
    window_val = int(window_val) if window_val is not None else 10

    threshold_val = threshold
    if threshold_val is None:
        threshold_val = cfg.HILL_CENTROID_THRESHOLD
    threshold_val = float(threshold_val) if threshold_val is not None else 0.0

    norm_val = norm
    if norm_val is None:
        norm_val = cfg.HILL_CENTROID_NORM
    norm_val = bool(norm_val)

    try:
        peaks = _peakdetect(windowed, window=window_val, threshold=threshold_val, ppm=ppm_val, norm=norm_val)
    except ValueError:
        peaks = _peakdetect(windowed, window=window_val, threshold=threshold_val, ppm=None, norm=norm_val)

    if isinstance(peaks, np.ndarray) and peaks.size:
        if len(peaks) > 1:
            peaks = peaks[np.argsort(peaks[:, 0])]
        return peaks

    return windowed
