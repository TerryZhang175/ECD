from __future__ import annotations

from bisect import bisect_left

import numpy as np


def nearest(array: np.ndarray, target: float) -> int:
    """
    In a sorted array, find the index of the element closest to target.
    """
    i = bisect_left(array, target)
    if i <= 0:
        return 0
    if i >= len(array) - 1:
        return len(array) - 1
    if abs(array[i] - target) > abs(array[i - 1] - target):
        i -= 1
    return int(i)


def peakdetect(data: np.ndarray, config=None, window: int = 10, threshold: float = 0.0,
               ppm: float | None = None, norm: bool = True) -> np.ndarray:
    """
    Simple peak detection. Mirrors UniDec's behavior for local maxima detection.
    """
    if config is not None:
        window = int(getattr(config, "peakwindow", window))
        threshold = float(getattr(config, "peakthresh", threshold))
        norm = bool(getattr(config, "normthresh", norm))

    peaks = []
    if data is None:
        return np.asarray(peaks, dtype=float)
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or len(data) == 0:
        return np.asarray(peaks, dtype=float)

    maxval = float(np.max(data[:, 1])) if norm else 1.0
    length = len(data)

    for i in range(length):
        if data[i, 1] <= maxval * threshold:
            continue
        if ppm is not None:
            ptmass = float(data[i, 0])
            newwin = float(ppm) * 1e-6 * ptmass
            start = nearest(data[:, 0], ptmass - newwin)
            end = nearest(data[:, 0], ptmass + newwin)
        else:
            start = int(i - window)
            end = int(i + window) + 1
            if start < 0:
                start = 0
            if end > length:
                end = length

        testmax = float(np.max(data[start:end, 1]))
        if data[i, 1] == testmax and np.all(data[i, 1] != data[start:i, 1]):
            peaks.append([data[i, 0], data[i, 1]])

    return np.asarray(peaks, dtype=float)
