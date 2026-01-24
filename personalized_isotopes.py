from __future__ import annotations

import numpy as np

ISOLEN_DEFAULT = 128

_FFT_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def _get_fft_tables(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = int(length)
    cached = _FFT_CACHE.get(length)
    if cached is not None:
        return cached

    buffer = np.zeros(length, dtype=float)
    h = np.array([1.0, 0.00015, 0.0, 0.0], dtype=float)
    c = np.array([1.0, 0.011, 0.0, 0.0], dtype=float)
    n = np.array([1.0, 0.0037, 0.0, 0.0], dtype=float)
    o = np.array([1.0, 0.0004, 0.002, 0.0], dtype=float)
    s = np.array([1.0, 0.0079, 0.044, 0.0], dtype=float)

    h = np.append(h, buffer)
    c = np.append(c, buffer)
    n = np.append(n, buffer)
    o = np.append(o, buffer)
    s = np.append(s, buffer)

    hft = np.fft.rfft(h).astype(np.complex128)
    cft = np.fft.rfft(c).astype(np.complex128)
    nft = np.fft.rfft(n).astype(np.complex128)
    oft = np.fft.rfft(o).astype(np.complex128)
    sft = np.fft.rfft(s).astype(np.complex128)

    _FFT_CACHE[length] = (cft, hft, nft, oft, sft)
    return _FFT_CACHE[length]


def isojim(isolist: np.ndarray, length: int = ISOLEN_DEFAULT) -> np.ndarray:
    """
    Compute isotopic distribution using FFT-based convolution (C/H/N/O/S only).
    """
    isolist = np.asarray(isolist, dtype=int).ravel()
    if isolist.size < 5:
        isolist = np.pad(isolist, (0, 5 - isolist.size), mode="constant")
    numc, numh, numn, numo, nums = (int(isolist[i]) for i in range(5))

    cft, hft, nft, oft, sft = _get_fft_tables(length)
    allft = cft ** numc * hft ** numh * nft ** numn * oft ** numo * sft ** nums
    allift = np.abs(np.fft.irfft(allft))
    maxval = float(np.max(allift)) if allift.size else 0.0
    if maxval > 0:
        allift = allift / maxval
    return allift[: int(length)]
