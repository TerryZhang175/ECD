from __future__ import annotations

import math
from typing import Optional


def variant_type_from_suffix(suffix: str) -> str:
    s = str(suffix or "")
    if not s:
        return "nonmodified"
    if "OxidizedLoop" in s:
        return "ss-specific"
    return "cys-specific"


def _as_float(value: Optional[object], default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


def variant_rank_key_from_result(result: dict) -> tuple[float, float, float]:
    css = result.get("final_cosine")
    if css is None:
        css = result.get("score")
    if css is None:
        css = result.get("raw_cosine_preanchor")
    css_val = _as_float(css, float("-inf"))

    obs_int = _as_float(result.get("obs_int", 0.0), 0.0)

    ppm = result.get("ppm")
    if ppm is None:
        ppm = result.get("anchor_ppm")
    if ppm is None:
        ppm = result.get("ppm_offset")
    ppm_val = _as_float(ppm, float("inf"))
    abs_ppm = abs(ppm_val) if math.isfinite(ppm_val) else float("inf")

    # Higher CSS, higher intensity, lower abs(ppm) are better.
    return (css_val, obs_int, -abs_ppm)
