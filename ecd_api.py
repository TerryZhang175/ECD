from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import logging
from typing import Any, Iterable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np

import personalized_config as cfg
from personalized import load_spectrum, preprocess_spectrum
from personalized_modes import run_charge_reduced_headless, run_diagnose_headless, run_fragments_headless, run_precursor_headless, run_raw_headless
from personalized_sequence import parse_custom_sequence


app = FastAPI(title="ECD Analyzer API", version="0.1.0")
_UI_DIR = Path(__file__).parent / "ui"
if _UI_DIR.exists():
    # Serve the UI from the same origin as the API (http://127.0.0.1:8001/ui/).
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecd_api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


def _normalize_ion_type(value: Optional[str]) -> Optional[str]:
    ion = (value or "").lower()
    if ion.startswith("b"):
        return "b"
    if ion.startswith("c"):
        return "c"
    if ion.startswith("y"):
        return "y"
    if ion.startswith("z"):
        return "z"
    return None


def _parse_disulfide_map(value: Optional[str]) -> list[tuple[int, int]]:
    if not value:
        return []
    pairs: list[tuple[int, int]] = []
    for left, right in re.findall(r"(\d+)\s*-\s*(\d+)", value):
        a = int(left)
        b = int(right)
        if a > 0 and b > 0:
            pairs.append((a, b))
    return pairs


def _disulfide_map_to_string(value: Any) -> str:
    if not value:
        return ""
    parts: list[str] = []
    for item in value:
        try:
            a, b = item
            parts.append(f"{int(a)}-{int(b)}")
        except Exception:
            continue
    return ", ".join(parts)


def _safe_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f


def _fragment_index(seq_len: int, ion_type: str, frag_len: int) -> Optional[int]:
    if frag_len <= 0 or seq_len <= 1:
        return None
    if ion_type in {"b", "c"}:
        idx = frag_len - 1
    elif ion_type in {"y", "z"}:
        idx = seq_len - frag_len - 1
    else:
        return None
    return idx if 0 <= idx < seq_len - 1 else None


def _downsample_xy(mz: np.ndarray, intensity: np.ndarray, max_points: int = 50000) -> tuple[list[float], list[float]]:
    if mz.size <= max_points:
        return mz.astype(float).tolist(), intensity.astype(float).tolist()
    stride = int(np.ceil(mz.size / max_points))
    return mz[::stride].astype(float).tolist(), intensity[::stride].astype(float).tolist()


def _extract_theory_peaks(best: list[dict[str, Any]], max_per_fragment: int = 8) -> tuple[list[float], list[float]]:
    peaks: dict[float, float] = {}
    for m in best:
        dist = m.get("dist")
        if not isinstance(dist, np.ndarray) or dist.size == 0:
            anchor_mz = _safe_float(m.get("anchor_theory_mz"))
            anchor_int = _safe_float(m.get("obs_int")) or _safe_float(m.get("score")) or 0.0
            if anchor_mz is not None:
                key = round(anchor_mz, 4)
                peaks[key] = max(peaks.get(key, 0.0), float(anchor_int))
            continue
        order = np.argsort(dist[:, 1])[::-1][:max_per_fragment]
        for idx in order:
            mz_val = float(dist[idx, 0])
            int_val = float(dist[idx, 1])
            key = round(mz_val, 4)
            peaks[key] = max(peaks.get(key, 0.0), int_val)
    mz_sorted = sorted(peaks.keys())
    return mz_sorted, [peaks[mz] for mz in mz_sorted]


def _theory_from_dist(dist: Any) -> tuple[list[float], list[float]]:
    if not isinstance(dist, np.ndarray) or dist.size == 0:
        return [], []
    return dist[:, 0].astype(float).tolist(), dist[:, 1].astype(float).tolist()


def _normalize_color(value: Optional[str]) -> str:
    color = (value or "").strip().lower()
    mapping = {
        "tab:blue": "#1f77b4",
        "tab:red": "#d62728",
        "tab:orange": "#ff7f0e",
        "tab:green": "#2ca02c",
    }
    return mapping.get(color, value or "#0f172a")


@dataclass
class _CfgOverride:
    key: str
    value: Any


@contextmanager
def _override_cfg(overrides: Iterable[_CfgOverride]):
    previous: dict[str, Any] = {}
    for item in overrides:
        previous[item.key] = getattr(cfg, item.key)
        setattr(cfg, item.key, item.value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(cfg, key, value)


class FragmentsRunRequest(BaseModel):
    filepath: str = Field(..., description="Path to the spectrum file")
    scan: int = Field(1, ge=1)
    peptide: str
    mz_min: Optional[float] = None
    mz_max: Optional[float] = None
    ion_types: Optional[list[str]] = None
    frag_min_charge: Optional[int] = Field(None, ge=1)
    frag_max_charge: Optional[int] = Field(None, ge=1)
    match_tol_ppm: Optional[float] = Field(None, gt=0)
    precursor_match_tol_ppm: Optional[float] = Field(None, gt=0, description="Separate ppm tolerance for precursor mode")
    min_cosine: Optional[float] = Field(None, ge=0, le=1)
    isodec_css_thresh: Optional[float] = Field(None, ge=0, le=1)
    copies: Optional[int] = Field(None, ge=1)
    amidated: Optional[bool] = None
    disulfide_bonds: Optional[int] = Field(None, ge=0)
    disulfide_map: Optional[str] = None
    enable_isodec_rules: Optional[bool] = None
    enable_h_transfer: Optional[bool] = None
    enable_neutral_losses: Optional[bool] = None


class DiagnoseRunRequest(BaseModel):
    filepath: str = Field(..., description="Path to the spectrum file")
    scan: int = Field(1, ge=1)
    peptide: str
    ion_spec: str = Field(..., description="Ion spec to diagnose, e.g., 'c7^2+', 'z12-2H2O^3+'")
    h_transfer: int = Field(0, ge=-2, le=2, description="H transfer degree (-2 to +2)")
    mz_min: Optional[float] = None
    mz_max: Optional[float] = None
    frag_min_charge: Optional[int] = Field(None, ge=1)
    frag_max_charge: Optional[int] = Field(None, ge=1)
    match_tol_ppm: Optional[float] = Field(None, gt=0)
    min_cosine: Optional[float] = Field(None, ge=0, le=1)
    copies: Optional[int] = Field(None, ge=1)
    amidated: Optional[bool] = None
    disulfide_bonds: Optional[int] = Field(None, ge=0)
    disulfide_map: Optional[str] = None
    enable_isodec_rules: Optional[bool] = None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    return {
        "filepath": str(cfg.filepath),
        "scan": int(cfg.SCAN),
        "peptide": str(cfg.PEPTIDE),
        "mz_min": cfg.MZ_MIN,
        "mz_max": cfg.MZ_MAX,
        "ion_types": list(cfg.ION_TYPES),
        "frag_min_charge": int(cfg.FRAG_MIN_CHARGE),
        "frag_max_charge": int(cfg.FRAG_MAX_CHARGE),
        "match_tol_ppm": float(cfg.MATCH_TOL_PPM),
        "precursor_match_tol_ppm": float(getattr(cfg, "PRECURSOR_MATCH_TOL_PPM", cfg.MATCH_TOL_PPM)),
        "min_cosine": float(cfg.MIN_COSINE),
        "isodec_css_thresh": float(cfg.ISODEC_CSS_THRESH),
        "copies": int(cfg.COPIES),
        "amidated": bool(cfg.AMIDATED),
        "disulfide_bonds": int(cfg.DISULFIDE_BONDS),
        "disulfide_map": _disulfide_map_to_string(cfg.DISULFIDE_MAP),
        "enable_isodec_rules": bool(cfg.ENABLE_ISODEC_RULES),
        "enable_h_transfer": bool(cfg.ENABLE_H_TRANSFER),
        "enable_neutral_losses": bool(cfg.ENABLE_NEUTRAL_LOSSES),
    }


def _build_overrides(req: FragmentsRunRequest, filepath: str, plot_mode: str = "fragments") -> list[_CfgOverride]:
    disulfide_map = _parse_disulfide_map(req.disulfide_map)
    ion_types = tuple(req.ion_types) if req.ion_types else tuple(cfg.ION_TYPES)
    overrides = [
        _CfgOverride("filepath", filepath),
        _CfgOverride("SCAN", int(req.scan)),
        _CfgOverride("PLOT_MODE", str(plot_mode)),
        _CfgOverride("PEPTIDE", req.peptide),
        _CfgOverride("MZ_MIN", req.mz_min),
        _CfgOverride("MZ_MAX", req.mz_max),
        _CfgOverride("ION_TYPES", ion_types),
        _CfgOverride("EXPORT_FRAGMENTS_CSV", False),
    ]
    if req.frag_min_charge is not None:
        overrides.append(_CfgOverride("FRAG_MIN_CHARGE", int(req.frag_min_charge)))
    if req.frag_max_charge is not None:
        overrides.append(_CfgOverride("FRAG_MAX_CHARGE", int(req.frag_max_charge)))
    if req.match_tol_ppm is not None:
        overrides.append(_CfgOverride("MATCH_TOL_PPM", float(req.match_tol_ppm)))
    if req.min_cosine is not None:
        overrides.append(_CfgOverride("MIN_COSINE", float(req.min_cosine)))
    css_thresh = req.isodec_css_thresh if req.isodec_css_thresh is not None else req.min_cosine
    if css_thresh is not None:
        overrides.append(_CfgOverride("ISODEC_CSS_THRESH", float(css_thresh)))
    if req.copies is not None:
        overrides.append(_CfgOverride("COPIES", int(req.copies)))
    if req.amidated is not None:
        overrides.append(_CfgOverride("AMIDATED", bool(req.amidated)))
    if req.disulfide_bonds is not None:
        overrides.append(_CfgOverride("DISULFIDE_BONDS", int(req.disulfide_bonds)))
    if disulfide_map:
        overrides.append(_CfgOverride("DISULFIDE_MAP", disulfide_map))
    if req.enable_isodec_rules is not None:
        overrides.append(_CfgOverride("ENABLE_ISODEC_RULES", bool(req.enable_isodec_rules)))
    if req.enable_h_transfer is not None:
        overrides.append(_CfgOverride("ENABLE_H_TRANSFER", bool(req.enable_h_transfer)))
    if req.enable_neutral_losses is not None:
        overrides.append(_CfgOverride("ENABLE_NEUTRAL_LOSSES", bool(req.enable_neutral_losses)))
    return overrides


def _build_precursor_overrides(req: FragmentsRunRequest, filepath: str) -> list[_CfgOverride]:
    disulfide_map = _parse_disulfide_map(req.disulfide_map)
    overrides = [
        _CfgOverride("filepath", filepath),
        _CfgOverride("SCAN", int(req.scan)),
        _CfgOverride("PLOT_MODE", "precursor"),
        _CfgOverride("PEPTIDE", req.peptide),
        _CfgOverride("MZ_MIN", req.mz_min),
        _CfgOverride("MZ_MAX", req.mz_max),
        _CfgOverride("EXPORT_FRAGMENTS_CSV", False),
    ]
    # Use precursor-specific ppm tolerance if provided, otherwise fall back to match_tol_ppm
    if req.precursor_match_tol_ppm is not None:
        overrides.append(_CfgOverride("PRECURSOR_MATCH_TOL_PPM", float(req.precursor_match_tol_ppm)))
    elif req.match_tol_ppm is not None:
        overrides.append(_CfgOverride("PRECURSOR_MATCH_TOL_PPM", float(req.match_tol_ppm)))
    if req.match_tol_ppm is not None:
        overrides.append(_CfgOverride("MATCH_TOL_PPM", float(req.match_tol_ppm)))
    if req.min_cosine is not None:
        overrides.append(_CfgOverride("MIN_COSINE", float(req.min_cosine)))
    css_thresh = req.isodec_css_thresh if req.isodec_css_thresh is not None else req.min_cosine
    if css_thresh is not None:
        overrides.append(_CfgOverride("ISODEC_CSS_THRESH", float(css_thresh)))
    if req.frag_min_charge is not None:
        overrides.append(_CfgOverride("PRECURSOR_MIN_CHARGE", int(req.frag_min_charge)))
    if req.frag_max_charge is not None:
        overrides.append(_CfgOverride("PRECURSOR_MAX_CHARGE", int(req.frag_max_charge)))
    if req.copies is not None:
        overrides.append(_CfgOverride("COPIES", int(req.copies)))
    if req.amidated is not None:
        overrides.append(_CfgOverride("AMIDATED", bool(req.amidated)))
    if req.disulfide_bonds is not None:
        overrides.append(_CfgOverride("DISULFIDE_BONDS", int(req.disulfide_bonds)))
    if disulfide_map:
        overrides.append(_CfgOverride("DISULFIDE_MAP", disulfide_map))
    if req.enable_isodec_rules is not None:
        overrides.append(_CfgOverride("ENABLE_ISODEC_RULES", bool(req.enable_isodec_rules)))
    return overrides


def _build_charge_reduced_overrides(req: FragmentsRunRequest, filepath: str) -> list[_CfgOverride]:
    disulfide_map = _parse_disulfide_map(req.disulfide_map)
    overrides = [
        _CfgOverride("filepath", filepath),
        _CfgOverride("SCAN", int(req.scan)),
        _CfgOverride("PLOT_MODE", "charge_reduced"),
        _CfgOverride("PEPTIDE", req.peptide),
        _CfgOverride("MZ_MIN", req.mz_min),
        _CfgOverride("MZ_MAX", req.mz_max),
        _CfgOverride("EXPORT_FRAGMENTS_CSV", False),
    ]
    if req.match_tol_ppm is not None:
        overrides.append(_CfgOverride("MATCH_TOL_PPM", float(req.match_tol_ppm)))
    if req.min_cosine is not None:
        overrides.append(_CfgOverride("MIN_COSINE", float(req.min_cosine)))
    css_thresh = req.isodec_css_thresh if req.isodec_css_thresh is not None else req.min_cosine
    if css_thresh is not None:
        overrides.append(_CfgOverride("ISODEC_CSS_THRESH", float(css_thresh)))
    if req.frag_min_charge is not None:
        overrides.append(_CfgOverride("CR_MIN_CHARGE", int(req.frag_min_charge)))
    if req.frag_max_charge is not None:
        overrides.append(_CfgOverride("CR_MAX_CHARGE", int(req.frag_max_charge)))
    if req.copies is not None:
        overrides.append(_CfgOverride("COPIES", int(req.copies)))
    if req.amidated is not None:
        overrides.append(_CfgOverride("AMIDATED", bool(req.amidated)))
    if req.disulfide_bonds is not None:
        overrides.append(_CfgOverride("DISULFIDE_BONDS", int(req.disulfide_bonds)))
    if disulfide_map:
        overrides.append(_CfgOverride("DISULFIDE_MAP", disulfide_map))
    if req.enable_isodec_rules is not None:
        overrides.append(_CfgOverride("ENABLE_ISODEC_RULES", bool(req.enable_isodec_rules)))
    return overrides


def _build_raw_overrides(req: FragmentsRunRequest, filepath: str) -> list[_CfgOverride]:
    return [
        _CfgOverride("filepath", filepath),
        _CfgOverride("SCAN", int(req.scan)),
        _CfgOverride("PLOT_MODE", "raw"),
        _CfgOverride("PEPTIDE", req.peptide),
        _CfgOverride("MZ_MIN", req.mz_min),
        _CfgOverride("MZ_MAX", req.mz_max),
        _CfgOverride("EXPORT_FRAGMENTS_CSV", False),
    ]


def _run_fragments_impl(
    req: FragmentsRunRequest,
    filepath_override: Optional[str] = None,
    plot_mode: str = "fragments",
) -> dict[str, Any]:
    filepath = filepath_override or req.filepath
    plot_mode = str(plot_mode or "fragments").lower()
    css_thresh = req.isodec_css_thresh if req.isodec_css_thresh is not None else req.min_cosine
    logger.info(
        "run %s: scan=%s tol_ppm=%s css_thresh=%s ions=%s mz=[%s,%s] charge=[%s,%s]",
        plot_mode,
        req.scan,
        req.match_tol_ppm,
        css_thresh,
        req.ion_types,
        req.mz_min,
        req.mz_max,
        req.frag_min_charge,
        req.frag_max_charge,
    )
    overrides = _build_overrides(req, filepath, plot_mode=plot_mode)
    precursor_result = None
    try:
        with _override_cfg(overrides):
            cfg.require_isodec_rules()
            isodec_config = cfg.build_isodec_config()
            residues = parse_custom_sequence(cfg.PEPTIDE)
            spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=bool(cfg.ENABLE_CENTROID))
            spectrum = preprocess_spectrum(spectrum)
            if bool(getattr(cfg, "PRECURSOR_CHAIN_TO_FRAGMENTS", False)):
                # Calibrate via precursor search without plotting.
                precursor_result = run_precursor_headless(residues, spectrum, isodec_config)
                spectrum = np.asarray(precursor_result.get("spectrum"), dtype=float)
            result = run_fragments_headless(residues, spectrum, isodec_config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("run %s complete: count=%s", plot_mode, len(result.get("best", []) or []))

    sequence_raw = cfg.PEPTIDE
    sequence = "".join(aa for aa, _mods in residues)
    seq_len = len(residues)
    spectrum_mz = np.asarray(result.get("spectrum_mz", []), dtype=float)
    spectrum_int = np.asarray(result.get("spectrum_int", []), dtype=float)
    spectrum_mz_ds, spectrum_int_ds = _downsample_xy(spectrum_mz, spectrum_int)
    theory_mz, theory_int = _extract_theory_peaks(result.get("best", []))

    fragments: list[dict[str, Any]] = []
    for m in result.get("best", []):
        ion_type_norm = _normalize_ion_type(str(m.get("ion_type", "")))
        frag_len = int(m.get("frag_len", 0) or 0)
        frag_index = _fragment_index(seq_len, ion_type_norm or "", frag_len)
        theory_mz_row, theory_int_row = _theory_from_dist(m.get("dist"))
        fragments.append(
            {
                "frag_id": m.get("frag_id", ""),
                "ion_type": ion_type_norm,
                "ion_type_raw": m.get("ion_type", ""),
                "series": m.get("series", ""),
                "frag_len": frag_len,
                "fragment_index": frag_index,
                "charge": int(m.get("charge", 0) or 0),
                "obs_mz": _safe_float(m.get("obs_mz")),
                "obs_int": _safe_float(m.get("obs_int")),
                "obs_rel_int": _safe_float(m.get("obs_rel_int")),
                "anchor_theory_mz": _safe_float(m.get("anchor_theory_mz")),
                "anchor_ppm": _safe_float(m.get("ppm")),
                "css": _safe_float(m.get("score")),
                "rawcos": _safe_float(m.get("raw_score")),
                "label": m.get("label", ""),
                "variant_suffix": m.get("variant_suffix", ""),
                "theory_mz": theory_mz_row,
                "theory_intensity": theory_int_row,
            }
        )

    fragments = [f for f in fragments if f["ion_type"] and f["fragment_index"] is not None]
    precursor_summary = None
    if precursor_result:
        precursor_summary = {
            "match_found": bool(precursor_result.get("match_found")),
            "best_charge": precursor_result.get("best_z"),
            "best_state": precursor_result.get("best_state"),
            "best_css": _safe_float(precursor_result.get("best_css")),
            "shift_ppm": _safe_float(precursor_result.get("shift_ppm")),
            "best_obs_mz": _safe_float(precursor_result.get("best_obs_mz")),
            "best_theory_mz": _safe_float(precursor_result.get("best_theory_mz")),
        }
    return {
        "mode": plot_mode,
        "sequence": sequence,
        "sequence_raw": sequence_raw,
        "scan": req.scan,
        "fragments": fragments,
        "count": len(fragments),
        "precursor": precursor_summary,
        "spectrum": {
            "mz": spectrum_mz_ds,
            "intensity": spectrum_int_ds,
            "raw_points": int(spectrum_mz.size),
            "points": len(spectrum_mz_ds),
        },
        "theory": {
            "mz": theory_mz,
            "intensity": theory_int,
        },
    }


def _run_precursor_impl(req: FragmentsRunRequest, filepath_override: Optional[str] = None) -> dict[str, Any]:
    filepath = filepath_override or req.filepath
    precursor_ppm = req.precursor_match_tol_ppm if req.precursor_match_tol_ppm is not None else req.match_tol_ppm
    logger.info(
        "run precursor: scan=%s precursor_tol_ppm=%s charge=[%s,%s]",
        req.scan,
        precursor_ppm,
        req.frag_min_charge,
        req.frag_max_charge,
    )
    overrides = _build_precursor_overrides(req, filepath)
    try:
        with _override_cfg(overrides):
            cfg.require_isodec_rules()
            isodec_config = cfg.build_isodec_config()
            residues = parse_custom_sequence(cfg.PEPTIDE)
            spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=bool(cfg.ENABLE_CENTROID))
            spectrum = preprocess_spectrum(spectrum)
            result = run_precursor_headless(residues, spectrum, isodec_config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sequence_raw = cfg.PEPTIDE
    sequence = "".join(aa for aa, _mods in residues)
    spectrum_mz = np.asarray(result.get("spectrum_mz", []), dtype=float)
    spectrum_int = np.asarray(result.get("spectrum_int", []), dtype=float)
    spectrum_mz_ds, spectrum_int_ds = _downsample_xy(spectrum_mz, spectrum_int)
    theory_mz, theory_int = _theory_from_dist(result.get("best_theory_dist"))

    candidates: list[dict[str, Any]] = []
    for cand in result.get("candidates", []) or []:
        cand_theory_mz, cand_theory_int = _theory_from_dist(cand.get("dist"))
        candidates.append(
            {
                "charge": int(cand.get("charge", 0) or 0),
                "state": cand.get("state"),
                "obs_mz": _safe_float(cand.get("obs_mz")),
                "anchor_theory_mz": _safe_float(cand.get("anchor_theory_mz")),
                "ppm": _safe_float(cand.get("ppm")),
                "css": _safe_float(cand.get("css")),
                "accepted": bool(cand.get("accepted", False)),
                "iteration": int(cand.get("iteration", 0) or 0),
                "theory_mz": cand_theory_mz,
                "theory_intensity": cand_theory_int,
                "color": "#ef4444",
            }
        )

    window_raw = result.get("plot_window")
    plot_window = None
    if isinstance(window_raw, (list, tuple)) and len(window_raw) == 2:
        w_min = _safe_float(window_raw[0])
        w_max = _safe_float(window_raw[1])
        if w_min is not None and w_max is not None and w_min < w_max:
            plot_window = {"min": w_min, "max": w_max}

    precursor_summary = {
        "match_found": bool(result.get("match_found")),
        "best_charge": result.get("best_z"),
        "best_state": result.get("best_state"),
        "best_css": _safe_float(result.get("best_css")),
        "shift_ppm": _safe_float(result.get("shift_ppm")),
        "best_obs_mz": _safe_float(result.get("best_obs_mz")),
        "best_theory_mz": _safe_float(result.get("best_theory_mz")),
    }

    logger.info(
        "run precursor complete: match=%s best_z=%s css=%s shift_ppm=%s",
        precursor_summary["match_found"],
        precursor_summary["best_charge"],
        precursor_summary["best_css"],
        precursor_summary["shift_ppm"],
    )

    return {
        "mode": "precursor",
        "sequence": sequence,
        "sequence_raw": sequence_raw,
        "scan": req.scan,
        "precursor": precursor_summary,
        "candidates": candidates,
        "count": len(candidates),
        "plot_window": plot_window,
        "spectrum": {
            "mz": spectrum_mz_ds,
            "intensity": spectrum_int_ds,
            "raw_points": int(spectrum_mz.size),
            "points": len(spectrum_mz_ds),
        },
        "theory": {
            "mz": theory_mz,
            "intensity": theory_int,
        },
    }


def _run_charge_reduced_impl(req: FragmentsRunRequest, filepath_override: Optional[str] = None) -> dict[str, Any]:
    filepath = filepath_override or req.filepath
    logger.info(
        "run charge_reduced: scan=%s tol_ppm=%s charge=[%s,%s]",
        req.scan,
        req.match_tol_ppm,
        req.frag_min_charge,
        req.frag_max_charge,
    )
    overrides = _build_charge_reduced_overrides(req, filepath)
    precursor_result = None
    try:
        with _override_cfg(overrides):
            cfg.require_isodec_rules()
            isodec_config = cfg.build_isodec_config()
            residues = parse_custom_sequence(cfg.PEPTIDE)
            spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=bool(cfg.ENABLE_CENTROID))
            spectrum = preprocess_spectrum(spectrum)
            if bool(getattr(cfg, "PRECURSOR_CHAIN_TO_FRAGMENTS", False)):
                # Calibrate via precursor search without plotting.
                precursor_result = run_precursor_headless(residues, spectrum, isodec_config)
                spectrum = np.asarray(precursor_result.get("spectrum"), dtype=float)
            result = run_charge_reduced_headless(residues, spectrum, isodec_config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sequence_raw = cfg.PEPTIDE
    sequence = "".join(aa for aa, _mods in residues)
    spectrum_mz = np.asarray(result.get("spectrum_mz", []), dtype=float)
    spectrum_int = np.asarray(result.get("spectrum_int", []), dtype=float)
    spectrum_mz_ds, spectrum_int_ds = _downsample_xy(spectrum_mz, spectrum_int)
    matches = result.get("matches", []) or []
    theory_mz, theory_int = _extract_theory_peaks(matches)

    overlays: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for m in matches:
        dist = m.get("dist")
        dist_full = m.get("dist_full")
        if isinstance(dist, np.ndarray) and dist.size:
            mz_vals = dist[:, 0].astype(float).tolist()
            int_vals = dist[:, 1].astype(float).tolist()
        else:
            mz_vals, int_vals = [], []
        color_norm = _normalize_color(m.get("color"))
        anchor_mz = None
        ppm = None
        if isinstance(dist_full, np.ndarray) and dist_full.size:
            anchor_idx = int(np.argmax(dist_full[:, 1]))
            anchor_mz = float(dist_full[anchor_idx, 0])
            obs_mz = _safe_float(m.get("obs_mz"))
            if obs_mz is not None and anchor_mz:
                ppm = ((obs_mz - anchor_mz) / anchor_mz) * 1e6
        overlays.append(
            {
                "label": m.get("label", ""),
                "mz": mz_vals,
                "intensity": int_vals,
                "color": color_norm,
                "charge": int(m.get("z", 0) or 0),
                "state": m.get("state", ""),
                "target": m.get("target", ""),
                "obs_mz": _safe_float(m.get("obs_mz")),
                "css": _safe_float(m.get("css")),
            }
        )
        candidates.append(
            {
                "label": m.get("label", ""),
                "charge": int(m.get("z", 0) or 0),
                "state": m.get("state", ""),
                "target": m.get("target", ""),
                "obs_mz": _safe_float(m.get("obs_mz")),
                "obs_int": _safe_float(m.get("obs_int")),
                "anchor_theory_mz": anchor_mz,
                "ppm": _safe_float(ppm),
                "css": _safe_float(m.get("css")),
                "theory_mz": mz_vals,
                "theory_intensity": int_vals,
                "color": color_norm,
            }
        )

    logger.info("run charge_reduced complete: count=%s", len(candidates))

    precursor_summary = None
    if precursor_result:
        precursor_summary = {
            "match_found": bool(precursor_result.get("match_found")),
            "best_charge": precursor_result.get("best_z"),
            "best_state": precursor_result.get("best_state"),
            "best_css": _safe_float(precursor_result.get("best_css")),
            "shift_ppm": _safe_float(precursor_result.get("shift_ppm")),
            "best_obs_mz": _safe_float(precursor_result.get("best_obs_mz")),
            "best_theory_mz": _safe_float(precursor_result.get("best_theory_mz")),
        }

    return {
        "mode": "charge_reduced",
        "sequence": sequence,
        "sequence_raw": sequence_raw,
        "scan": req.scan,
        "candidates": candidates,
        "count": len(candidates),
        "overlays": overlays,
        "precursor": precursor_summary,
        "spectrum": {
            "mz": spectrum_mz_ds,
            "intensity": spectrum_int_ds,
            "raw_points": int(spectrum_mz.size),
            "points": len(spectrum_mz_ds),
        },
        "theory": {
            "mz": theory_mz,
            "intensity": theory_int,
        },
    }


def _run_raw_impl(req: FragmentsRunRequest, filepath_override: Optional[str] = None) -> dict[str, Any]:
    filepath = filepath_override or req.filepath
    logger.info("run raw: scan=%s mz=[%s,%s]", req.scan, req.mz_min, req.mz_max)
    overrides = _build_raw_overrides(req, filepath)
    try:
        with _override_cfg(overrides):
            residues = parse_custom_sequence(cfg.PEPTIDE) if cfg.PEPTIDE else []
            spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=bool(cfg.ENABLE_CENTROID))
            spectrum = preprocess_spectrum(spectrum)
            result = run_raw_headless(spectrum)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sequence_raw = cfg.PEPTIDE
    sequence = "".join(aa for aa, _mods in residues) if residues else sequence_raw
    spectrum_mz = np.asarray(result.get("spectrum_mz", []), dtype=float)
    spectrum_int = np.asarray(result.get("spectrum_int", []), dtype=float)
    spectrum_mz_ds, spectrum_int_ds = _downsample_xy(spectrum_mz, spectrum_int)

    logger.info("run raw complete: points=%s", len(spectrum_mz_ds))

    return {
        "mode": "raw",
        "sequence": sequence,
        "sequence_raw": sequence_raw,
        "scan": req.scan,
        "count": 0,
        "spectrum": {
            "mz": spectrum_mz_ds,
            "intensity": spectrum_int_ds,
            "raw_points": int(spectrum_mz.size),
            "points": len(spectrum_mz_ds),
        },
        "theory": {"mz": [], "intensity": []},
    }


@app.post("/api/run/raw")
def run_raw(req: FragmentsRunRequest) -> dict[str, Any]:
    return _run_raw_impl(req)


@app.post("/api/run/raw/upload")
async def run_raw_upload(file: UploadFile = File(...), payload: str = Form(...)) -> dict[str, Any]:
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid payload JSON") from exc

    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())
        payload_data["filepath"] = temp_path
        req = FragmentsRunRequest(**payload_data)
        return _run_raw_impl(req, filepath_override=temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/run/fragments")
def run_fragments(req: FragmentsRunRequest) -> dict[str, Any]:
    return _run_fragments_impl(req, plot_mode="fragments")


@app.post("/api/run/fragments/upload")
async def run_fragments_upload(file: UploadFile = File(...), payload: str = Form(...)) -> dict[str, Any]:
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid payload JSON") from exc

    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())
        payload_data["filepath"] = temp_path
        req = FragmentsRunRequest(**payload_data)
        return _run_fragments_impl(req, filepath_override=temp_path, plot_mode="fragments")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/run/complex_fragments")
def run_complex_fragments(req: FragmentsRunRequest) -> dict[str, Any]:
    return _run_fragments_impl(req, plot_mode="complex_fragments")


@app.post("/api/run/complex_fragments/upload")
async def run_complex_fragments_upload(file: UploadFile = File(...), payload: str = Form(...)) -> dict[str, Any]:
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid payload JSON") from exc

    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())
        payload_data["filepath"] = temp_path
        req = FragmentsRunRequest(**payload_data)
        return _run_fragments_impl(req, filepath_override=temp_path, plot_mode="complex_fragments")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/run/precursor")
def run_precursor(req: FragmentsRunRequest) -> dict[str, Any]:
    return _run_precursor_impl(req)


@app.post("/api/run/precursor/upload")
async def run_precursor_upload(file: UploadFile = File(...), payload: str = Form(...)) -> dict[str, Any]:
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid payload JSON") from exc

    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())
        payload_data["filepath"] = temp_path
        req = FragmentsRunRequest(**payload_data)
        return _run_precursor_impl(req, filepath_override=temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/run/charge_reduced")
def run_charge_reduced(req: FragmentsRunRequest) -> dict[str, Any]:
    return _run_charge_reduced_impl(req)


@app.post("/api/run/charge_reduced/upload")
async def run_charge_reduced_upload(file: UploadFile = File(...), payload: str = Form(...)) -> dict[str, Any]:
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid payload JSON") from exc

    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())
        payload_data["filepath"] = temp_path
        req = FragmentsRunRequest(**payload_data)
        return _run_charge_reduced_impl(req, filepath_override=temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _build_diagnose_overrides(req: DiagnoseRunRequest, filepath: str) -> list[_CfgOverride]:
    disulfide_map = _parse_disulfide_map(req.disulfide_map) if hasattr(req, 'disulfide_map') and req.disulfide_map else []
    overrides = [
        _CfgOverride("filepath", filepath),
        _CfgOverride("SCAN", int(req.scan)),
        _CfgOverride("PLOT_MODE", "diagnose"),
        _CfgOverride("PEPTIDE", req.peptide),
        _CfgOverride("MZ_MIN", req.mz_min),
        _CfgOverride("MZ_MAX", req.mz_max),
        _CfgOverride("DIAGNOSE_ION_SPEC", req.ion_spec),
        _CfgOverride("DIAGNOSE_H_TRANSFER", int(req.h_transfer)),
        _CfgOverride("DIAGNOSE_EXPORT_CSV", False),
        _CfgOverride("DIAGNOSE_SHOW_PLOT", False),
    ]
    if req.frag_min_charge is not None:
        overrides.append(_CfgOverride("FRAG_MIN_CHARGE", int(req.frag_min_charge)))
    if req.frag_max_charge is not None:
        overrides.append(_CfgOverride("FRAG_MAX_CHARGE", int(req.frag_max_charge)))
    if req.match_tol_ppm is not None:
        overrides.append(_CfgOverride("MATCH_TOL_PPM", float(req.match_tol_ppm)))
    if req.min_cosine is not None:
        overrides.append(_CfgOverride("MIN_COSINE", float(req.min_cosine)))
        overrides.append(_CfgOverride("ISODEC_CSS_THRESH", float(req.min_cosine)))
    if req.copies is not None:
        overrides.append(_CfgOverride("COPIES", int(req.copies)))
    if req.amidated is not None:
        overrides.append(_CfgOverride("AMIDATED", bool(req.amidated)))
    if req.disulfide_bonds is not None:
        overrides.append(_CfgOverride("DISULFIDE_BONDS", int(req.disulfide_bonds)))
    if disulfide_map:
        overrides.append(_CfgOverride("DISULFIDE_MAP", disulfide_map))
    if req.enable_isodec_rules is not None:
        overrides.append(_CfgOverride("ENABLE_ISODEC_RULES", bool(req.enable_isodec_rules)))
    return overrides


def _run_diagnose_impl(req: DiagnoseRunRequest, filepath_override: Optional[str] = None) -> dict[str, Any]:
    filepath = filepath_override or req.filepath
    logger.info(
        "run diagnose: scan=%s ion_spec=%s h_transfer=%s",
        req.scan,
        req.ion_spec,
        req.h_transfer,
    )
    overrides = _build_diagnose_overrides(req, filepath)
    try:
        with _override_cfg(overrides):
            cfg.require_isodec_rules()
            isodec_config = cfg.build_isodec_config()
            residues = parse_custom_sequence(cfg.PEPTIDE)
            spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=bool(cfg.ENABLE_CENTROID))
            spectrum = preprocess_spectrum(spectrum)
            result = run_diagnose_headless(
                residues, spectrum, isodec_config,
                ion_spec=req.ion_spec,
                h_transfer=int(req.h_transfer),
            )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sequence_raw = cfg.PEPTIDE
    sequence = "".join(aa for aa, _mods in residues)
    spectrum_mz = result.get("spectrum_mz", [])
    spectrum_int = result.get("spectrum_int", [])
    spectrum_mz_ds, spectrum_int_ds = _downsample_xy(
        np.array(spectrum_mz, dtype=float),
        np.array(spectrum_int, dtype=float),
    )
    theory_mz = result.get("theory_mz", [])
    theory_int = result.get("theory_int", [])

    logger.info("run diagnose complete: results=%s", len(result.get("results", [])))

    return {
        "mode": "diagnose",
        "sequence": sequence,
        "sequence_raw": sequence_raw,
        "scan": req.scan,
        "ion_spec": req.ion_spec,
        "h_transfer": req.h_transfer,
        "parsed": result.get("parsed"),
        "charges_scanned": result.get("charges_scanned", []),
        "results": result.get("results", []),
        "best": result.get("best"),
        "count": len(result.get("results", [])),
        "spectrum": {
            "mz": spectrum_mz_ds,
            "intensity": spectrum_int_ds,
            "raw_points": len(spectrum_mz),
            "points": len(spectrum_mz_ds),
        },
        "theory": {
            "mz": theory_mz,
            "intensity": theory_int,
        },
    }


@app.post("/api/run/diagnose")
def run_diagnose(req: DiagnoseRunRequest) -> dict[str, Any]:
    return _run_diagnose_impl(req)


@app.post("/api/run/diagnose/upload")
async def run_diagnose_upload(file: UploadFile = File(...), payload: str = Form(...)) -> dict[str, Any]:
    try:
        payload_data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid payload JSON") from exc

    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = tmp.name
            tmp.write(await file.read())
        payload_data["filepath"] = temp_path
        req = DiagnoseRunRequest(**payload_data)
        return _run_diagnose_impl(req, filepath_override=temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ecd_api:app", host="127.0.0.1", port=8001, reload=False)
