from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import re
import tempfile
import logging
from typing import Any, Iterable, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

import personalized_config as cfg
from personalized import load_spectrum, preprocess_spectrum
from personalized_modes import run_fragments_headless
from personalized_sequence import parse_custom_sequence


app = FastAPI(title="ECD Analyzer API", version="0.1.0")

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
    min_cosine: Optional[float] = Field(None, ge=0, le=1)
    isodec_css_thresh: Optional[float] = Field(None, ge=0, le=1)
    copies: Optional[int] = Field(None, ge=1)
    amidated: Optional[bool] = None
    disulfide_bonds: Optional[int] = Field(None, ge=0)
    disulfide_map: Optional[str] = None
    enable_isodec_rules: Optional[bool] = None
    enable_h_transfer: Optional[bool] = None
    enable_neutral_losses: Optional[bool] = None


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


def _build_overrides(req: FragmentsRunRequest, filepath: str) -> list[_CfgOverride]:
    disulfide_map = _parse_disulfide_map(req.disulfide_map)
    ion_types = tuple(req.ion_types) if req.ion_types else tuple(cfg.ION_TYPES)
    overrides = [
        _CfgOverride("filepath", filepath),
        _CfgOverride("SCAN", int(req.scan)),
        _CfgOverride("PLOT_MODE", "fragments"),
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


def _run_fragments_impl(req: FragmentsRunRequest, filepath_override: Optional[str] = None) -> dict[str, Any]:
    filepath = filepath_override or req.filepath
    css_thresh = req.isodec_css_thresh if req.isodec_css_thresh is not None else req.min_cosine
    logger.info(
        "run fragments: scan=%s tol_ppm=%s css_thresh=%s ions=%s mz=[%s,%s] charge=[%s,%s]",
        req.scan,
        req.match_tol_ppm,
        css_thresh,
        req.ion_types,
        req.mz_min,
        req.mz_max,
        req.frag_min_charge,
        req.frag_max_charge,
    )
    overrides = _build_overrides(req, filepath)
    try:
        with _override_cfg(overrides):
            cfg.require_isodec_rules()
            isodec_config = cfg.build_isodec_config()
            residues = parse_custom_sequence(cfg.PEPTIDE)
            spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=True)
            spectrum = preprocess_spectrum(spectrum)
            result = run_fragments_headless(residues, spectrum, isodec_config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info("run fragments complete: count=%s", len(result.get("best", []) or []))

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
            }
        )

    fragments = [f for f in fragments if f["ion_type"] and f["fragment_index"] is not None]
    return {
        "sequence": sequence,
        "sequence_raw": sequence_raw,
        "scan": req.scan,
        "fragments": fragments,
        "count": len(fragments),
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


@app.post("/api/run/fragments")
def run_fragments(req: FragmentsRunRequest) -> dict[str, Any]:
    return _run_fragments_impl(req)


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
        return _run_fragments_impl(req, filepath_override=temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ecd_api:app", host="127.0.0.1", port=8001, reload=False)
