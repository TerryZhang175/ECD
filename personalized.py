from __future__ import annotations

import numpy as np
from personalized_importer import ImporterFactory

import personalized_config as cfg
from personalized_modes import (
    run_diagnose_mode,
    run_fragments_mode,
    run_charge_reduced_mode,
    run_precursor_mode,
    run_raw_mode,
)
from personalized_sequence import parse_custom_sequence


def load_spectrum(filepath: str, scan: int, prefer_centroid: bool = True) -> np.ndarray:
    importer = ImporterFactory.create_importer(filepath)
    if prefer_centroid and hasattr(importer, "grab_centroid_data"):
        spectrum = importer.grab_centroid_data(scan)
    elif hasattr(importer, "get_single_scan"):
        spectrum = importer.get_single_scan(scan)
    elif hasattr(importer, "grab_centroid_data"):
        spectrum = importer.grab_centroid_data(scan)
    else:
        raise AttributeError("Importer does not support grab_centroid_data or get_single_scan.")
    return np.asarray(spectrum, dtype=float)


def preprocess_spectrum(spectrum: np.ndarray) -> np.ndarray:
    if spectrum.ndim != 2 or spectrum.shape[1] != 2:
        raise ValueError(f"Expected spectrum shape (N, 2), got {spectrum.shape}")
    spectrum = spectrum[np.isfinite(spectrum[:, 0]) & np.isfinite(spectrum[:, 1])]
    spectrum = spectrum[spectrum[:, 1] > 0]
    spectrum = spectrum[np.argsort(spectrum[:, 0])]

    if cfg.MZ_MIN is not None or cfg.MZ_MAX is not None:
        mz_min = -np.inf if cfg.MZ_MIN is None else float(cfg.MZ_MIN)
        mz_max = np.inf if cfg.MZ_MAX is None else float(cfg.MZ_MAX)
        if mz_min >= mz_max:
            raise ValueError(f"Invalid m/z window: MZ_MIN={cfg.MZ_MIN}, MZ_MAX={cfg.MZ_MAX}")
        spectrum = spectrum[(spectrum[:, 0] >= mz_min) & (spectrum[:, 0] <= mz_max)]
        if len(spectrum) == 0:
            raise ValueError(f"No peaks remain after applying m/z window: [{mz_min}, {mz_max}]")

    return spectrum


def main() -> None:
    mode = str(cfg.PLOT_MODE).lower()
    spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=(mode != "raw"))
    if mode == "raw":
        run_raw_mode(spectrum)
        return

    cfg.require_isodec_rules()
    isodec_config = cfg.build_isodec_config()

    if not cfg.PEPTIDE:
        raise ValueError('Set PEPTIDE (e.g. "ACDEFGHIK" or "CKLH[PO4]CKLAH")')
    residues = parse_custom_sequence(cfg.PEPTIDE)
    spectrum = preprocess_spectrum(spectrum)

    if mode == "precursor":
        run_precursor_mode(residues, spectrum, isodec_config)
    elif mode == "charge_reduced":
        run_charge_reduced_mode(residues, spectrum, isodec_config)
    elif mode == "fragments" or mode == "complex_fragments":
        if bool(getattr(cfg, "PRECURSOR_CHAIN_TO_FRAGMENTS", False)):
            spectrum = run_precursor_mode(residues, spectrum, isodec_config)
        run_fragments_mode(residues, spectrum, isodec_config)
    elif mode == "diagnose":
        run_diagnose_mode(residues, spectrum, isodec_config)
    else:
        raise ValueError(
            'PLOT_MODE must be "raw", "precursor", "charge_reduced", "fragments", "complex_fragments", or "diagnose".'
        )


if __name__ == "__main__":
    main()
