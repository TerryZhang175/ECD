# ECD Analyzer

ECD Analyzer is a focused mass spectrometry (MS/MS) tool for Electron Capture Dissociation (ECD) data. It provides:
- A configurable Python analysis pipeline
- A FastAPI backend for programmatic access
- A lightweight vanilla JS UI for interactive exploration

The core workflow is: load a spectrum, optionally calibrate using precursor lock-mass, match theoretical ions, and explore results through coverage and spectrum overlays.

## Key Features

- Multiple analysis modes: `raw`, `precursor`, `charge_reduced`, `fragments`, and `complex_fragments`
- Per-run parameter overrides through the API
- Interactive UI:
  - Fragment coverage view inspired by Alphaviz-style layouts
  - Click/hover an ion to zoom the spectrum
  - Selecting a specific ion temporarily isolates its theoretical peaks

## Project Structure

```text
ECD/
|- personalized.py           # CLI entry point
|- personalized_config.py    # Global defaults and switches
|- personalized_modes.py     # Mode implementations
|- personalized_sequence.py  # Peptide parsing and compositions
|- personalized_theory.py    # Isotope distributions and scoring helpers
|- personalized_match.py     # Matching and ranking logic
|- personalized_plot.py      # Matplotlib plotting (CLI)
|- ecd_api.py                # FastAPI backend
|- ui/
|  |- index.html             # UI shell
|  |- app.js                 # UI logic (Plotly rendering)
|  |- styles.css             # UI styles
|- match_outputs/            # CSV outputs from CLI runs
```

## Requirements

- Python 3.10+
- Recommended packages:
  - `numpy`
  - `pyteomics`
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `matplotlib`
  - `numba` (optional, used when available)

## Quick Start

### 1) Create an environment and install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pyteomics fastapi uvicorn pydantic matplotlib numba
```

### 2) Run the API

```bash
python ecd_api.py
```

The API runs at `http://127.0.0.1:8001`.

### 3) Open the UI (no separate server needed)

```bash
# API now serves the UI:
open http://127.0.0.1:8001/ui/
```

Make sure the API is already running at `http://127.0.0.1:8001`.

### 4) Run from the CLI (optional)

```bash
python personalized.py
```

CLI behavior is controlled by `personalized_config.py`.

## Input Data Format

The importer currently supports single-scan text formats:
- `.txt`
- `.dat`
- `.csv`

Expected layout: two columns of `m/z` and `intensity` values.

## Modes

Set `PLOT_MODE` in `personalized_config.py` (CLI) or choose a mode in the UI/API:

- `raw`: show the spectrum without theory overlays
- `precursor`: precursor search and lock-mass calibration (uses `PRECURSOR_MATCH_TOL_PPM` for matching tolerance)
- `charge_reduced`: charge-reduced precursor search
- `fragments`: backbone fragment matching
- `complex_fragments`: monomer + fragment non-covalent complexes

Note: `diagnose` mode still exists in the CLI for deep debugging, but it is not exposed in the UI.

## API Endpoints

### Health and config

- `GET /api/health`
- `GET /api/config`

### Run modes

Each mode has both JSON and upload variants:

- `POST /api/run/raw`
- `POST /api/run/raw/upload`
- `POST /api/run/precursor`
- `POST /api/run/precursor/upload`
- `POST /api/run/charge_reduced`
- `POST /api/run/charge_reduced/upload`
- `POST /api/run/fragments`
- `POST /api/run/fragments/upload`
- `POST /api/run/complex_fragments`
- `POST /api/run/complex_fragments/upload`

### Example: fragments run

```bash
curl -X POST http://127.0.0.1:8001/api/run/fragments \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "/absolute/path/to/scan.txt",
    "scan": 1,
    "peptide": "PEPTIDE",
    "mz_min": 400,
    "mz_max": 2000,
    "ion_types": ["b", "y", "c", "z"],
    "frag_min_charge": 1,
    "frag_max_charge": 6,
    "match_tol_ppm": 20,
    "min_cosine": 0.7
  }'
```

## Notes and Tips

- The API overrides `personalized_config.py` values per request using a safe context manager.
- `PRECURSOR_MATCH_TOL_PPM` (default: 20 ppm) provides a separate matching tolerance for precursor mode, independent of `MATCH_TOL_PPM` used for fragment matching.
- When `PRECURSOR_CHAIN_TO_FRAGMENTS = True`, fragments and complex fragments runs are calibrated first using precursor lock-mass correction (without plotting in the API path).
- In the UI, clicking an ion in Fragment Coverage or the Results Table will:
  1) zoom the spectrum around that ion, and
  2) temporarily show only that ion's theoretical peaks.
