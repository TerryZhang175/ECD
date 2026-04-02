# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

| Action | Command |
|--------|---------|
| Start API server | `python ecd_api.py` (runs at `http://127.0.0.1:8001`) |
| Run CLI analysis | `python personalized.py` (config via `personalized_config.py`) |
| Open UI | `open http://127.0.0.1:8001/ui/` |

## Architecture Overview

ECD Analyzer is a mass spectrometry tool for Electron Capture Dissociation (ECD) data. It has three interfaces:

1. **CLI Pipeline** (`personalized.py`) - Entry point that loads spectra and dispatches to mode functions
2. **FastAPI Backend** (`ecd_api.py`) - REST API at port 8001, serves UI static files
3. **Vanilla JS UI** (`ui/`) - Plotly-based spectrum visualization, no bundler/framework

### Core Module Flow

```
personalized.py (entry)
    ↓ loads spectrum via
personalized_importer.py → SingleScanImporter (.txt/.dat/.csv)
    ↓ parses peptide via
personalized_sequence.py → residue compositions, ion series, neutral losses
    ↓ generates theory via
personalized_theory.py → isotope distributions, simplex fitting
    ↓ matches and scores via
personalized_match.py → cosine similarity, IsoDec rules, ranking
    ↓ outputs via
personalized_modes.py → mode implementations (precursor, fragments, diagnose, etc.)
```

### Key Configuration Pattern

`personalized_config.py` holds all tunable parameters. The API uses a context manager (`_override_cfg`) to temporarily override config values per-request without mutating global state:

```python
@contextmanager
def _override_cfg(overrides: Iterable[_CfgOverride]):
    # Temporarily set config values, restore on exit
```

### Optional Dependencies Pattern

The codebase gracefully falls back when UniDec is unavailable:

```python
try:
    from unidec.modules.unidecstructure import IsoDecConfig
except Exception:
    from personalized_isodec import IsoDecConfig  # fallback
```

### Chained Calibration

When `PRECURSOR_CHAIN_TO_FRAGMENTS=True` (default), fragment modes first run precursor calibration to correct mass drift before matching.

## Analysis Modes

Set via `PLOT_MODE` in config or API parameter:

| Mode | Purpose |
|------|---------|
| `raw` | Display spectrum without theory overlays |
| `precursor` | Precursor search and lock-mass calibration |
| `charge_reduced` | Charge-reduced precursor search (ECD/ETD) |
| `fragments` | Backbone fragment matching (b, y, c, z ions) |
| `complex_fragments` | Monomer + fragment non-covalent complexes |
| `diagnose` | Detailed diagnostics for a specific ion (why matched/failed) |

## UI Development Notes

- **Cache busting**: `ui/index.html` uses query strings for JS/CSS (e.g., `app.js?v=20260329-1`)
- **After UI changes**: Bump version in `index.html` to force browser cache refresh
- **See `.codex/skills/ecd-ui-debug/SKILL.md`** for detailed UI debugging workflow

## Sample Data

- `sample/WT/` - Wild-type test data
- `sample/Q10R/`, `sample/Q10R2/` - Q10R mutant test data

## Output Directories

- `match_outputs/` - CLI CSV outputs
- `diagnose_outputs/` - Diagnostics CSV files
- `reports/` - Generated analysis reports
