# AGENTS.md — ECD Analyzer

## Skills

A skill is a set of local instructions stored in a `SKILL.md` file under `.codex/skills/`.

### Available Skills

- **ecd-ui-debug**: Debug and modify the ECD UI served from `/ui`. Use when changing frontend behavior, investigating UI/backend mismatches, or when a fix seems correct in code but not visible in the browser. Includes cache-busting checks and browser-side verification. (file: `.codex/skills/ecd-ui-debug/SKILL.md`)
- **ecd-response-style**: Repository-specific response style for ECD work. Use when replying about this project so answers stay focused on the code/task and do not append generic product or workflow promotion. (file: `.codex/skills/ecd-response-style/SKILL.md`)
- **ecd-truth-comparison**: Compare manual `b/c` truth annotations against the UI-equivalent fragments algorithm over `sample/`. Auto-discovers raw/truth files, pairs scans, and scores TP/FP/FN with strict prefix plus chemical-formula matching. (file: `.codex/skills/ecd-truth-comparison/SKILL.md`)

### How to Use Skills

- If the task clearly involves the ECD frontend, use `ecd-ui-debug`.
- Use `ecd-response-style` for normal user-facing replies in this repository.
- If the task is to compare manual truth annotations against algorithm output across sample datasets, use `ecd-truth-comparison`.
- Read only the `SKILL.md` body unless more files are explicitly referenced from it.

---

## Project Overview

ECD Analyzer is a mass spectrometry (MS/MS) tool for Electron Capture Dissociation data. It provides:
- A configurable Python analysis pipeline
- A FastAPI backend for programmatic access
- A lightweight vanilla JS UI for interactive exploration (Plotly for charts)

### Architecture

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic, NumPy, Pyteomics
- **Frontend**: Vanilla JS (`ui/app.js`), HTML (`ui/index.html`), CSS (`ui/styles.css`)
- **No formal build system**: No bundler, no TypeScript. Frontend is served as static files.
- **No pyproject.toml**: Dependencies are installed manually via pip.

---

## Build / Run Commands

| Action | Command |
|---|---|
| **Install dependencies** | `pip install numpy pyteomics fastapi uvicorn pydantic matplotlib numba` |
| **Start API server** | `python ecd_api.py` (runs at `http://127.0.0.1:8001`) |
| **Open UI** | `open http://127.0.0.1:8001/ui/` (served by the API, no separate server) |
| **Run CLI analysis** | `python personalized.py` (behavior controlled by `personalized_config.py`) |
| **Run a script** | `python scripts/<script_name>.py` |

### Testing

- **No formal test framework** (no pytest, unittest, or test runner configured).
- To verify changes: start the API, open the UI, and test manually.
- For browser automation: Playwright is available via `npm` (`node_modules/playwright`).
- Run Playwright scripts with: `npx playwright test` or `node <script>.js`.

---

## Code Style Guidelines

### Python

#### Imports

- Use `from __future__ import annotations` at the top of every module.
- Order: standard library → third-party → local.
- Import `personalized_config` as `cfg` (convention throughout the codebase).
- Use explicit imports, not `from module import *`.

```python
from __future__ import annotations

import re
from typing import Optional

import numpy as np
from fastapi import FastAPI

import personalized_config as cfg
from personalized_match import composition_to_formula
```

#### Type Hints

- Use Python 3.10+ style type hints (list, tuple, dict, not List, Tuple, Dict).
- Use `Optional[X]` for nullable values.
- Use `Any` sparingly, only when the type is truly dynamic.

```python
def _safe_float(value: Any) -> Optional[float]:
    ...

def nearest_peak_index(sorted_mzs: np.ndarray, target_mz: float) -> int:
    ...
```

#### Naming Conventions

- **Functions/variables**: `snake_case`
- **Private helpers**: prefix with `_` (e.g., `_safe_float`, `_normalize_ion_type`)
- **Constants**: `UPPER_SNAKE_CASE` (defined in `personalized_config.py`)
- **Classes**: `PascalCase`

#### Error Handling

- Raise `ValueError` for invalid input or configuration errors.
- Raise `HTTPException` in FastAPI endpoints with appropriate status codes.
- Use `try/except` for optional dependencies (e.g., unidec fallback).
- Never silently swallow exceptions — log or re-raise.

```python
if charge == 0:
    raise ValueError("charge must be non-zero")
```

#### Docstrings

- Use docstrings sparingly — only for non-obvious public functions.
- Keep them concise. No need for docstrings on private helpers if the name is clear.

### JavaScript (ui/app.js)

- Vanilla JS, no framework, no TypeScript.
- Use `const` for DOM references and constants.
- Use `let` for mutable variables.
- camelCase for functions and variables.
- Cache DOM elements at the top of the file.
- Use `addEventListener` for event binding.
- No module system — single file, global scope.

### CSS (ui/styles.css)

- Use CSS custom properties (`--variable-name`) for theming.
- BEM-like naming for component classes (`.panel-header`, `.status-pill`).
- No preprocessor (no Sass/LESS).

---

## Project Structure

```
ECD/
├── personalized.py           # CLI entry point
├── personalized_config.py    # Global defaults and switches
├── personalized_modes.py     # Mode implementations (raw, precursor, fragments, etc.)
├── personalized_sequence.py  # Peptide parsing and compositions
├── personalized_theory.py    # Isotope distributions and scoring helpers
├── personalized_match.py     # Matching and ranking logic
├── personalized_plot.py      # Matplotlib plotting (CLI)
├── ecd_api.py                # FastAPI backend (1300+ lines)
├── ui/
│   ├── index.html            # UI shell (static HTML)
│   ├── app.js                # UI logic (Plotly rendering, DOM manipulation)
│   └── styles.css            # UI styles
├── scripts/                  # Analysis and reporting scripts
├── match_outputs/            # CSV outputs from CLI runs
└── .codex/skills/            # Skill definitions for this repository
```

---

## Key Patterns

### Optional Dependencies (try/except fallback)

The codebase gracefully handles optional dependencies like `unidec`:

```python
try:
    from unidec.modules.unidecstructure import IsoDecConfig
except Exception:
    from personalized_isodec import IsoDecConfig
```

### API Request Handling

- Each run mode has both JSON and upload variants (e.g., `/api/run/fragments` and `/api/run/fragments/upload`).
- Use Pydantic `BaseModel` for request/response schemas.
- Config overrides are applied per-request using a context manager (`safe_config_context`).

### UI Cache Busting

When modifying `ui/app.js` or `ui/styles.css`, bump the version query string in `ui/index.html`:

```html
<link rel="stylesheet" href="styles.css?v=20260314-2" />
<script defer src="app.js?v=20260314-2"></script>
```

Always verify the browser loaded the new URLs (see `ecd-ui-debug` skill).

---

## What NOT to Do

- Do not add a build system, bundler, or TypeScript to the frontend.
- Do not use `from module import *`.
- Do not suppress type errors with `as any`, `@ts-ignore`, or `# type: ignore` without explicit justification.
- Do not add unnecessary dependencies — prefer stdlib or existing packages.
- Do not refactor code unrelated to your task.
- Do not add empty docstrings or excessive comments.

---

## Verification Checklist

After making changes:

1. Start the API: `python ecd_api.py`
2. Open the UI at `http://127.0.0.1:8001/ui/`
3. Test the affected mode/feature manually
4. For UI changes: verify browser loaded updated assets (check network tab or cache-busting version)
5. Run any relevant scripts to confirm they still work
