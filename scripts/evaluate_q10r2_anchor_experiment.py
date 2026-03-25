from __future__ import annotations

import csv
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import personalized_config as cfg
from ecd_api import FragmentsRunRequest, _build_overrides, _override_cfg, parse_custom_sequence
from personalized import load_spectrum, preprocess_spectrum
from personalized_modes import run_fragments_headless

SAMPLE_DIR = ROOT / "sample" / "Q10R2"
REPORT_DIR = ROOT / "reports" / "q10r2_anchor_experiment"
SUMMARY_JSON = REPORT_DIR / "summary.json"
SUMMARY_MD = REPORT_DIR / "summary.md"
PER_RE_CSV = REPORT_DIR / "per_re.csv"

PEPTIDE = "KCNTATCATRRLANFLVHSSNNFGAILSSTNVGSNTY"
COPIES = 2
AMIDATED = True
DISULFIDE_BONDS = 2
DISULFIDE_MAP = "2-7"
FRAG_MIN_CHARGE = 1
FRAG_MAX_CHARGE = 5

RE_PAT = re.compile(r"ECDRE(\d+)-ion-state_man\.csv$", re.I)


def parse_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def item_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    def safe(key: str) -> float:
        value = parse_float(item.get(key))
        return value if value is not None else float("-inf")

    return (
        safe("selection_score"),
        safe("score"),
        safe("coverage"),
        safe("css"),
    )


def load_manual_truth() -> tuple[dict[int, set[tuple[str, int, int]]], list[int]]:
    scan_truth: dict[int, set[tuple[str, int, int]]] = {}
    re_nums: list[int] = []
    for ann_path in sorted(SAMPLE_DIR.glob("ECDRE*-ion-state_man.csv"), key=lambda p: int(RE_PAT.search(p.name).group(1))):
        match = RE_PAT.search(ann_path.name)
        if not match:
            continue
        re_num = int(match.group(1))
        re_nums.append(re_num)
        truth = set()
        with ann_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ion_type = str(row.get("Base Type") or row.get("Type") or "").strip().lower()
                if ion_type not in {"b", "c"}:
                    continue
                pos = parse_float(row.get("Pos"))
                charge = parse_float(row.get("Charge"))
                matched = parse_float(row.get("Matched"))
                if pos is None or charge is None:
                    continue
                if matched == 1.0:
                    truth.add((ion_type, int(pos), int(charge)))
        scan_truth[re_num] = truth
    return scan_truth, re_nums


@contextmanager
def anchor_experiment_enabled(enabled: bool):
    old = cfg.FRAG_ANCHOR_USE_HYPOTHESIS_SCORING
    cfg.FRAG_ANCHOR_USE_HYPOTHESIS_SCORING = bool(enabled)
    try:
        yield
    finally:
        cfg.FRAG_ANCHOR_USE_HYPOTHESIS_SCORING = old


@contextmanager
def anchor_intensity_fallback_enabled(enabled: bool):
    old = cfg.FRAG_ANCHOR_USE_INTENSITY_FALLBACK
    cfg.FRAG_ANCHOR_USE_INTENSITY_FALLBACK = bool(enabled)
    try:
        yield
    finally:
        cfg.FRAG_ANCHOR_USE_INTENSITY_FALLBACK = old


def run_scan(re_num: int) -> dict[str, Any]:
    spec_path = SAMPLE_DIR / f"ECDRE{re_num}.txt"
    req = FragmentsRunRequest(
        filepath=str(spec_path.resolve()),
        scan=1,
        peptide=PEPTIDE,
        ion_types=["b", "c"],
        frag_min_charge=FRAG_MIN_CHARGE,
        frag_max_charge=FRAG_MAX_CHARGE,
        copies=COPIES,
        amidated=AMIDATED,
        disulfide_bonds=DISULFIDE_BONDS,
        disulfide_map=DISULFIDE_MAP,
        precursor_calibration=False,
    )
    overrides = _build_overrides(req, str(spec_path.resolve()), plot_mode="fragments")
    with _override_cfg(overrides):
        cfg.require_isodec_rules()
        isodec_config = cfg.build_isodec_config()
        residues = parse_custom_sequence(cfg.PEPTIDE)
        spectrum = load_spectrum(cfg.filepath, cfg.SCAN, prefer_centroid=bool(cfg.ENABLE_CENTROID))
        spectrum = preprocess_spectrum(spectrum)
        return run_fragments_headless(residues, spectrum, isodec_config)


def metrics(pred_keys: set[tuple[str, int, int]], truth_keys: set[tuple[str, int, int]]) -> dict[str, float]:
    tp = len(pred_keys & truth_keys)
    fp = len(pred_keys - truth_keys)
    fn = len(truth_keys - pred_keys)
    tn = 0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def result_maps(result: dict[str, Any]) -> tuple[set[tuple[str, int, int]], set[tuple[str, int, int]]]:
    raw_best: dict[tuple[str, int, int], dict[str, Any]] = {}
    final_best: dict[tuple[str, int, int], dict[str, Any]] = {}
    for group_name, target in (("matches", raw_best), ("best", final_best)):
        for item in result.get(group_name, []) or []:
            ion_type = str(item.get("ion_type") or "").lower()
            if ion_type not in {"b", "c"}:
                continue
            pos = int(item.get("frag_len") or 0)
            charge = int(item.get("charge") or 0)
            key = (ion_type, pos, charge)
            current = target.get(key)
            if current is None or item_sort_key(item) > item_sort_key(current):
                target[key] = item
    return set(raw_best), set(final_best)


def run_case(
    label: str,
    *,
    use_anchor_hypothesis: bool,
    use_anchor_intensity_fallback: bool,
    truth_by_re: dict[int, set[tuple[str, int, int]]],
    re_nums: list[int],
) -> dict[str, Any]:
    per_re: list[dict[str, Any]] = []
    total_raw = {"tp": 0, "fp": 0, "fn": 0}
    total_final = {"tp": 0, "fp": 0, "fn": 0}

    with anchor_experiment_enabled(use_anchor_hypothesis), anchor_intensity_fallback_enabled(use_anchor_intensity_fallback):
        for re_num in re_nums:
            result = run_scan(re_num)
            raw_keys, final_keys = result_maps(result)
            truth_keys = truth_by_re.get(re_num, set())
            raw_metrics = metrics(raw_keys, truth_keys)
            final_metrics = metrics(final_keys, truth_keys)
            per_re.append(
                {
                    "re": f"RE{re_num}",
                    "mode": "raw",
                    **raw_metrics,
                    "predicted": len(raw_keys),
                    "truth": len(truth_keys),
                }
            )
            per_re.append(
                {
                    "re": f"RE{re_num}",
                    "mode": "final",
                    **final_metrics,
                    "predicted": len(final_keys),
                    "truth": len(truth_keys),
                }
            )
            for key in ("tp", "fp", "fn"):
                total_raw[key] += int(raw_metrics[key])
                total_final[key] += int(final_metrics[key])

    def finalize(total: dict[str, int]) -> dict[str, float]:
        precision = total["tp"] / (total["tp"] + total["fp"]) if (total["tp"] + total["fp"]) else 0.0
        recall = total["tp"] / (total["tp"] + total["fn"]) if (total["tp"] + total["fn"]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            **total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "label": label,
        "use_anchor_hypothesis": bool(use_anchor_hypothesis),
        "use_anchor_intensity_fallback": bool(use_anchor_intensity_fallback),
        "raw": finalize(total_raw),
        "final": finalize(total_final),
        "per_re": per_re,
    }


def main() -> None:
    truth_by_re, re_nums = load_manual_truth()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [
        run_case(
            "baseline",
            use_anchor_hypothesis=False,
            use_anchor_intensity_fallback=False,
            truth_by_re=truth_by_re,
            re_nums=re_nums,
        ),
        run_case(
            "intensity_fallback",
            use_anchor_hypothesis=False,
            use_anchor_intensity_fallback=True,
            truth_by_re=truth_by_re,
            re_nums=re_nums,
        ),
    ]
    SUMMARY_JSON.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    rows = []
    for case in cases:
        for row in case["per_re"]:
            rows.append({"case": case["label"], **row})
    pd.DataFrame(rows).to_csv(PER_RE_CSV, index=False)

    lines = ["# Q10R2 Anchor Experiment", ""]
    for case in cases:
        lines.append(f"## {case['label']}")
        lines.append("")
        lines.append(
            f"- Raw: TP `{case['raw']['tp']}`, FP `{case['raw']['fp']}`, FN `{case['raw']['fn']}`, "
            f"precision `{case['raw']['precision']:.4f}`, recall `{case['raw']['recall']:.4f}`, F1 `{case['raw']['f1']:.4f}`"
        )
        lines.append(
            f"- Final: TP `{case['final']['tp']}`, FP `{case['final']['fp']}`, FN `{case['final']['fn']}`, "
            f"precision `{case['final']['precision']:.4f}`, recall `{case['final']['recall']:.4f}`, F1 `{case['final']['f1']:.4f}`"
        )
        lines.append("")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for case in cases:
        print(case["label"])
        print(
            "raw",
            f"tp={case['raw']['tp']}",
            f"fp={case['raw']['fp']}",
            f"fn={case['raw']['fn']}",
            f"precision={case['raw']['precision']:.4f}",
            f"recall={case['raw']['recall']:.4f}",
            f"f1={case['raw']['f1']:.4f}",
        )
        print(
            "final",
            f"tp={case['final']['tp']}",
            f"fp={case['final']['fp']}",
            f"fn={case['final']['fn']}",
            f"precision={case['final']['precision']:.4f}",
            f"recall={case['final']['recall']:.4f}",
            f"f1={case['final']['f1']:.4f}",
        )


if __name__ == "__main__":
    main()
