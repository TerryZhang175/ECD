from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import personalized_config as cfg
from ecd_api import FragmentsRunRequest, _run_fragments_impl

ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = ROOT / 'sample' / 'Q10R' / 'Q10R_annotated'
SPECTRUM_DIR = ROOT / 'sample' / 'Q10R' / 'Centroid (lock mass)_副本'
REPORT_DIR = ROOT / 'reports' / 'q10r_truth_score_fragments_eval'
SUMMARY_JSON = REPORT_DIR / 'summary.json'
SUMMARY_MD = REPORT_DIR / 'summary.md'

PEPTIDE = 'KCNTATCATRRLANFLVHSSNNFGAILSSTNVGSNTY'
COPIES = 2
AMIDATED = True
DISULFIDE_BONDS = 2
DISULFIDE_MAP = '2-7'
FRAG_MIN_CHARGE = 1
FRAG_MAX_CHARGE = 5


def parse_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def load_manual_truth() -> tuple[dict[int, set[tuple[str, int, int]]], list[int]]:
    scan_truth: dict[int, set[tuple[str, int, int]]] = {}
    re_nums: list[int] = []
    for ann_path in sorted(ANNOTATION_DIR.glob('ECDRE*-ion-state_man.csv'), key=lambda p: int(p.name.split('ECDRE')[1].split('-')[0])):
        re_num = int(ann_path.name.split('ECDRE')[1].split('-')[0])
        re_nums.append(re_num)
        truth = set()
        with ann_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ion_type = str(row.get('Base Type') or row.get('Type') or '').strip().lower()
                if ion_type not in {'b', 'c'}:
                    continue
                pos = parse_float(row.get('Pos'))
                charge = parse_float(row.get('Charge'))
                matched = parse_float(row.get('Matched'))
                if pos is None or charge is None:
                    continue
                if matched == 1.0:
                    truth.add((ion_type, int(pos), int(charge)))
        scan_truth[re_num] = truth
    return scan_truth, re_nums


def item_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    def safe(key: str) -> float:
        value = parse_float(item.get(key))
        return value if value is not None else float('-inf')
    return (
        safe('selection_score'),
        safe('score'),
        safe('coverage'),
        safe('css'),
    )


def run_eval(*, truth_score_enable: bool, threshold: float, use_ranking: bool) -> dict[str, Any]:
    manual_truth, re_nums = load_manual_truth()
    old = (
        cfg.FRAG_TRUTH_SCORE_ENABLE,
        cfg.FRAG_TRUTH_SCORE_THRESHOLD,
        cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING,
    )
    cfg.FRAG_TRUTH_SCORE_ENABLE = bool(truth_score_enable)
    cfg.FRAG_TRUTH_SCORE_THRESHOLD = float(threshold)
    cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING = bool(use_ranking)

    try:
        per_re = []
        total_tp = total_fp = total_fn = 0
        total_pred = total_truth = 0

        for re_num in re_nums:
            spec_path = SPECTRUM_DIR / f'ECDRE{re_num}.txt'
            req = FragmentsRunRequest(
                filepath=str(spec_path.resolve()),
                scan=1,
                peptide=PEPTIDE,
                ion_types=['b', 'c'],
                frag_min_charge=FRAG_MIN_CHARGE,
                frag_max_charge=FRAG_MAX_CHARGE,
                copies=COPIES,
                amidated=AMIDATED,
                disulfide_bonds=DISULFIDE_BONDS,
                disulfide_map=DISULFIDE_MAP,
            )
            result = _run_fragments_impl(req)
            predicted_best: dict[tuple[str, int, int], dict[str, Any]] = {}
            for item in result.get('fragments', []):
                ion_type = str(item.get('ion_type') or '').lower()
                if ion_type not in {'b', 'c'}:
                    continue
                pos = int(item.get('frag_len') or 0)
                charge = int(item.get('charge') or 0)
                key = (ion_type, pos, charge)
                cur = predicted_best.get(key)
                if cur is None or item_sort_key(item) > item_sort_key(cur):
                    predicted_best[key] = item

            pred_keys = set(predicted_best)
            truth_keys = manual_truth.get(re_num, set())
            tp = len(pred_keys & truth_keys)
            fp = len(pred_keys - truth_keys)
            fn = len(truth_keys - pred_keys)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_re.append(
                {
                    're': f'RE{re_num}',
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'predicted': len(pred_keys),
                    'truth': len(truth_keys),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_pred += len(pred_keys)
            total_truth += len(truth_keys)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            'truth_score_enable': bool(truth_score_enable),
            'threshold': float(threshold),
            'use_ranking': bool(use_ranking),
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'predicted': total_pred,
            'truth': total_truth,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'per_re': per_re,
        }
    finally:
        (
            cfg.FRAG_TRUTH_SCORE_ENABLE,
            cfg.FRAG_TRUTH_SCORE_THRESHOLD,
            cfg.FRAG_TRUTH_SCORE_USE_FOR_RANKING,
        ) = old


def write_report(results: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(results, indent=2), encoding='utf-8')
    lines = ['# Q10R Truth Score Fragments Eval', '']
    for result in results:
        if not result['truth_score_enable']:
            title = 'baseline'
        elif float(result['threshold']) > 1.0:
            title = 'truth-score ranking-only'
        else:
            title = 'truth-score enabled'
        lines.append(f"## {title} | threshold={result['threshold']:.6f}")
        lines.append('')
        lines.append(f"- TP: `{result['tp']}`")
        lines.append(f"- FP: `{result['fp']}`")
        lines.append(f"- FN: `{result['fn']}`")
        lines.append(f"- Precision: `{result['precision']:.4f}`")
        lines.append(f"- Recall: `{result['recall']:.4f}`")
        lines.append(f"- F1: `{result['f1']:.4f}`")
        lines.append('')
    SUMMARY_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    cases = [
        (False, cfg.FRAG_TRUTH_SCORE_THRESHOLD, False),
        (True, 1.1, True),
        (True, 0.95, True),
        (True, 0.90, True),
        (True, 0.85, True),
        (True, 0.80, True),
        (True, 0.50, True),
        (True, 0.39323091648480935, True),
    ]
    results = [
        run_eval(truth_score_enable=enable, threshold=threshold, use_ranking=use_ranking)
        for enable, threshold, use_ranking in cases
    ]
    write_report(results)
    for result in results:
        if not result['truth_score_enable']:
            label = 'baseline'
        elif float(result['threshold']) > 1.0:
            label = 'truth-score@ranking-only'
        else:
            label = f"truth-score@{result['threshold']:.6f}"
        print(label, f"precision={result['precision']:.4f}", f"recall={result['recall']:.4f}", f"f1={result['f1']:.4f}", f"tp={result['tp']}", f"fp={result['fp']}", f"fn={result['fn']}")


if __name__ == '__main__':
    main()
