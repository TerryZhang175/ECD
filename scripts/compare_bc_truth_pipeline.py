from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecd_api import FragmentsRunRequest, _run_fragments_impl


SCAN_RE = re.compile(r"(?:ECD)?RE\s*(\d+)", re.IGNORECASE)
FORMULA_RE = re.compile(r"([A-Z][a-z]?)([-+]?\d*)")
FORMULA_ORDER = [
    "C",
    "H",
    "N",
    "O",
    "S",
    "P",
    "Cl",
    "Br",
    "Na",
    "K",
    "Fe",
    "Ca",
    "Mg",
    "Zn",
    "Ni",
]
MANUAL_REQUIRED_COLUMNS = {"Name", "Matched", "Charge", "Pos"}
MANUAL_TYPE_COLUMNS = {"Base Type", "Type"}
LOSS_NORMALIZATIONS = {
    "h2o": "H2O",
    "nh3": "NH3",
    "co2": "CO2",
    "co": "CO",
}
MANUAL_FILENAME_HINTS = ("-man", "_man", "manual", "mannual", "manually")


@dataclass(frozen=True)
class MetaCandidate:
    path: Path
    peptide: str
    copies: int
    amidated: bool
    disulfide_bonds: int
    disulfide_map: str
    frag_min_charge: int
    frag_max_charge: int
    token: str


@dataclass(frozen=True)
class DatasetContext:
    name: str
    root: Path
    meta_file: Path
    peptide: str
    copies: int
    amidated: bool
    disulfide_bonds: int
    disulfide_map: str
    frag_min_charge: int
    frag_max_charge: int


@dataclass(frozen=True)
class ScanPair:
    dataset: str
    scan_num: int
    raw_path: Path
    truth_path: Path
    pair_distance: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare manual b/c truth annotations under sample/ against UI-equivalent "
            "fragments outputs using strict prefix+charge+formula matching."
        )
    )
    parser.add_argument(
        "--sample-root",
        default=str(ROOT / "sample"),
        help="Root directory containing dataset subfolders (default: sample/).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "bc_truth_comparison"),
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Restrict to one or more top-level dataset directories under sample/ (repeatable).",
    )
    parser.add_argument(
        "--scan",
        action="append",
        type=int,
        default=[],
        help="Restrict to one or more scan numbers (repeatable).",
    )
    parser.add_argument(
        "--disulfide-variant-mode",
        default="algorithm",
        choices=["algorithm", "manual_annotation"],
        help=(
            "Disulfide variant generation mode for fragments runs. "
            "'algorithm' keeps current hypotheses; 'manual_annotation' keeps only the "
            "variant formulas used by current manual truth CSVs."
        ),
    )
    return parser.parse_args()


def quiet_call(func, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(*args, **kwargs)


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def strip_sequence_suffix(text: str) -> str:
    text = str(text)
    lowered = text.lower()
    if lowered.endswith("_sequence"):
        return text[: -len("_sequence")]
    return text


def dataset_family_token(dataset_name: str) -> str:
    token = normalize_token(dataset_name)
    return re.sub(r"\d+$", "", token) or token


def extract_scan_num(path: Path) -> Optional[int]:
    match = SCAN_RE.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def looks_like_meta_file(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    try:
        preview = path.read_text(encoding="utf-8", errors="replace").splitlines()[:12]
    except Exception:
        return False
    joined = "\n".join(line.strip().lower() for line in preview if line.strip())
    return "sequence:" in joined and ("copies=" in joined or "disulfide" in joined)


def parse_sequence_meta(path: Path) -> MetaCandidate:
    peptide = ""
    copies = 2
    amidated = True
    disulfide_bonds = 0
    disulfide_map = ""
    frag_min_charge = 1
    frag_max_charge = 5
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if line.startswith("Sequence:"):
            peptide = line.split(":", 1)[1].strip()
        elif lower.startswith("copies="):
            copies = int(line.split("=", 1)[1].strip())
        elif lower.startswith("admidated=") or lower.startswith("amidated="):
            amidated = line.split("=", 1)[1].strip().lower() == "true"
        elif lower.startswith("disulfide bonds="):
            disulfide_bonds = int(line.split("=", 1)[1].strip())
        elif lower.startswith("disulfide map="):
            disulfide_map = line.split("=", 1)[1].strip()
        elif lower.startswith("frag min charge="):
            frag_min_charge = int(line.split("=", 1)[1].strip())
        elif lower.startswith("frag max charge="):
            frag_max_charge = int(line.split("=", 1)[1].strip())
    if not peptide:
        raise ValueError(f"No Sequence found in {path}")
    token = normalize_token(strip_sequence_suffix(path.stem))
    return MetaCandidate(
        path=path,
        peptide=peptide,
        copies=copies,
        amidated=amidated,
        disulfide_bonds=disulfide_bonds,
        disulfide_map=disulfide_map,
        frag_min_charge=frag_min_charge,
        frag_max_charge=frag_max_charge,
        token=token,
    )


def discover_meta_candidates(sample_root: Path) -> list[MetaCandidate]:
    candidates: list[MetaCandidate] = []
    for path in sorted(sample_root.rglob("*.txt")):
        if not looks_like_meta_file(path):
            continue
        try:
            candidates.append(parse_sequence_meta(path))
        except Exception:
            continue
    return candidates


def choose_meta_candidate(
    dataset_root: Path, sample_root: Path, meta_candidates: list[MetaCandidate]
) -> Optional[MetaCandidate]:
    local = [cand for cand in meta_candidates if dataset_root == cand.path.parent or dataset_root in cand.path.parents]
    if local:
        return min(local, key=lambda cand: (len(cand.path.relative_to(dataset_root).parts), len(cand.path.name)))

    root_token = normalize_token(dataset_root.name)
    root_family = dataset_family_token(dataset_root.name)

    def score(cand: MetaCandidate) -> tuple[int, int, int, str]:
        cand_token = cand.token
        cand_family = re.sub(r"\d+$", "", cand_token) or cand_token
        relation = 9
        if cand_token == root_token:
            relation = 0
        elif cand_token == root_family or cand_family == root_token:
            relation = 1
        elif cand_family == root_family:
            relation = 2
        elif cand_token in root_token or root_token in cand_token:
            relation = 3
        elif cand_family in root_token or root_family in cand_token:
            relation = 4
        try:
            rel_depth = len(cand.path.relative_to(sample_root).parts)
        except Exception:
            rel_depth = len(cand.path.parts)
        return (relation, rel_depth, len(cand.path.name), str(cand.path))

    best = min(meta_candidates, key=score, default=None)
    if best is None:
        return None
    return best if score(best)[0] <= 4 else None


def is_manual_truth_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    if extract_scan_num(path) is None:
        return False
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except Exception:
        return False
    columns = {str(col).strip() for col in header}
    if not MANUAL_REQUIRED_COLUMNS.issubset(columns):
        return False
    if not (MANUAL_TYPE_COLUMNS & columns):
        return False
    return True


def manual_path_priority(path: Path) -> int:
    haystack = f"{path.parent.name}/{path.name}".lower()
    return 0 if any(hint in haystack for hint in MANUAL_FILENAME_HINTS) else 1


def discover_raw_files(dataset_root: Path) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(dataset_root.rglob("*.txt")):
        if looks_like_meta_file(path):
            continue
        scan_num = extract_scan_num(path)
        if scan_num is None:
            continue
        out[scan_num].append(path)
    return out


def discover_truth_files(dataset_root: Path) -> dict[int, list[Path]]:
    out: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(dataset_root.rglob("*.csv")):
        if not is_manual_truth_csv(path):
            continue
        scan_num = extract_scan_num(path)
        if scan_num is None:
            continue
        out[scan_num].append(path)
    return out


def path_distance(a: Path, b: Path) -> int:
    common = Path(os.path.commonpath([a.parent, b.parent]))
    a_parts = len(a.parent.relative_to(common).parts)
    b_parts = len(b.parent.relative_to(common).parts)
    return a_parts + b_parts


def pair_scan_files(
    dataset_name: str,
    raw_map: dict[int, list[Path]],
    truth_map: dict[int, list[Path]],
    scan_filter: set[int],
) -> tuple[list[ScanPair], list[Path], list[Path]]:
    pairs: list[ScanPair] = []
    unpaired_raw: list[Path] = []
    unpaired_truth: list[Path] = []

    scan_nums = sorted(set(raw_map) | set(truth_map))
    if scan_filter:
        scan_nums = [scan for scan in scan_nums if scan in scan_filter]

    for scan_num in scan_nums:
        raws = list(raw_map.get(scan_num, []))
        truths = list(truth_map.get(scan_num, []))
        if not raws:
            unpaired_truth.extend(truths)
            continue
        if not truths:
            unpaired_raw.extend(raws)
            continue

        candidate_pairs: list[tuple[int, str, str, Path, Path]] = []
        for raw_path in raws:
            for truth_path in truths:
                candidate_pairs.append(
                    (
                        path_distance(raw_path, truth_path),
                        manual_path_priority(truth_path),
                        str(raw_path),
                        str(truth_path),
                        raw_path,
                        truth_path,
                    )
                )
        candidate_pairs.sort()

        used_raw: set[Path] = set()
        used_truth: set[Path] = set()
        for distance, _truth_priority, _raw_key, _truth_key, raw_path, truth_path in candidate_pairs:
            if raw_path in used_raw or truth_path in used_truth:
                continue
            used_raw.add(raw_path)
            used_truth.add(truth_path)
            pairs.append(
                ScanPair(
                    dataset=dataset_name,
                    scan_num=scan_num,
                    raw_path=raw_path,
                    truth_path=truth_path,
                    pair_distance=distance,
                )
            )

        unpaired_raw.extend(path for path in raws if path not in used_raw)
        unpaired_truth.extend(path for path in truths if path not in used_truth)

    return pairs, unpaired_raw, unpaired_truth


def normalize_manual_label(label: str) -> str:
    normalized = str(label).strip()
    normalized = normalized.replace("(H2O)", "-H2O")
    normalized = normalized.replace("(NH3)", "-NH3")
    normalized = normalized.replace("(CO2)", "-CO2")
    normalized = normalized.replace("(CO)", "-CO")
    return re.sub(r"\s+", " ", normalized)


def normalize_prefix(prefix: str) -> str:
    prefix = str(prefix).strip()
    match = re.match(r"^([bcBC])(\d+)(.*)$", prefix)
    if not match:
        return prefix
    ion_type = match.group(1).lower()
    frag_len = str(int(match.group(2)))
    tail = match.group(3).replace(" ", "")

    def repl(loss_match: re.Match[str]) -> str:
        count = loss_match.group(1) or ""
        formula = LOSS_NORMALIZATIONS[loss_match.group(2).lower()]
        return f"-{count}{formula}"

    tail = re.sub(r"-(\d*)(h2o|nh3|co2|co)", repl, tail, flags=re.IGNORECASE)
    return f"{ion_type}{frag_len}{tail}"


def canonicalize_formula(text: Any) -> Optional[str]:
    raw = str(text or "").strip().replace(" ", "")
    if not raw:
        return None
    pos = 0
    items: dict[str, int] = defaultdict(int)
    while pos < len(raw):
        match = FORMULA_RE.match(raw, pos)
        if not match:
            return None
        element = match.group(1)
        count_text = match.group(2)
        if count_text in {"", "+", "-"}:
            count = 1 if count_text != "-" else -1
        else:
            try:
                count = int(count_text)
            except Exception:
                return None
        items[element] += count
        pos = match.end()
    parts: list[str] = []
    used: set[str] = set()
    for element in FORMULA_ORDER:
        count = items.get(element, 0)
        if count:
            parts.append(f"{element}{count}")
            used.add(element)
    for element in sorted(k for k, v in items.items() if v and k not in used):
        parts.append(f"{element}{items[element]}")
    return "".join(parts) if parts else None


def manual_prefix_and_charge(row: dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
    ion_type = str(row.get("Base Type") or row.get("Type") or "").strip().lower()
    pos = parse_float(row.get("Pos"))
    charge = parse_float(row.get("Charge"))
    label = normalize_manual_label(
        str(row.get("Name") or "")
        or (f"{ion_type}{int(pos)} {int(charge)}+" if ion_type in {"b", "c"} and pos is not None and charge is not None else "")
    )
    match = re.match(r"^([A-Za-z0-9+\-]+)\s+(\d+)\+$", label)
    if match:
        return normalize_prefix(match.group(1)), int(match.group(2))
    if ion_type in {"b", "c"} and pos is not None and charge is not None:
        return normalize_prefix(f"{ion_type}{int(pos)}"), int(charge)
    return None, None


def build_manual_context(truth_path: Path) -> dict[str, Any]:
    positives: dict[tuple[str, int, str], dict[str, Any]] = {}
    positives_missing_formula: dict[tuple[str, int, str], dict[str, Any]] = {}
    positive_by_prefix_charge: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    all_by_prefix_charge: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    with truth_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=2):
            ion_type = str(row.get("Base Type") or row.get("Type") or "").strip().lower()
            if ion_type not in {"b", "c"}:
                continue
            prefix, charge = manual_prefix_and_charge(row)
            if prefix is None or charge is None:
                continue
            record = {
                "source_path": str(truth_path),
                "source_row": row_idx,
                "name": str(row.get("Name") or "").strip(),
                "ion_type": ion_type,
                "pos": int(parse_float(row.get("Pos")) or 0),
                "charge": int(charge),
                "prefix": prefix,
                "formula": canonicalize_formula(row.get("Molecular Composition")),
                "matched": int(parse_float(row.get("Matched")) == 1.0),
                "selected": int(parse_float(row.get("Selected")) == 1.0),
                "molecular_composition": str(row.get("Molecular Composition") or "").strip(),
                "avg_ppm_error": parse_float(row.get("Avg PPM Error")),
                "ion_score": parse_float(row.get("IonScore")),
                "gof_confidence": parse_float(row.get("Gof Confidence")),
                "peaks_matched": parse_float(row.get("Peaks Matched")),
            }
            all_by_prefix_charge[(prefix, charge)].append(record)
            if not record["matched"]:
                continue
            positive_by_prefix_charge[(prefix, charge)].append(record)
            if record["formula"]:
                positives[(prefix, charge, str(record["formula"]))] = record
            else:
                missing_key = (prefix, charge, record["name"] or f"{prefix}^{charge}+")
                positives_missing_formula[missing_key] = record

    return {
        "positive_exact": positives,
        "positive_missing_formula": positives_missing_formula,
        "positive_by_prefix_charge": positive_by_prefix_charge,
        "all_by_prefix_charge": all_by_prefix_charge,
    }


def prediction_prefix(fragment: dict[str, Any]) -> Optional[str]:
    frag_id = str(fragment.get("frag_id") or "").strip()
    if not frag_id:
        return None
    variant_suffix = str(fragment.get("variant_suffix") or "").strip()
    if variant_suffix:
        frag_id = frag_id.replace(variant_suffix, "", 1)
    return normalize_prefix(frag_id)


def build_prediction_records(
    dataset: DatasetContext,
    scan_num: int,
    raw_path: Path,
    disulfide_variant_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    req = FragmentsRunRequest(
        filepath=str(raw_path.resolve()),
        scan=1,
        peptide=dataset.peptide,
        ion_types=["b", "c"],
        frag_min_charge=dataset.frag_min_charge,
        frag_max_charge=dataset.frag_max_charge,
        copies=dataset.copies,
        amidated=dataset.amidated,
        disulfide_bonds=dataset.disulfide_bonds,
        disulfide_map=dataset.disulfide_map,
        disulfide_variant_mode=disulfide_variant_mode,
    )
    result = quiet_call(_run_fragments_impl, req)

    deduped: dict[tuple[str, int, str], dict[str, Any]] = {}
    for item in result.get("fragments", []) or []:
        ion_type = str(item.get("ion_type") or "").strip().lower()
        if ion_type not in {"b", "c"}:
            continue
        prefix = prediction_prefix(item)
        charge = int(item.get("charge", 0) or 0)
        formula = canonicalize_formula(item.get("formula"))
        if prefix is None or charge <= 0 or not formula:
            continue
        record = {
            "dataset": dataset.name,
            "scan_num": scan_num,
            "raw_path": str(raw_path),
            "frag_id": str(item.get("frag_id") or ""),
            "label": str(item.get("label") or ""),
            "ion_type": ion_type,
            "frag_len": int(item.get("frag_len", 0) or 0),
            "charge": charge,
            "prefix": prefix,
            "formula": formula,
            "variant_suffix": str(item.get("variant_suffix") or ""),
            "obs_mz": parse_float(item.get("obs_mz")),
            "anchor_theory_mz": parse_float(item.get("anchor_theory_mz")),
            "anchor_ppm": parse_float(item.get("anchor_ppm")),
            "score": parse_float(item.get("score")),
            "selection_score": parse_float(item.get("selection_score")),
            "evidence_score": parse_float(item.get("evidence_score")),
            "css": parse_float(item.get("css")),
            "coverage": parse_float(item.get("coverage")),
            "match_count": int(item.get("match_count", 0) or 0),
            "correlation_coefficient": parse_float(item.get("correlation_coefficient")),
            "fit_score": parse_float(item.get("fit_score")),
            "pc_missing_peaks": parse_float(item.get("pc_missing_peaks")),
        }
        key = (prefix, charge, formula)
        current = deduped.get(key)
        current_score = (
            (current.get("selection_score") if current is not None else None)
            if current is not None
            else None
        )
        new_score = record.get("selection_score")
        if current is None or (new_score is not None and (current_score is None or new_score > current_score)):
            deduped[key] = record

    return sorted(
        deduped.values(),
        key=lambda row: (
            row["prefix"],
            row["charge"],
            row["formula"],
        ),
    ), result


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_pair(
    dataset: DatasetContext,
    pair: ScanPair,
    disulfide_variant_mode: str,
) -> dict[str, Any]:
    manual = build_manual_context(pair.truth_path)
    predictions, fragments_result = build_prediction_records(
        dataset,
        pair.scan_num,
        pair.raw_path,
        disulfide_variant_mode,
    )

    predicted_exact = {
        (row["prefix"], int(row["charge"]), str(row["formula"])): row for row in predictions
    }
    predicted_by_prefix_charge: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        predicted_by_prefix_charge[(str(row["prefix"]), int(row["charge"]))].append(row)

    manual_exact = manual["positive_exact"]
    manual_missing_formula = manual["positive_missing_formula"]
    manual_positive_by_prefix_charge = manual["positive_by_prefix_charge"]
    manual_all_by_prefix_charge = manual["all_by_prefix_charge"]

    tp_keys = sorted(set(manual_exact) & set(predicted_exact))
    fn_keys = sorted(set(manual_exact) - set(predicted_exact))
    fp_keys = sorted(set(predicted_exact) - set(manual_exact))

    matches: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    missing_formula_rows: list[dict[str, Any]] = []

    for key in tp_keys:
        manual_row = manual_exact[key]
        pred_row = predicted_exact[key]
        matches.append(
            {
                "dataset": dataset.name,
                "scan_num": pair.scan_num,
                "raw_path": str(pair.raw_path),
                "truth_path": str(pair.truth_path),
                "prefix": key[0],
                "charge": key[1],
                "formula": key[2],
                "manual_name": manual_row["name"],
                "manual_row": manual_row["source_row"],
                "pred_label": pred_row["label"],
                "pred_frag_id": pred_row["frag_id"],
                "pred_variant_suffix": pred_row["variant_suffix"],
                "pred_score": pred_row["score"],
                "pred_selection_score": pred_row["selection_score"],
                "pred_css": pred_row["css"],
                "pred_coverage": pred_row["coverage"],
            }
        )

    for key in fn_keys:
        manual_row = manual_exact[key]
        prefix_charge = (key[0], key[1])
        same_prefix_predictions = predicted_by_prefix_charge.get(prefix_charge, [])
        conflict_type = "wrong_formula" if same_prefix_predictions else "no_prefix_prediction"
        false_negatives.append(
            {
                "dataset": dataset.name,
                "scan_num": pair.scan_num,
                "raw_path": str(pair.raw_path),
                "truth_path": str(pair.truth_path),
                "prefix": key[0],
                "charge": key[1],
                "formula": key[2],
                "manual_name": manual_row["name"],
                "manual_row": manual_row["source_row"],
                "conflict_type": conflict_type,
                "same_prefix_predicted_labels_json": json_text([row["label"] for row in same_prefix_predictions]),
                "same_prefix_predicted_formulas_json": json_text([row["formula"] for row in same_prefix_predictions]),
            }
        )

    for key in fp_keys:
        pred_row = predicted_exact[key]
        prefix_charge = (key[0], key[1])
        same_prefix_manual_true = manual_positive_by_prefix_charge.get(prefix_charge, [])
        same_prefix_manual_all = manual_all_by_prefix_charge.get(prefix_charge, [])
        if any(row["formula"] for row in same_prefix_manual_true):
            conflict_type = "wrong_formula"
        elif any(not row["formula"] for row in same_prefix_manual_true):
            conflict_type = "manual_formula_missing"
        elif same_prefix_manual_all:
            conflict_type = "manual_negative"
        else:
            conflict_type = "absent_in_manual"
        false_positives.append(
            {
                "dataset": dataset.name,
                "scan_num": pair.scan_num,
                "raw_path": str(pair.raw_path),
                "truth_path": str(pair.truth_path),
                "prefix": key[0],
                "charge": key[1],
                "formula": key[2],
                "pred_label": pred_row["label"],
                "pred_frag_id": pred_row["frag_id"],
                "pred_variant_suffix": pred_row["variant_suffix"],
                "pred_score": pred_row["score"],
                "pred_selection_score": pred_row["selection_score"],
                "pred_css": pred_row["css"],
                "pred_coverage": pred_row["coverage"],
                "conflict_type": conflict_type,
                "same_prefix_manual_labels_json": json_text([row["name"] for row in same_prefix_manual_all]),
                "same_prefix_manual_formulas_json": json_text(
                    [row["formula"] for row in same_prefix_manual_true if row["formula"]]
                ),
            }
        )

    for missing_key, manual_row in sorted(manual_missing_formula.items()):
        prefix = missing_key[0]
        charge = missing_key[1]
        same_prefix_predictions = predicted_by_prefix_charge.get((prefix, charge), [])
        missing_formula_rows.append(
            {
                "dataset": dataset.name,
                "scan_num": pair.scan_num,
                "raw_path": str(pair.raw_path),
                "truth_path": str(pair.truth_path),
                "prefix": prefix,
                "charge": charge,
                "manual_name": manual_row["name"],
                "manual_row": manual_row["source_row"],
                "formula": "",
                "prefix_hit": int(bool(same_prefix_predictions)),
                "same_prefix_predicted_labels_json": json_text([row["label"] for row in same_prefix_predictions]),
                "same_prefix_predicted_formulas_json": json_text([row["formula"] for row in same_prefix_predictions]),
            }
        )

    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)

    summary = {
        "dataset": dataset.name,
        "scan_num": pair.scan_num,
        "raw_path": str(pair.raw_path),
        "truth_path": str(pair.truth_path),
        "pair_distance": int(pair.pair_distance),
        "disulfide_variant_mode": disulfide_variant_mode,
        "predicted_selected_count": len(predictions),
        "manual_positive_exact_count": len(manual_exact),
        "manual_positive_missing_formula_count": len(manual_missing_formula),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_wrong_formula": sum(row["conflict_type"] == "wrong_formula" for row in false_positives),
        "fp_manual_formula_missing": sum(row["conflict_type"] == "manual_formula_missing" for row in false_positives),
        "fp_manual_negative": sum(row["conflict_type"] == "manual_negative" for row in false_positives),
        "fp_absent_in_manual": sum(row["conflict_type"] == "absent_in_manual" for row in false_positives),
        "fn_wrong_formula": sum(row["conflict_type"] == "wrong_formula" for row in false_negatives),
        "fn_no_prefix_prediction": sum(row["conflict_type"] == "no_prefix_prediction" for row in false_negatives),
        "manual_missing_formula_prefix_hit": sum(row["prefix_hit"] == 1 for row in missing_formula_rows),
        "manual_missing_formula_prefix_miss": sum(row["prefix_hit"] == 0 for row in missing_formula_rows),
        "precursor_calibration_applied": int(
            bool((fragments_result.get("precursor") or {}).get("calibration_applied"))
        ),
    }

    return {
        "summary": summary,
        "matches": matches,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "missing_formula_rows": missing_formula_rows,
        "predictions": predictions,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset_contexts(
    sample_root: Path, dataset_filters: set[str]
) -> tuple[list[DatasetContext], list[dict[str, Any]]]:
    meta_candidates = discover_meta_candidates(sample_root)
    contexts: list[DatasetContext] = []
    skipped: list[dict[str, Any]] = []

    for dataset_root in sorted(path for path in sample_root.iterdir() if path.is_dir()):
        if dataset_filters and dataset_root.name not in dataset_filters:
            continue
        meta = choose_meta_candidate(dataset_root, sample_root, meta_candidates)
        if meta is None:
            skipped.append(
                {
                    "dataset": dataset_root.name,
                    "reason": "meta_not_found",
                    "root": str(dataset_root),
                }
            )
            continue
        contexts.append(
            DatasetContext(
                name=dataset_root.name,
                root=dataset_root,
                meta_file=meta.path,
                peptide=meta.peptide,
                copies=meta.copies,
                amidated=meta.amidated,
                disulfide_bonds=meta.disulfide_bonds,
                disulfide_map=meta.disulfide_map,
                frag_min_charge=meta.frag_min_charge,
                frag_max_charge=meta.frag_max_charge,
            )
        )
    return contexts, skipped


def summarize_by_group(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)

    summaries: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        tp = sum(int(item["tp"]) for item in items)
        fp = sum(int(item["fp"]) for item in items)
        fn = sum(int(item["fn"]) for item in items)
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        summaries.append(
            {
                group_key: key,
                "scans": len(items),
                "predicted_selected_count": sum(int(item["predicted_selected_count"]) for item in items),
                "manual_positive_exact_count": sum(int(item["manual_positive_exact_count"]) for item in items),
                "manual_positive_missing_formula_count": sum(
                    int(item["manual_positive_missing_formula_count"]) for item in items
                ),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fp_wrong_formula": sum(int(item["fp_wrong_formula"]) for item in items),
                "fn_wrong_formula": sum(int(item["fn_wrong_formula"]) for item in items),
                "manual_missing_formula_prefix_hit": sum(
                    int(item["manual_missing_formula_prefix_hit"]) for item in items
                ),
            }
        )
    return summaries


def main() -> int:
    args = parse_args()
    sample_root = Path(args.sample_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_filters = set(args.dataset or [])
    scan_filter = set(args.scan or [])

    contexts, skipped_datasets = build_dataset_contexts(sample_root, dataset_filters)

    scan_pairs: list[ScanPair] = []
    unpaired_raw: list[dict[str, Any]] = []
    unpaired_truth: list[dict[str, Any]] = []
    for context in contexts:
        raw_map = discover_raw_files(context.root)
        truth_map = discover_truth_files(context.root)
        pairs, raw_only, truth_only = pair_scan_files(
            context.name, raw_map, truth_map, scan_filter
        )
        scan_pairs.extend(pairs)
        unpaired_raw.extend(
            {
                "dataset": context.name,
                "scan_num": extract_scan_num(path),
                "raw_path": str(path),
            }
            for path in raw_only
        )
        unpaired_truth.extend(
            {
                "dataset": context.name,
                "scan_num": extract_scan_num(path),
                "truth_path": str(path),
            }
            for path in truth_only
        )

    pair_rows = [
        {
            "dataset": pair.dataset,
            "scan_num": pair.scan_num,
            "raw_path": str(pair.raw_path),
            "truth_path": str(pair.truth_path),
            "pair_distance": pair.pair_distance,
        }
        for pair in sorted(scan_pairs, key=lambda row: (row.dataset, row.scan_num, str(row.raw_path)))
    ]

    per_scan_summary: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    missing_formula_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    context_by_name = {context.name: context for context in contexts}
    for pair in sorted(scan_pairs, key=lambda row: (row.dataset, row.scan_num, str(row.raw_path))):
        evaluation = evaluate_pair(
            context_by_name[pair.dataset],
            pair,
            args.disulfide_variant_mode,
        )
        per_scan_summary.append(evaluation["summary"])
        matches.extend(evaluation["matches"])
        false_negatives.extend(evaluation["false_negatives"])
        false_positives.extend(evaluation["false_positives"])
        missing_formula_rows.extend(evaluation["missing_formula_rows"])
        predictions.extend(evaluation["predictions"])

    dataset_summary = summarize_by_group(per_scan_summary, "dataset")
    overall_tp = sum(int(row["tp"]) for row in per_scan_summary)
    overall_fp = sum(int(row["fp"]) for row in per_scan_summary)
    overall_fn = sum(int(row["fn"]) for row in per_scan_summary)
    overall_precision, overall_recall, overall_f1 = precision_recall_f1(
        overall_tp, overall_fp, overall_fn
    )

    summary = {
        "sample_root": str(sample_root),
        "output_dir": str(output_dir),
        "disulfide_variant_mode": args.disulfide_variant_mode,
        "datasets_requested": sorted(dataset_filters),
        "scans_requested": sorted(scan_filter),
        "datasets_processed": [context.name for context in contexts],
        "datasets_skipped": skipped_datasets,
        "scan_pair_count": len(scan_pairs),
        "unpaired_raw_count": len(unpaired_raw),
        "unpaired_truth_count": len(unpaired_truth),
        "predicted_selected_count": len(predictions),
        "manual_positive_exact_count": sum(
            int(row["manual_positive_exact_count"]) for row in per_scan_summary
        ),
        "manual_positive_missing_formula_count": sum(
            int(row["manual_positive_missing_formula_count"]) for row in per_scan_summary
        ),
        "tp": overall_tp,
        "fp": overall_fp,
        "fn": overall_fn,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "fp_wrong_formula": sum(int(row["fp_wrong_formula"]) for row in per_scan_summary),
        "fn_wrong_formula": sum(int(row["fn_wrong_formula"]) for row in per_scan_summary),
        "manual_missing_formula_prefix_hit": sum(
            int(row["manual_missing_formula_prefix_hit"]) for row in per_scan_summary
        ),
        "manual_missing_formula_prefix_miss": sum(
            int(row["manual_missing_formula_prefix_miss"]) for row in per_scan_summary
        ),
        "pairing_rows": pair_rows,
        "dataset_summary": dataset_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scan_pairs.csv", pair_rows)
    write_csv(output_dir / "per_scan_summary.csv", per_scan_summary)
    write_csv(output_dir / "per_dataset_summary.csv", dataset_summary)
    write_csv(output_dir / "algorithm_predictions.csv", predictions)
    write_csv(output_dir / "exact_matches.csv", matches)
    write_csv(output_dir / "false_negatives.csv", false_negatives)
    write_csv(output_dir / "false_positives.csv", false_positives)
    write_csv(output_dir / "manual_missing_formula.csv", missing_formula_rows)
    write_csv(output_dir / "unpaired_raw.csv", unpaired_raw)
    write_csv(output_dir / "unpaired_truth.csv", unpaired_truth)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "disulfide_variant_mode": args.disulfide_variant_mode,
                "scan_pair_count": len(scan_pairs),
                "tp": overall_tp,
                "fp": overall_fp,
                "fn": overall_fn,
                "precision": round(overall_precision, 6),
                "recall": round(overall_recall, 6),
                "f1": round(overall_f1, 6),
                "manual_positive_missing_formula_count": summary[
                    "manual_positive_missing_formula_count"
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
