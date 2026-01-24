from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np

_SPLIT_RE = re.compile(r"[,\s]+")
_COMMENT_PREFIXES = ("#", ";", "//")


def _is_comment_or_empty(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith(_COMMENT_PREFIXES)


def header_test(path: str) -> int:
    """
    Count header lines before the first fully numeric data line.
    """
    header = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if _is_comment_or_empty(line):
                    header += 1
                    continue
                parts = [p for p in _SPLIT_RE.split(line.strip()) if p]
                if len(parts) < 2:
                    header += 1
                    continue
                ok = True
                for part in parts:
                    try:
                        float(part)
                    except ValueError:
                        ok = False
                        break
                if ok:
                    break
                header += 1
    except OSError as exc:
        print("Failed header test:", exc)
        header = 0
    if header > 0:
        print("Header Length:", header)
    return int(header)


def _iter_xy_lines(path: str, skip: int) -> Iterable[tuple[float, float]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            if idx < skip:
                continue
            if _is_comment_or_empty(line):
                continue
            parts = [p for p in _SPLIT_RE.split(line.strip()) if p]
            if len(parts) < 2:
                continue
            try:
                mz = float(parts[0])
                inten = float(parts[1])
            except ValueError:
                continue
            yield mz, inten


def load_xy_data(path: str) -> np.ndarray:
    skip = header_test(path)
    data = list(_iter_xy_lines(path, skip=skip))
    if not data:
        raise ValueError(f"No numeric (mz, intensity) data found in {path}")
    return np.asarray(data, dtype=float)


class SingleScanImporter:
    def __init__(self, filename: str):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"File not found: {filename}")
        self._file_path = filename
        self.ext = os.path.splitext(filename)[1].lower()
        self.scans = [1]
        self.times = [0.0]
        self.scan_range = [1, 1]
        self.centroided = False
        self.polarity = "Positive"
        self.scan_number = 1
        self.injection_time = 1
        self.data = None

    def load_data(self) -> np.ndarray:
        if self.data is None:
            self.data = load_xy_data(self._file_path)
        return self.data

    def get_single_scan(self, scan=None) -> np.ndarray:
        return self.load_data()

    def get_all_scans(self) -> list[np.ndarray]:
        return [self.load_data()]

    def get_avg_scan(self, scan_range=None, time_range=None) -> np.ndarray:
        return self.load_data()

    def grab_centroid_data(self, scan=None) -> np.ndarray:
        return self.load_data()


class ImporterFactory:
    recognized_types = [".txt", ".dat", ".csv"]

    @staticmethod
    def create_importer(file_path: str):
        ending = os.path.splitext(file_path)[1].lower()
        if ending in ImporterFactory.recognized_types:
            return SingleScanImporter(file_path)
        raise IOError(f"Unsupported file type: {ending} ({file_path})")
