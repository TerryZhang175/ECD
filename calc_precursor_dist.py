#!/usr/bin/env python3
"""
Standalone script to calculate precursor theoretical isotope distribution.

Only requires: numpy, pyteomics, matplotlib (optional, for plotting).
No dependency on unidec / IsoDec.

Usage:
    python3 calc_precursor_dist.py
"""
from __future__ import annotations

import numpy as np
import pyteomics.mass as ms

# ── User parameters (edit these) ────────────────────────────────────────────
PEPTIDE = "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY"
COPIES = 2                  # 1=monomer, 2=dimer
AMIDATED = True             # C-terminal amidation per copy
AMIDATION_FORMULA = "H1N1O-1"
DISULFIDE_BONDS = 2         # total disulfide bonds in complex (each removes H2)

MIN_CHARGE = 1
MAX_CHARGE = 10

ADDUCT_MASS = 1.007276467   # proton mass (Da)
MASS_DIFF_C = 1.0033        # ~C13-C12 mass difference (Da)
ISOLEN = 128                # FFT isotope distribution length
REL_INTENSITY_CUTOFF = 0.01 # drop peaks < 1% of max

ENABLE_H_TRANSFER = True
H_TRANSFER_MASS = 1.007276467
# ────────────────────────────────────────────────────────────────────────────


# ── Sequence parsing (from personalized_sequence.py) ────────────────────────
def parse_custom_sequence(peptide: str) -> list[tuple[str, list[str]]]:
    peptide = peptide.strip()
    residues: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(peptide):
        aa = peptide[i]
        if aa.isspace():
            i += 1
            continue
        if not ("A" <= aa <= "Z"):
            raise ValueError(f"Unsupported character '{aa}' in PEPTIDE.")
        i += 1
        mods: list[str] = []
        while i < len(peptide) and peptide[i] == "[":
            j = peptide.find("]", i + 1)
            if j == -1:
                raise ValueError('Unclosed "[" in PEPTIDE.')
            mod = peptide[i + 1 : j].strip()
            if not mod:
                raise ValueError("Empty [] modification in PEPTIDE.")
            mods.append(mod)
            i = j + 1
        residues.append((aa, mods))
    return residues


def residue_range_composition(
    residues: list[tuple[str, list[str]]], start: int, end: int
) -> ms.Composition:
    comp = ms.Composition()
    for aa, mods in residues[start:end]:
        comp += ms.std_aa_comp[aa]
        for mod in mods:
            comp += ms.Composition(mod)
    return comp


def get_precursor_composition(residues: list[tuple[str, list[str]]]) -> ms.Composition:
    monomer_base = residue_range_composition(residues, 0, len(residues)) + ms.Composition("H2O")
    complex_comp = monomer_base * int(COPIES)
    if AMIDATED:
        complex_comp += ms.Composition(AMIDATION_FORMULA) * int(COPIES)
    if int(DISULFIDE_BONDS) > 0:
        complex_comp -= ms.Composition(f"H{2 * int(DISULFIDE_BONDS)}")
    return complex_comp


# ── FFT isotope distribution (from personalized_isotopes.py) ────────────────
_FFT_CACHE: dict[int, tuple] = {}


def _get_fft_tables(length: int):
    cached = _FFT_CACHE.get(length)
    if cached is not None:
        return cached
    buf = np.zeros(length, dtype=float)
    h = np.append([1.0, 0.00015, 0.0, 0.0], buf)
    c = np.append([1.0, 0.011,   0.0, 0.0], buf)
    n = np.append([1.0, 0.0037,  0.0, 0.0], buf)
    o = np.append([1.0, 0.0004,  0.002, 0.0], buf)
    s = np.append([1.0, 0.0079,  0.044, 0.0], buf)
    result = tuple(np.fft.rfft(x).astype(np.complex128) for x in (c, h, n, o, s))
    _FFT_CACHE[length] = result
    return result


def isojim(isolist: np.ndarray, length: int = ISOLEN) -> np.ndarray:
    isolist = np.asarray(isolist, dtype=int).ravel()
    if isolist.size < 5:
        isolist = np.pad(isolist, (0, 5 - isolist.size))
    numc, numh, numn, numo, nums = (int(isolist[i]) for i in range(5))
    cft, hft, nft, oft, sft = _get_fft_tables(length)
    allft = cft**numc * hft**numh * nft**numn * oft**numo * sft**nums
    allift = np.abs(np.fft.irfft(allft))
    maxval = float(np.max(allift)) if allift.size else 0.0
    if maxval > 0:
        allift /= maxval
    return allift[:length]


# ── Theoretical m/z (from personalized_theory.py) ──────────────────────────
def theoretical_isodist_from_comp(comp: ms.Composition, charge: int) -> np.ndarray:
    mono_mass = float(comp.mass())
    isolist = np.array([
        comp.get("C", 0), comp.get("H", 0), comp.get("N", 0),
        comp.get("O", 0), comp.get("S", 0),
    ], dtype=int)
    if np.any(isolist < 0):
        raise ValueError(f"Negative elemental composition: {isolist}")

    intensities = np.asarray(isojim(isolist, length=ISOLEN), dtype=float)
    isotope_index = np.arange(len(intensities), dtype=float)
    masses = mono_mass + isotope_index * MASS_DIFF_C
    mz = (masses + charge * ADDUCT_MASS) / abs(charge)
    dist = np.column_stack([mz, intensities])
    max_int = float(np.max(dist[:, 1]))
    return dist[dist[:, 1] >= max_int * REL_INTENSITY_CUTOFF].copy()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    residues = parse_custom_sequence(PEPTIDE)
    comp = get_precursor_composition(residues)

    print("=" * 70)
    print("Precursor Isotope Distribution Calculator")
    print("=" * 70)
    print(f"  Peptide:    {PEPTIDE}")
    print(f"  Copies:     {COPIES}")
    print(f"  Amidated:   {AMIDATED}")
    print(f"  S-S bonds:  {DISULFIDE_BONDS}")
    print()

    elem_str = ", ".join(f"{k}={v}" for k, v in sorted(comp.items()) if v != 0)
    mono_mass = float(comp.mass())
    print(f"  Composition:      {elem_str}")
    print(f"  Monoisotopic M:   {mono_mass:.4f} Da")
    print()

    state_shifts = [("neutral", 0)]
    if ENABLE_H_TRANSFER:
        state_shifts = [
            ("neutral", 0),
            ("+H",  1), ("+2H", 2),
            ("-H", -1), ("-2H", -2),
        ]

    all_dists: list[tuple[int, str, np.ndarray]] = []

    for z in range(MIN_CHARGE, MAX_CHARGE + 1):
        try:
            base_dist = theoretical_isodist_from_comp(comp, z)
        except ValueError:
            continue
        if base_dist.size == 0:
            continue

        for state_label, h_shift in state_shifts:
            if h_shift:
                dist = base_dist.copy()
                dist[:, 0] += (h_shift * H_TRANSFER_MASS) / z
            else:
                dist = base_dist

            anchor_idx = int(np.argmax(dist[:, 1]))
            anchor_mz = float(dist[anchor_idx, 0])
            mz_range = (float(dist[0, 0]), float(dist[-1, 0]))

            all_dists.append((z, state_label, dist))

            if state_label == "neutral":
                print(f"  z={z:>2}+  anchor m/z = {anchor_mz:>12.4f}   "
                      f"range [{mz_range[0]:.4f} – {mz_range[1]:.4f}]   "
                      f"peaks = {len(dist)}")

    print()
    print(f"  Total theoretical envelopes: {len(all_dists)}")
    print("=" * 70)

    # ── Optional plot ───────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed; skipping plot)")
        return

    fig, axes = plt.subplots(
        min(MAX_CHARGE - MIN_CHARGE + 1, 5), 1,
        figsize=(12, 2.5 * min(MAX_CHARGE - MIN_CHARGE + 1, 5)),
        sharex=False,
    )
    if not hasattr(axes, "__len__"):
        axes = [axes]

    plot_charges = list(range(MIN_CHARGE, min(MIN_CHARGE + len(axes), MAX_CHARGE + 1)))
    for ax, z in zip(axes, plot_charges):
        for charge, state_label, dist in all_dists:
            if charge != z:
                continue
            if state_label == "neutral":
                ax.stem(dist[:, 0], dist[:, 1], linefmt="C0-", markerfmt="C0o",
                        basefmt=" ", label=f"z={z}+ neutral")
            elif state_label in ("+H", "-H"):
                ax.stem(dist[:, 0], dist[:, 1], linefmt="C1--", markerfmt="C1^",
                        basefmt=" ", label=f"z={z}+ {state_label}")
        ax.set_ylabel("Rel. intensity")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(f"Charge state z = {z}+")

    axes[-1].set_xlabel("m/z")
    fig.suptitle(f"Precursor isotope distributions – {PEPTIDE[:20]}…", fontsize=12)
    fig.tight_layout()
    plt.savefig("precursor_dist.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to precursor_dist.png")
    plt.show()


if __name__ == "__main__":
    main()
