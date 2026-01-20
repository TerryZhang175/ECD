from __future__ import annotations

from typing import Optional

import pyteomics.mass as ms

import personalized_config as cfg


def parse_custom_sequence(peptide: str) -> list[tuple[str, list[str]]]:
    """
    Parse a peptide string like 'AC[O]DEK' into per-residue (aa, [formula_mods]).

    Notes:
    - Bracket contents are treated as elemental formulas understood by pyteomics (e.g. "O", "HPO3", "C2H3NO").
    - This parser intentionally does NOT support raw mass deltas (e.g. [+15.99]) because isotope modeling needs formulas.
    """
    peptide = peptide.strip()
    residues: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(peptide):
        aa = peptide[i]
        if aa.isspace():
            i += 1
            continue
        if not ("A" <= aa <= "Z"):
            raise ValueError(
                f"Unsupported character '{aa}' in PEPTIDE. "
                "Use an uppercase amino-acid sequence with optional '[FORMULA]' mods."
            )
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
        try:
            comp += ms.std_aa_comp[aa]
        except KeyError as e:
            raise ValueError(f"Unsupported residue '{aa}' in PEPTIDE.") from e
        for mod in mods:
            comp += ms.Composition(mod)
    return comp


def ion_composition_from_sequence(
    residues: list[tuple[str, list[str]]],
    ion_type: str,
    frag_len: int,
    amidated: bool,
) -> tuple[str, ms.Composition]:
    n = len(residues)
    if frag_len <= 0 or frag_len >= n:
        raise ValueError(f"Invalid fragment length {frag_len} for peptide length {n}.")
    if ion_type not in ms.std_ion_comp:
        raise ValueError(f"Unsupported ion_type '{ion_type}'. Try one of: {sorted(ms.std_ion_comp.keys())}")

    # pyteomics fragment ion comps are defined relative to the corresponding neutral peptide composition (residues + H2O).
    if ion_type.startswith(("a", "b", "c")):
        frag_res = residue_range_composition(residues, 0, frag_len)
        name = f"{ion_type}{frag_len}"
        has_c_term = False
    else:
        frag_res = residue_range_composition(residues, n - frag_len, n)
        name = f"{ion_type}{frag_len}"
        has_c_term = True

    pep_comp = frag_res + ms.Composition("H2O")
    if amidated and has_c_term:
        pep_comp += ms.Composition(cfg.AMIDATION_FORMULA)

    ion_comp = pep_comp + ms.std_ion_comp[ion_type]
    return name, ion_comp


def ion_series(ion_type: str) -> str:
    """
    Return the base ion series letter for a pyteomics ion type string.
    Examples: 'c' -> 'c', 'c-dot' -> 'c', 'z-H2O' -> 'z'.
    """
    if not ion_type:
        return ""
    return ion_type.split("-", 1)[0][:1]


def neutral_loss_label(n: int, formula: str) -> str:
    if n <= 0:
        return ""
    if n == 1:
        return f"-{formula}"
    return f"-{n}{formula}"


def neutral_loss_variants(comp: ms.Composition, ion_series_letter: str) -> list[tuple[str, ms.Composition]]:
    """
    Generate neutral-loss variants of a fragment composition.
    Returns list of (suffix, composition), including the neutral (no-loss) variant.
    """
    variants: list[tuple[str, ms.Composition]] = [("", comp)]
    if not cfg.ENABLE_NEUTRAL_LOSSES:
        return variants
    if ion_series_letter not in set(cfg.NEUTRAL_LOSS_ION_SERIES):
        return variants

    max_h2o = int(max(0, cfg.NEUTRAL_LOSS_MAX_H2O))
    max_nh3 = int(max(0, cfg.NEUTRAL_LOSS_MAX_NH3))
    max_co = int(max(0, cfg.NEUTRAL_LOSS_MAX_CO))
    max_co2 = int(max(0, cfg.NEUTRAL_LOSS_MAX_CO2))
    if max_h2o == 0 and max_nh3 == 0 and max_co == 0 and max_co2 == 0:
        return variants

    h2o = ms.Composition("H2O")
    nh3 = ms.Composition("NH3")
    co = ms.Composition("CO")
    co2 = ms.Composition("CO2")

    for n_h2o in range(1, max_h2o + 1):
        suffix = neutral_loss_label(n_h2o, "H2O")
        try:
            new_comp = comp - (h2o * n_h2o)
        except Exception:
            continue
        variants.append((suffix, new_comp))

    for n_nh3 in range(1, max_nh3 + 1):
        suffix = neutral_loss_label(n_nh3, "NH3")
        try:
            new_comp = comp - (nh3 * n_nh3)
        except Exception:
            continue
        variants.append((suffix, new_comp))

    for n_co in range(1, max_co + 1):
        suffix = neutral_loss_label(n_co, "CO")
        try:
            new_comp = comp - (co * n_co)
        except Exception:
            continue
        variants.append((suffix, new_comp))

    for n_co2 in range(1, max_co2 + 1):
        suffix = neutral_loss_label(n_co2, "CO2")
        try:
            new_comp = comp - (co2 * n_co2)
        except Exception:
            continue
        variants.append((suffix, new_comp))

    return variants


def neutral_loss_columns(loss_suffix: str) -> dict[str, int]:
    cols = {"H2O": 0, "NH3": 0, "CO": 0, "CO2": 0, "2H2O": 0, "2NH3": 0}
    s = str(loss_suffix or "").strip()
    if not s:
        return cols
    if s == "-H2O":
        cols["H2O"] = 1
    elif s == "-NH3":
        cols["NH3"] = 1
    elif s == "-CO":
        cols["CO"] = 1
    elif s == "-CO2":
        cols["CO2"] = 1
    elif s == "-2H2O":
        cols["2H2O"] = 1
    elif s == "-2NH3":
        cols["2NH3"] = 1
    return cols


def apply_neutral_loss(comp: ms.Composition, formula: str, count: int) -> ms.Composition:
    if not formula or int(count) <= 0:
        return comp
    if formula not in {"H2O", "NH3", "CO", "CO2"}:
        raise ValueError(f"Unsupported neutral loss formula '{formula}'.")
    loss_comp = ms.Composition(formula)
    return comp - (loss_comp * int(count))
