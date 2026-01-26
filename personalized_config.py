from __future__ import annotations

from typing import Optional

try:
    from unidec.modules.unidecstructure import IsoDecConfig
    from unidec.IsoDec.match import (
        calculate_cosinesimilarity as isodec_calculate_cosinesimilarity,
        find_matches as isodec_find_matches,
        find_matched_intensities as isodec_find_matched_intensities,
        make_shifted_peak as isodec_make_shifted_peak,
    )
except Exception:
    try:
        from personalized_isodec import (
            IsoDecConfig,
            calculate_cosinesimilarity as isodec_calculate_cosinesimilarity,
            find_matches as isodec_find_matches,
            find_matched_intensities as isodec_find_matched_intensities,
            make_shifted_peak as isodec_make_shifted_peak,
        )
    except Exception:
        IsoDecConfig = None
        isodec_calculate_cosinesimilarity = None
        isodec_find_matches = None
        isodec_find_matched_intensities = None
        isodec_make_shifted_peak = None


filepath = '/Users/terry/Downloads/WT 2n5 ConA ECDRE34.txt'
# What to generate/plot:
# - "precursor": precursor charge/state inspection and lock-mass calibration
# - "charge_reduced": charge-reduced precursor search (ECD/ETD)
# - "fragments": peptide backbone fragments (b/y/c/z for ECD-style MS/MS)
# - "complex_fragments": monomer + fragment non-covalent complexes
# - "diagnose": detailed diagnostics for a specific fragment ion
# - "raw": plot raw spectrum only (no preprocessing)
PLOT_MODE = "fragments"  # options: "precursor", "charge_reduced", "fragments", "complex_fragments", "diagnose", "raw"
SCAN = 1
ENABLE_CENTROID = False  # Global toggle for centroid usage (import + local re-centroiding).

# Optional: focus on an m/z region of interest.
# Set to None to use the full scan.
# Examples:
#   MZ_MIN, MZ_MAX = 900, 1100
#   MZ_MIN, MZ_MAX = 300, 2000
MZ_MIN = None
MZ_MAX = None

# Peptide sequence. Supports chemical-formula mods in brackets, e.g.:
# - Oxidation: "M[O]"
# - Phospho: "S[HPO3]" (equivalent to PO3H)
# - Carbamidomethyl (IAA): "C[C2H3NO]"
# Bracket contents are interpreted as an elemental formula (not a mass delta).
PEPTIDE = "KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY"
COPIES = 2  # 1=monomer, 2=dimer (two copies of the same peptide)
AMIDATED = True  # C-terminal amidation (adds HN, removes O; delta = H1N1O-1) per copy
DISULFIDE_BONDS = 2  # total disulfide bonds in the complex (each removes H2, ~-2.01565 Da per bond)
# Define disulfide bond pairs (1-based indices)
# Example: For KCNT... sequence, Cys at positions 2 and 7
DISULFIDE_MAP = [(2, 7)]
# For dimer inter-chain bonds, can define cross-chain mode (requires subsequent logic support)
# DISULFIDE_MAP = [("A2", "B2"), ("A7", "B7")]
# Example (amidated disulfide-linked dimer): COPIES=2, AMIDATED=True, DISULFIDE_BONDS=2

# Internal disulfides per monomer
INTERNAL_DISULFIDES_PER_MONOMER = DISULFIDE_BONDS // COPIES

ION_TYPES = ("b", "y", "c", "z-dot")  # For ECD you may want ("b","y","c","z-dot") depending on your annotation
FRAG_MIN_CHARGE = 1
FRAG_MAX_CHARGE = 6
MATCH_TOL_PPM = 20
MIN_OBS_REL_INT = 0.0
MAX_PLOTTED_FRAGMENTS = 40
LABEL_MATCHES = False
ANCHOR_TOP_N = 3
ANCHOR_MIN_MATCHES = 1
MIN_COSINE = 0.70  # minimum cosine similarity threshold for match acceptance
FRAG_ANCHOR_CENTROID_WINDOW_DA = 0.2

# Hydrogen-transfer handling (ExD/ECD-style). Uses H+ mass (not H atom mass).
# Only enabled for c/z by default; accept transfer only if cosine similarity
# improves by >= 20% vs the neutral (no-transfer) model.
ENABLE_H_TRANSFER = True
H_TRANSFER_MASS = 1.007276467
H_TRANSFER_ION_TYPES_1H = ("c", "z")
H_TRANSFER_ION_TYPES_2H = ("c", "z")
H_TRANSFER_MIN_REL_IMPROVEMENT = 0.20  # 0.20 = +20% vs neutral cosine score

# Neutral losses (configurable). Single-loss candidates only (no mixed-loss combinations).
ENABLE_NEUTRAL_LOSSES = True
NEUTRAL_LOSS_ION_SERIES = ("b", "y", "c", "z")
NEUTRAL_LOSS_MAX_H2O = 2  # 0, 1, 2 allowed
NEUTRAL_LOSS_MAX_NH3 = 2  # 0, 1, 2 allowed
NEUTRAL_LOSS_MAX_CO = 1   # 0, 1 allowed
NEUTRAL_LOSS_MAX_CO2 = 1  # 0, 1 allowed

# Hill-climb centroiding (peakdetect) for fragments mode.
# Uses UniDec's peakdetect with a ppm window to re-centroid spectra.
ENABLE_HILL_CENTROID = True
HILL_CENTROID_WINDOW = 10
HILL_CENTROID_THRESHOLD = 0.0
HILL_CENTROID_PPM = None  # if None, uses MATCH_TOL_PPM
HILL_CENTROID_NORM = True

# Fragment-driven intensity cap stripping:
# 1) Generate theoretical fragment ions (incl. neutral losses + H-transfer shifts).
# 2) For each candidate, take the highest observed peak within MATCH_TOL_PPM of its anchor m/z.
# 3) Let cap = max of those observed intensities.
# 4) Remove all peaks with intensity > cap (reduces dynamic-range domination by precursor/charge-reduced peaks).
ENABLE_FRAGMENT_INTENSITY_CAP = False
FRAGMENT_INTENSITY_CAP_MZ_MIN = 300.0
FRAGMENT_INTENSITY_CAP_MZ_MAX = 2000.0
FRAGMENT_INTENSITY_CAP_TOL_PPM = None  # if None, uses MATCH_TOL_PPM
FRAGMENT_INTENSITY_CAP_MIN_HITS = 1  # require at least this many non-zero windows to activate
FRAGMENT_INTENSITY_CAP_VERBOSE = True

# Diagnostics: set these to inspect why an expected ion was not selected.
# Examples:
#   DIAGNOSE_ION_SPEC = "c7^2+"
#   DIAGNOSE_ION_SPEC = "z12-2H2O^3+"
#   DIAGNOSE_ION_SPEC = "z-dot12-CO"  # will scan charge range if no ^z+ suffix is present
DIAGNOSE_ION_SPEC = "b4-2H2O^1+"
# Hydrogen transfer degree (H+). Use an integer in {-2,-1,0,1,2}.
# Set to 0 to enable automatic selection using fragments mode's mixture model
DIAGNOSE_H_TRANSFER = 0
# If ion spec has no charge, scan FRAG_MIN_CHARGE..FRAG_MAX_CHARGE and report all.
DIAGNOSE_SCAN_CHARGES = True
DIAGNOSE_SHOW_PLOT = True
DIAGNOSE_MAX_TABLE_ROWS = 12
DIAGNOSE_EXPORT_CSV = True
# If None, writes to `ECD/diagnose_outputs/` with an auto filename.
DIAGNOSE_CSV_SUMMARY_PATH = None
DIAGNOSE_CSV_PEAKS_PATH = None

# CSV export for normal matching modes (e.g., PLOT_MODE="fragments").
EXPORT_FRAGMENTS_CSV = True
# If None, writes to `ECD/match_outputs/` with an auto filename.
FRAGMENTS_CSV_SUMMARY_PATH = None
FRAGMENTS_CSV_PEAKS_PATH = None

# CSV export for charge-reduced precursor mode.
CHARGE_REDUCED_EXPORT_CSV = True
CHARGE_REDUCED_CSV_SUMMARY_PATH = None
CHARGE_REDUCED_CSV_PEAKS_PATH = None

# IsoDec-style false-positive suppression rules (preferred over ad-hoc gates).
ENABLE_ISODEC_RULES = True
ISODEC_MINPEAKS = 3
ISODEC_CSS_THRESH = 0.70
ISODEC_MIN_AREA_COVERED = 0.20
ISODEC_MZ_WINDOW_LB = -1.05
ISODEC_MZ_WINDOW_UB = 4.05
ISODEC_PLUSONE_INT_WINDOW_LB = 0.10
ISODEC_PLUSONE_INT_WINDOW_UB = 0.60
ISODEC_MINUSONE_AS_ZERO = True
ISODEC_VERBOSE = False
ISODEC_USE_AREA_COVERED = True

ISOLEN = 128
ADDUCT_MASS = 1.007276467  # proton mass for positive-mode m/z conversion
MASS_DIFF_C = 1.0033  # ~C13-C12 mass difference (Da)
AMIDATION_FORMULA = "H1N1O-1"

REL_INTENSITY_CUTOFF = 0.01

# Precursor mode settings.
PRECURSOR_MIN_CHARGE = 1
PRECURSOR_MAX_CHARGE = 10
PRECURSOR_WINDOW_DA = 5.1
PRECURSOR_SEARCH_ITERATIONS = 5
PRECURSOR_MATCH_TOL_PPM = 80  # separate ppm tolerance for precursor mode (independent of MATCH_TOL_PPM)
ENABLE_LOCK_MASS = True
PRECURSOR_CHAIN_TO_FRAGMENTS = True

# Charge-reduced precursor settings.
CR_MIN_CHARGE = 1
CR_MAX_CHARGE = 10


def require_isodec_rules() -> None:
    if ENABLE_ISODEC_RULES and IsoDecConfig is None:
        raise ImportError(
            "ENABLE_ISODEC_RULES=True but IsoDec modules could not be imported. "
            "Install UniDec IsoDec deps or set ENABLE_ISODEC_RULES=False."
        )


def build_isodec_config() -> Optional[IsoDecConfig]:
    if IsoDecConfig is None:
        return None
    config = IsoDecConfig()
    config.verbose = 1 if ISODEC_VERBOSE else 0
    config.matchtol = float(MATCH_TOL_PPM)
    config.minpeaks = int(ISODEC_MINPEAKS)
    config.css_thresh = float(ISODEC_CSS_THRESH)
    config.minareacovered = float(ISODEC_MIN_AREA_COVERED) if ISODEC_USE_AREA_COVERED else 0.0
    config.mzwindowlb = float(ISODEC_MZ_WINDOW_LB)
    config.mzwindowub = float(ISODEC_MZ_WINDOW_UB)
    if hasattr(config, "plusoneintwindowlb"):
        config.plusoneintwindowlb = float(ISODEC_PLUSONE_INT_WINDOW_LB)
    if hasattr(config, "plusoneintwindowub"):
        config.plusoneintwindowub = float(ISODEC_PLUSONE_INT_WINDOW_UB)
    config.minusoneaszero = 1 if ISODEC_MINUSONE_AS_ZERO else 0
    config.isotopethreshold = float(REL_INTENSITY_CUTOFF)
    return config
