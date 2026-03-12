const sessionTime = document.getElementById('sessionTime');
const elapsedTime = document.getElementById('elapsedTime');
const runButton = document.getElementById('runButton');
const stopButton = document.getElementById('stopButton');
const statusPill = document.getElementById('statusPill');
const progressBar = document.getElementById('progressBar');
const warningList = document.getElementById('warningList');
const warningCount = document.getElementById('warningCount');
const fileInput = document.getElementById('fileInput');
const fileBrowse = document.getElementById('fileBrowse');
const filePath = document.getElementById('filePath');
const fileStatus = document.getElementById('fileStatus');
const peptideInput = document.getElementById('peptide');
const scanSelect = document.getElementById('scanSelect');
const modeSelect = document.getElementById('modeSelect');
const mzMinInput = document.getElementById('mzMin');
const mzMaxInput = document.getElementById('mzMax');
const copiesInput = document.getElementById('copies');
const amidatedInput = document.getElementById('amidated');
const disulfideBondsInput = document.getElementById('disulfideBonds');
const disulfideMapInput = document.getElementById('disulfideMap');
const fragMinChargeInput = document.getElementById('fragMinCharge');
const fragMaxChargeInput = document.getElementById('fragMaxCharge');
const matchTolInput = document.getElementById('matchTol');
const precursorMatchTolInput = document.getElementById('precursorMatchTol');
const isoDecInput = document.getElementById('isoDec');
const hTransferInput = document.getElementById('hTransfer');
const neutralLossInput = document.getElementById('neutralLoss');
const precursorCalibrationInput = document.getElementById('precursorCalibration');
const enableCentroidInput = document.getElementById('enableCentroid');
const anchorModeSelect = document.getElementById('anchorMode');
const isodecMinPeaksInput = document.getElementById('isodecMinPeaks');
const isodecMinAreaCoveredInput = document.getElementById('isodecMinAreaCovered');
const isodecMzWindowLbInput = document.getElementById('isodecMzWindowLb');
const isodecMzWindowUbInput = document.getElementById('isodecMzWindowUb');
const isodecPlusoneIntWindowLbInput = document.getElementById('isodecPlusoneIntWindowLb');
const isodecPlusoneIntWindowUbInput = document.getElementById('isodecPlusoneIntWindowUb');
const isodecMinusoneAsZeroInput = document.getElementById('isodecMinusoneAsZero');
const coverageLoadButton = document.getElementById('coverageLoadButton');
const coverageFile = document.getElementById('coverageFile');
const coverageStatus = document.getElementById('coverageStatus');
const coveragePopover = document.getElementById('coveragePopover');
const spectrumResetButton = document.getElementById('spectrumResetButton');
const spectrumZoomButton = document.getElementById('spectrumZoomButton');
const cosineSlider = document.getElementById('minCosine');
const cosineValue = document.getElementById('cosineValue');
const diagnoseSpecInput = document.getElementById('diagnoseSpec');
const diagnoseHTransferSelect = document.getElementById('diagnoseHTransfer');
const resultsFilter = document.getElementById('resultsFilter');
const resultsSort = document.getElementById('resultsSort');
const resultsView = document.getElementById('resultsView');
const theoreticalMode = document.getElementById('theoreticalMode');
const theoreticalIonFilter = document.getElementById('theoreticalIonFilter');
const theoreticalChargeFilter = document.getElementById('theoreticalChargeFilter');
const resultsTable = document.getElementById('resultsTable');
const ionTypeChips = Array.from(document.querySelectorAll('.chip-group .chip'));
const visualsStack = document.getElementById('visualsStack');
const swapCoverageResultsButtons = Array.from(document.querySelectorAll('[data-swap-coverage-results]'));

const sessionStart = Date.now();
let runStart = null;
let runTimer = null;
let progress = 0;
let isRunning = false;
let coverageRows = null;
let theoreticalRows = [];
let theoreticalDisplayRows = [];
let activeTableView = 'results';
let tableRows = Array.from(resultsTable.querySelectorAll('tbody tr'));
let manualFilePathOverride = false;
let coverageGroupMap = new Map();
let coveragePopoverHideTimer = null;
let coveragePopoverActiveKey = null;
let activeResultRow = null;
let lastPrecursorWindow = null;
let lastSpectrumState = null;
let lastSpectrumOptions = {};
let swapHighlightTimer = null;

const VISUAL_PANEL_ORDER_KEY = 'ecd.visual-panel-order.v1';

const refreshTableRows = () => {
  tableRows = Array.from(resultsTable.querySelectorAll('tbody tr'));
};

const API_BASE = 'http://127.0.0.1:8001';

const formatClock = (ms) => {
  const totalSeconds = Math.floor(ms / 1000);
  const hours = String(Math.floor(totalSeconds / 3600)).padStart(2, '0');
  const minutes = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, '0');
  const seconds = String(totalSeconds % 60).padStart(2, '0');
  return `${hours}:${minutes}:${seconds}`;
};

const formatElapsed = (ms) => {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
  const seconds = String(totalSeconds % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
};

const RESULTS_TABLE_CONFIG = {
  columns: [
    { key: 'ion', label: 'Ion' },
    { key: 'formula', label: 'Formula' },
    { key: 'obsMz', label: 'Obs m/z' },
    { key: 'theoryMz', label: 'Theory m/z' },
    { key: 'ppm', label: 'ppm' },
    { key: 'charge', label: 'Charge' },
    { key: 'intensity', label: 'Intensity' },
    { key: 'score', label: 'Score' },
  ],
  sorts: {
    score: { label: 'Sort by score', direction: 'desc' },
    mz: { label: 'Sort by m/z', direction: 'desc' },
    charge: { label: 'Sort by charge', direction: 'desc' },
  },
};

const THEORETICAL_TABLE_CONFIG = {
  columns: [
    { key: 'ion', label: 'Ion' },
    { key: 'cleavage', label: 'Cleavage' },
    { key: 'chargeSummary', label: 'Charge States' },
    { key: 'anchorRange', label: 'Anchor m/z Range' },
    { key: 'variants', label: 'Variants' },
    { key: 'theoryPeak', label: 'Max Theory Peak' },
  ],
  sorts: {
    score: { label: 'Sort by variants', direction: 'desc' },
    mz: { label: 'Sort by anchor m/z', direction: 'asc' },
    charge: { label: 'Sort by cleavage', direction: 'asc' },
  },
};

const THEORETICAL_ALL_TABLE_CONFIG = {
  columns: [
    { key: 'ion', label: 'Ion' },
    { key: 'cleavage', label: 'Cleavage' },
    { key: 'charge', label: 'Charge' },
    { key: 'anchorMz', label: 'Anchor m/z' },
    { key: 'variant', label: 'Variant' },
    { key: 'theoryPeak', label: 'Max Theory Peak' },
  ],
  sorts: {
    score: { label: 'Sort by ion', direction: 'asc' },
    mz: { label: 'Sort by m/z', direction: 'asc' },
    charge: { label: 'Sort by charge', direction: 'asc' },
  },
};

const ION_RANK = { b: 0, c: 1, y: 2, z: 3 };

const normalizeIonSeries = (value) => {
  const ion = (value || '').toLowerCase();
  if (ion.startsWith('b')) return 'b';
  if (ion.startsWith('c')) return 'c';
  if (ion.startsWith('y')) return 'y';
  if (ion.startsWith('z')) return 'z';
  return null;
};

const normalizeIonType = (value) => {
  const ion = (value || '').toLowerCase();
  if (ion === 'c-dot') return 'c-dot';
  if (ion === 'z-dot') return 'z-dot';
  return normalizeIonSeries(ion);
};

const getIonRank = (value) => ION_RANK[normalizeIonSeries(value)] ?? 9;

const formatTheoreticalIonFilterLabel = (ionType) => `${ionType} ions`;

const getTheoreticalModeValue = () => (
  theoreticalMode && theoreticalMode.value === 'all' ? 'all' : 'grouped'
);

const getResultsTableConfig = () => {
  if (activeTableView !== 'theoretical') {
    return RESULTS_TABLE_CONFIG;
  }
  return getTheoreticalModeValue() === 'all' ? THEORETICAL_ALL_TABLE_CONFIG : THEORETICAL_TABLE_CONFIG;
};

const formatTableMz = (value) => (Number.isFinite(value) ? Number(value).toFixed(4) : '');
const formatTablePpm = (value, fallback = '') => (Number.isFinite(value) ? Number(value).toFixed(4) : fallback);

const formatChargeSummary = (charges) => {
  const values = Array.isArray(charges)
    ? charges.filter((value) => Number.isFinite(value)).sort((a, b) => a - b)
    : [];
  return values.map((value) => `${value}+`).join(', ');
};

const formatAnchorRange = (minValue, maxValue) => {
  const min = Number.isFinite(minValue) ? Number(minValue) : null;
  const max = Number.isFinite(maxValue) ? Number(maxValue) : null;
  if (min == null && max == null) {
    return '';
  }
  if (min != null && max != null && Math.abs(min - max) > 1e-6) {
    return `${min.toFixed(4)} - ${max.toFixed(4)}`;
  }
  return formatTableMz(min ?? max);
};

const getSequenceText = () => String(peptideInput?.value || '').trim();

const getCleavageIndex = (row) => {
  const sequenceLength = getSequenceText().length;
  if (sequenceLength >= 2) {
    return computeFragmentIndex(row.ionType, row.fragLen, sequenceLength, row.fragmentIndex);
  }
  return Number.isFinite(row.fragmentIndex) ? Math.floor(row.fragmentIndex) : null;
};

const formatCleavageLabel = (row) => {
  const index = getCleavageIndex(row);
  if (!Number.isFinite(index)) {
    return '';
  }
  const leftPos = index + 1;
  const rightPos = index + 2;
  const sequence = getSequenceText();
  if (sequence.length > rightPos - 1) {
    return `${sequence[index]}${leftPos}|${sequence[index + 1]}${rightPos}`;
  }
  return `${leftPos}|${rightPos}`;
};

const syncResultsTableControls = () => {
  if (!resultsTable) {
    return;
  }
  const config = getResultsTableConfig();
  const headerRow = resultsTable.querySelector('thead tr');
  if (headerRow) {
    headerRow.innerHTML = config.columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join('');
  }
  if (!resultsSort) {
    return;
  }
  Object.entries(config.sorts).forEach(([key, meta]) => {
    const option = resultsSort.querySelector(`option[value="${key}"]`);
    if (option) {
      option.textContent = meta.label;
    }
  });
  const showTheoreticalFilters = activeTableView === 'theoretical';
  if (theoreticalMode) {
    theoreticalMode.hidden = !showTheoreticalFilters;
  }
  if (theoreticalIonFilter) {
    theoreticalIonFilter.hidden = !showTheoreticalFilters;
  }
  if (theoreticalChargeFilter) {
    theoreticalChargeFilter.hidden = !showTheoreticalFilters;
  }
};

const getResultsRowViewModel = (frag) => ({
  cells: {
    ion: frag.displayLabel || (frag.ionType ? `${frag.ionType}${frag.fragLen ?? ''}` : ''),
    formula: frag.formula || '',
    obsMz: formatTableMz(frag.obsMz),
    theoryMz: formatTableMz(frag.anchorMz),
    ppm: formatTablePpm(frag.anchorPpm),
    charge: frag.charge ? `${frag.charge}+` : '',
    intensity: frag.obsInt != null ? Math.round(frag.obsInt).toLocaleString() : '',
    score: Number.isFinite(frag.score) ? frag.score.toFixed(3) : (frag.css != null ? frag.css.toFixed(3) : ''),
  },
  sortValues: {
    score: Number.isFinite(frag.score) ? frag.score : (Number.isFinite(frag.css) ? frag.css : null),
    mz: Number.isFinite(frag.obsMz) ? frag.obsMz : frag.anchorMz,
    charge: Number.isFinite(frag.charge) ? frag.charge : null,
  },
  centerMz: Number.isFinite(frag.obsMz) ? frag.obsMz : frag.anchorMz,
});

const getTheoreticalRowViewModel = (frag) => ({
  cells: {
    ion: frag.displayLabel || (frag.ionType ? `${frag.ionType}${frag.fragLen ?? ''}` : ''),
    cleavage: formatCleavageLabel(frag),
    chargeSummary: frag.chargeSummary || formatChargeSummary(frag.chargeStates),
    anchorRange: formatAnchorRange(frag.anchorMzMin, frag.anchorMzMax),
    variants: Number.isFinite(frag.variantCount) ? String(frag.variantCount) : '',
    theoryPeak: Number.isFinite(frag.theoryPeakMax) ? Math.round(frag.theoryPeakMax).toLocaleString() : '',
  },
  sortValues: {
    score: Number.isFinite(frag.variantCount) ? frag.variantCount : null,
    mz: Number.isFinite(frag.anchorMzMin) ? frag.anchorMzMin : frag.anchorMz,
    charge: Number.isFinite(getCleavageIndex(frag)) ? getCleavageIndex(frag) + 1 : null,
  },
  centerMz: Number.isFinite(frag.anchorMz) ? frag.anchorMz : frag.anchorMzMin,
});

const getTheoreticalAllRowViewModel = (frag) => {
  const ionRank = getIonRank(frag.ionType);
  const variantParts = [frag.variantSuffix, frag.lossSuffix].filter(Boolean);
  return {
    cells: {
      ion: frag.displayLabel || frag.label || (frag.ionType ? `${frag.ionType}${frag.fragLen ?? ''}` : ''),
      cleavage: formatCleavageLabel(frag),
      charge: frag.charge ? `${frag.charge}+` : '',
      anchorMz: formatTableMz(frag.anchorMz),
      variant: variantParts.length ? variantParts.join(' ') : 'neutral',
      theoryPeak: Number.isFinite(frag.theoryPeakMax) ? Math.round(frag.theoryPeakMax).toLocaleString() : '',
    },
    sortValues: {
      score: ionRank * 1000 + (Number.isFinite(frag.fragLen) ? frag.fragLen : 0),
      mz: Number.isFinite(frag.anchorMz) ? frag.anchorMz : null,
      charge: Number.isFinite(frag.charge) ? frag.charge : null,
    },
    centerMz: Number.isFinite(frag.anchorMz) ? frag.anchorMz : null,
  };
};

const getRowSortDataset = (sortValues) => ({
  score: Number.isFinite(sortValues.score) ? String(sortValues.score) : '',
  mz: Number.isFinite(sortValues.mz) ? String(sortValues.mz) : '',
  charge: Number.isFinite(sortValues.charge) ? String(sortValues.charge) : '',
});

const getSelectedTheoreticalCharge = () => {
  if (!theoreticalChargeFilter || theoreticalChargeFilter.value === 'all') {
    return null;
  }
  const value = Number(theoreticalChargeFilter.value);
  return Number.isFinite(value) ? value : null;
};

const rowMatchesTheoreticalFilters = (row) => {
  const ionValue = theoreticalIonFilter ? theoreticalIonFilter.value : 'all';
  if (ionValue !== 'all' && row.ionType !== ionValue) {
    return false;
  }
  const chargeValue = getSelectedTheoreticalCharge();
  if (!Number.isFinite(chargeValue)) {
    return true;
  }
  if (getTheoreticalModeValue() === 'all') {
    return Number.isFinite(row.charge) && row.charge === chargeValue;
  }
  if (Array.isArray(row.chargeStates) && row.chargeStates.length) {
    return row.chargeStates.includes(chargeValue);
  }
  return Number.isFinite(row.charge) && row.charge === chargeValue;
};

const getFilteredRawTheoreticalRows = () => (
  (Array.isArray(theoreticalRows) ? theoreticalRows : []).filter(rowMatchesTheoreticalFilters)
);

const getFilteredTheoreticalRows = () => {
  const baseRows = getTheoreticalModeValue() === 'all'
    ? theoreticalRows
    : (Array.isArray(theoreticalDisplayRows) && theoreticalDisplayRows.length ? theoreticalDisplayRows : theoreticalRows);
  return (Array.isArray(baseRows) ? baseRows : []).filter(rowMatchesTheoreticalFilters);
};

const syncTheoreticalChargeFilterOptions = () => {
  if (!theoreticalChargeFilter) {
    return;
  }
  const currentValue = theoreticalChargeFilter.value || 'all';
  const charges = Array.from(
    new Set(
      (Array.isArray(theoreticalRows) ? theoreticalRows : [])
        .map((row) => row.charge)
        .filter((value) => Number.isFinite(value))
    )
  ).sort((a, b) => a - b);
  theoreticalChargeFilter.innerHTML = [
    '<option value="all">All charges</option>',
    ...charges.map((charge) => `<option value="${charge}">${charge}+</option>`),
  ].join('');
  theoreticalChargeFilter.value = charges.some((charge) => String(charge) === currentValue) ? currentValue : 'all';
};

const syncTheoreticalIonFilterOptions = () => {
  if (!theoreticalIonFilter) {
    return;
  }
  const currentValue = theoreticalIonFilter.value || 'all';
  const ionTypes = Array.from(
    new Set(
      (Array.isArray(theoreticalRows) ? theoreticalRows : [])
        .map((row) => normalizeIonType(row.ionType))
        .filter(Boolean)
    )
  ).sort((left, right) => (
    getIonRank(left) - getIonRank(right) || String(left).localeCompare(String(right))
  ));
  theoreticalIonFilter.innerHTML = [
    '<option value="all">All ion types</option>',
    ...ionTypes.map((ionType) => `<option value="${ionType}">${formatTheoreticalIonFilterLabel(ionType)}</option>`),
  ].join('');
  theoreticalIonFilter.value = ionTypes.includes(currentValue) ? currentValue : 'all';
};

const getVisualPanels = () => (
  visualsStack ? Array.from(visualsStack.querySelectorAll('[data-reorderable-panel]')) : []
);

const resizeVisualPlots = () => {
  if (typeof Plotly === 'undefined' || !Plotly?.Plots || typeof Plotly.Plots.resize !== 'function') {
    return;
  }
  ['spectrumPlot', 'coveragePlot'].forEach((id) => {
    const plot = document.getElementById(id);
    if (!plot) {
      return;
    }
    try {
      Plotly.Plots.resize(plot);
    } catch (error) {
      // Ignore transient resize failures while panels are moving.
    }
  });
};

const persistVisualPanelOrder = () => {
  if (!visualsStack) {
    return;
  }
  try {
    const order = getVisualPanels()
      .map((panel) => panel.dataset.panelId)
      .filter(Boolean);
    localStorage.setItem(VISUAL_PANEL_ORDER_KEY, JSON.stringify(order));
  } catch (error) {
    // Ignore localStorage failures.
  }
};

const applyVisualPanelOrder = (order) => {
  if (!visualsStack || !Array.isArray(order) || !order.length) {
    return;
  }
  const panels = new Map(getVisualPanels().map((panel) => [panel.dataset.panelId, panel]));
  order.forEach((id) => {
    const panel = panels.get(id);
    if (panel) {
      visualsStack.appendChild(panel);
    }
  });
  requestAnimationFrame(() => resizeVisualPlots());
};

const swapVisualPanels = (firstId, secondId) => {
  if (!visualsStack || !firstId || !secondId || firstId === secondId) {
    return null;
  }
  const order = getVisualPanels()
    .map((panel) => panel.dataset.panelId)
    .filter(Boolean);
  const firstIndex = order.indexOf(firstId);
  const secondIndex = order.indexOf(secondId);
  if (firstIndex === -1 || secondIndex === -1) {
    return null;
  }
  [order[firstIndex], order[secondIndex]] = [order[secondIndex], order[firstIndex]];
  applyVisualPanelOrder(order);
  persistVisualPanelOrder();
  highlightSwappedPanels(firstId, secondId);
  return new Map(getVisualPanels().map((panel) => [panel.dataset.panelId, panel]));
};

const loadVisualPanelOrder = () => {
  try {
    const raw = localStorage.getItem(VISUAL_PANEL_ORDER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
};

const highlightSwappedPanels = (...panelIds) => {
  const ids = panelIds.filter(Boolean);
  if (!ids.length) {
    return;
  }
  if (swapHighlightTimer) {
    clearTimeout(swapHighlightTimer);
    swapHighlightTimer = null;
  }
  getVisualPanels().forEach((panel) => {
    panel.classList.toggle('is-swapped', ids.includes(panel.dataset.panelId));
  });
  swapHighlightTimer = setTimeout(() => {
    getVisualPanels().forEach((panel) => panel.classList.remove('is-swapped'));
    swapHighlightTimer = null;
  }, 1200);
};

const setWarnings = (items) => {
  warningList.innerHTML = '';
  if (!items.length) {
    warningCount.textContent = '0';
    const item = document.createElement('div');
    item.className = 'warning-item';
    item.textContent = 'No warnings.';
    warningList.appendChild(item);
    return;
  }
  warningCount.textContent = String(items.length);
  items.forEach((text) => {
    const item = document.createElement('div');
    item.className = 'warning-item';
    item.textContent = text;
    warningList.appendChild(item);
  });
};

const updateSession = () => {
  sessionTime.textContent = formatClock(Date.now() - sessionStart);
  if (isRunning && runStart) {
    elapsedTime.textContent = formatElapsed(Date.now() - runStart);
  }
};

const stopRun = () => {
  isRunning = false;
  if (runTimer) {
    clearInterval(runTimer);
    runTimer = null;
  }
  statusPill.textContent = 'Idle';
  statusPill.classList.remove('status-running');
  statusPill.classList.add('status-idle');
  runButton.disabled = false;
  stopButton.disabled = true;
};

const toNumberOrNull = (value) => {
  if (value == null) {
    return null;
  }
  const text = String(value).trim();
  if (!text) {
    return null;
  }
  const n = Number(text);
  return Number.isFinite(n) ? n : null;
};

const getIonTypesFromChips = () => {
  const alias = { 'z*': 'z-dot', 'c*': 'c-dot' };
  const selected = ionTypeChips
    .map((chip) => {
      const input = chip.querySelector('input');
      const label = chip.querySelector('span');
      if (!input || !label || !input.checked) {
        return null;
      }
      const raw = label.textContent.trim().toLowerCase();
      return alias[raw] || raw;
    })
    .filter(Boolean);
  return selected.length ? selected : ['b', 'y', 'c', 'z-dot'];
};

const ionTypeToChipLabel = (ionType) => {
  const ion = String(ionType || '').toLowerCase();
  if (ion === 'z-dot') return 'z*';
  if (ion === 'c-dot') return 'c*';
  return ion;
};

const setChipSelection = (ionTypes) => {
  if (!ionTypes || !ionTypes.length) {
    return;
  }
  const selected = new Set(ionTypes.map(ionTypeToChipLabel));
  ionTypeChips.forEach((chip) => {
    const input = chip.querySelector('input');
    const label = chip.querySelector('span');
    if (!input || !label) {
      return;
    }
    const key = label.textContent.trim().toLowerCase();
    input.checked = selected.has(key);
  });
};

const loadConfig = async () => {
  try {
    const response = await fetch(`${API_BASE}/api/config`);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    const hasSelectedFile = Boolean(fileInput.files && fileInput.files.length);
    if (filePath && data.filepath && !manualFilePathOverride && !hasSelectedFile) {
      filePath.value = data.filepath;
      fileStatus.textContent = 'Using config defaults';
      manualFilePathOverride = false;
    }
    if (scanSelect && data.scan) scanSelect.value = data.scan;
    if (peptideInput && data.peptide) peptideInput.value = data.peptide;
    if (mzMinInput) mzMinInput.value = data.mz_min ?? '';
    if (mzMaxInput) mzMaxInput.value = data.mz_max ?? '';
    if (copiesInput && data.copies) copiesInput.value = data.copies;
    if (amidatedInput) amidatedInput.checked = Boolean(data.amidated);
    if (disulfideBondsInput && data.disulfide_bonds !== undefined) disulfideBondsInput.value = data.disulfide_bonds;
    if (disulfideMapInput) disulfideMapInput.value = data.disulfide_map || '';
    if (fragMinChargeInput && data.frag_min_charge) fragMinChargeInput.value = data.frag_min_charge;
    if (fragMaxChargeInput && data.frag_max_charge) fragMaxChargeInput.value = data.frag_max_charge;
    if (matchTolInput && data.match_tol_ppm) matchTolInput.value = data.match_tol_ppm;
    if (precursorMatchTolInput && data.precursor_match_tol_ppm) precursorMatchTolInput.value = data.precursor_match_tol_ppm;
    const cosineDefault =
      data.isodec_css_thresh !== undefined ? data.isodec_css_thresh : data.min_cosine;
    if (cosineSlider && cosineDefault !== undefined) {
      cosineSlider.value = cosineDefault;
      cosineValue.textContent = Number(cosineDefault).toFixed(2);
    }
    if (isoDecInput && data.enable_isodec_rules !== undefined) isoDecInput.checked = Boolean(data.enable_isodec_rules);
    if (hTransferInput && data.enable_h_transfer !== undefined) hTransferInput.checked = Boolean(data.enable_h_transfer);
    if (neutralLossInput && data.enable_neutral_losses !== undefined) neutralLossInput.checked = Boolean(data.enable_neutral_losses);
    if (precursorCalibrationInput && data.precursor_calibration !== undefined) precursorCalibrationInput.checked = Boolean(data.precursor_calibration);
    if (enableCentroidInput && data.enable_centroid !== undefined) enableCentroidInput.checked = Boolean(data.enable_centroid);
    if (isodecMinPeaksInput && data.isodec_min_peaks !== undefined) isodecMinPeaksInput.value = data.isodec_min_peaks;
    if (isodecMinAreaCoveredInput && data.isodec_min_area_covered !== undefined) isodecMinAreaCoveredInput.value = data.isodec_min_area_covered;
    if (isodecMzWindowLbInput && data.isodec_mz_window_lb !== undefined) isodecMzWindowLbInput.value = data.isodec_mz_window_lb;
    if (isodecMzWindowUbInput && data.isodec_mz_window_ub !== undefined) isodecMzWindowUbInput.value = data.isodec_mz_window_ub;
    if (isodecPlusoneIntWindowLbInput && data.isodec_plusone_int_window_lb !== undefined) isodecPlusoneIntWindowLbInput.value = data.isodec_plusone_int_window_lb;
    if (isodecPlusoneIntWindowUbInput && data.isodec_plusone_int_window_ub !== undefined) isodecPlusoneIntWindowUbInput.value = data.isodec_plusone_int_window_ub;
    if (isodecMinusoneAsZeroInput && data.isodec_minusone_as_zero !== undefined) isodecMinusoneAsZeroInput.checked = Boolean(data.isodec_minusone_as_zero);
    setChipSelection(data.ion_types || []);
    setCoverageStatus('Config loaded');
  } catch (error) {
    setCoverageStatus('Config unavailable', true);
  }
};

const updateResultsTable = (fragments) => {
  syncResultsTableControls();
  const tbody = resultsTable.querySelector('tbody');
  const withIndex = fragments.map((frag, idx) => ({ ...frag, _idx: idx }));
  const ranked = withIndex.sort((a, b) => {
    const aScore = Number.isFinite(a.score) ? a.score : (Number.isFinite(a.css) ? a.css : -Infinity);
    const bScore = Number.isFinite(b.score) ? b.score : (Number.isFinite(b.css) ? b.css : -Infinity);
    return bScore - aScore;
  });
  tbody.innerHTML = ranked
    .map((frag) => {
      const isTheoreticalRow = activeTableView === 'theoretical';
      const config = getResultsTableConfig();
      const viewModel = isTheoreticalRow
        ? (getTheoreticalModeValue() === 'all' ? getTheoreticalAllRowViewModel(frag) : getTheoreticalRowViewModel(frag))
        : getResultsRowViewModel(frag);
      const sortDataset = getRowSortDataset(viewModel.sortValues);
      const centerMz = viewModel.centerMz;
      const centerAttr = Number.isFinite(centerMz) ? ` data-center-mz="${centerMz}"` : '';
      const theoryRowIndex = Number.isFinite(frag.theoryRowIndex) ? frag.theoryRowIndex : frag._idx;
      const matchAttr = Number.isFinite(frag._idx)
        ? (isTheoreticalRow ? ` data-theory-idx="${theoryRowIndex}"` : ` data-match-idx="${frag._idx}"`)
        : '';
      const hasTheory = Array.isArray(frag.theoryMz) && frag.theoryMz.length;
      const theoryAttr = hasTheory ? ' data-has-theory="1"' : '';
      const sortAttrs =
        ` data-sort-score="${escapeHtml(sortDataset.score)}"` +
        ` data-sort-mz="${escapeHtml(sortDataset.mz)}"` +
        ` data-sort-charge="${escapeHtml(sortDataset.charge)}"`;
      const cells = config.columns
        .map((column) => `<td>${escapeHtml(viewModel.cells[column.key] ?? '')}</td>`)
        .join('');
      return `<tr${centerAttr}${matchAttr}${theoryAttr}${sortAttrs}>${cells}</tr>`;
    })
    .join('');
  activeResultRow = null;
  refreshTableRows();
  applyTableFilter();
  resultsSort.dispatchEvent(new Event('change'));
};

const buildTheoreticalRows = (rows) => {
  const source = Array.isArray(rows) ? rows : [];
  return source
    .filter((row) => Array.isArray(row.theoryMz) && row.theoryMz.length && Array.isArray(row.theoryIntensity) && row.theoryIntensity.length)
    .map((row, idx) => ({
      displayLabel: row.displayLabel || row.label || (row.ionType ? `${row.ionType}${row.fragLen ?? ''}` : 'Theory'),
      ionType: row.ionType,
      ionSeries: row.ionSeries || normalizeIonSeries(row.ionType),
      fragLen: row.fragLen,
      fragmentIndex: row.fragmentIndex,
      charge: row.charge,
      obsMz: null,
      obsInt: null,
      css: row.css,
      score: row.score,
      anchorMz: Number.isFinite(row.anchorMz) ? row.anchorMz : row.theoryMz[0],
      label: row.label,
      formula: row.formula,
      theoryMz: row.theoryMz,
      theoryIntensity: row.theoryIntensity,
      theoryPeakMax: Math.max(...row.theoryIntensity.filter((v) => Number.isFinite(v)), 0),
      overlayColor: row.overlayColor || '#8b5cf6',
      target: row.target,
      variantSuffix: row.variantSuffix || '',
      lossSuffix: row.lossSuffix || '',
      theoryRowIndex: idx,
    }));
};

const normalizeTheoreticalRows = (rows) => {
  const source = Array.isArray(rows) ? rows : [];
  return source
    .map((row, idx) => {
      const theoryMz = Array.isArray(row.theory_mz)
        ? row.theory_mz
        : (Array.isArray(row.theoryMz) ? row.theoryMz : []);
      const theoryIntensity = Array.isArray(row.theory_intensity)
        ? row.theory_intensity
        : (Array.isArray(row.theoryIntensity) ? row.theoryIntensity : []);
      if (!theoryMz.length || !theoryIntensity.length) {
        return null;
      }
      const ionTypeRaw = row.ion_type || row.ionType || '';
      const ionType = normalizeIonType(ionTypeRaw);
      const ionSeries = normalizeIonSeries(row.series || row.ionSeries || ionType);
      const fragLenRaw = row.frag_len ?? row.fragLen;
      const chargeRaw = row.charge;
      const anchorRaw = row.anchor_theory_mz ?? row.anchorMz;
      const anchorMz = Number.isFinite(Number(anchorRaw)) ? Number(anchorRaw) : theoryMz[0];
      return {
        displayLabel: row.label || (ionType && Number.isFinite(Number(fragLenRaw)) ? `${ionType}${Number(fragLenRaw)}` : 'Theory'),
        ionType,
        ionSeries,
        fragLen: Number.isFinite(Number(fragLenRaw)) ? Number(fragLenRaw) : null,
        fragmentIndex: row.fragment_index ?? row.fragmentIndex ?? null,
        charge: Number.isFinite(Number(chargeRaw)) ? Number(chargeRaw) : null,
        obsMz: null,
        obsInt: null,
        css: Number.isFinite(Number(row.css)) ? Number(row.css) : 0,
        score: Number.isFinite(Number(row.score)) ? Number(row.score) : 0,
        anchorMz,
        label: row.label || '',
        formula: row.formula || '',
        theoryMz,
        theoryIntensity,
        theoryPeakMax: Math.max(...theoryIntensity.filter((v) => Number.isFinite(v)), 0),
        overlayColor: row.overlayColor || '#8b5cf6',
        target: row.target,
        variantSuffix: row.variant_suffix || row.variantSuffix || '',
        lossSuffix: row.loss_suffix || row.lossSuffix || '',
        theoryRowIndex: Number.isFinite(Number(row.theoryRowIndex)) ? Number(row.theoryRowIndex) : idx,
      };
    })
    .filter(Boolean);
};

const compareTheoreticalRepresentative = (left, right) => {
  const leftCharge = Number.isFinite(left?.charge) ? left.charge : Infinity;
  const rightCharge = Number.isFinite(right?.charge) ? right.charge : Infinity;
  if (leftCharge !== rightCharge) {
    return leftCharge - rightCharge;
  }
  const leftMz = Number.isFinite(left?.anchorMz) ? left.anchorMz : Infinity;
  const rightMz = Number.isFinite(right?.anchorMz) ? right.anchorMz : Infinity;
  if (leftMz !== rightMz) {
    return leftMz - rightMz;
  }
  return String(left?.label || '').localeCompare(String(right?.label || ''));
};

const buildTheoreticalDisplayRows = (rows) => {
  const source = Array.isArray(rows) ? rows : [];
  const groups = new Map();
  source.forEach((row, idx) => {
    if (!row || !row.ionType || !Number.isFinite(row.fragLen)) {
      return;
    }
    const fragmentIndex = Number.isFinite(row.fragmentIndex) ? row.fragmentIndex : null;
    const key = `${row.ionType}:${row.fragLen}:${fragmentIndex ?? 'na'}`;
    const current = groups.get(key);
    if (!current) {
      groups.set(key, {
        key,
        ionType: row.ionType,
        ionSeries: row.ionSeries || normalizeIonSeries(row.ionType),
        fragLen: row.fragLen,
        fragmentIndex,
        representative: row,
        representativeIndex: idx,
        variantCount: 1,
        chargeStates: new Set(Number.isFinite(row.charge) ? [row.charge] : []),
        anchorMzMin: Number.isFinite(row.anchorMz) ? row.anchorMz : Infinity,
        anchorMzMax: Number.isFinite(row.anchorMz) ? row.anchorMz : -Infinity,
        theoryPeakMax: Number.isFinite(row.theoryPeakMax) ? row.theoryPeakMax : -Infinity,
      });
      return;
    }
    current.variantCount += 1;
    if (Number.isFinite(row.charge)) {
      current.chargeStates.add(row.charge);
    }
    if (Number.isFinite(row.anchorMz)) {
      current.anchorMzMin = Math.min(current.anchorMzMin, row.anchorMz);
      current.anchorMzMax = Math.max(current.anchorMzMax, row.anchorMz);
    }
    if (Number.isFinite(row.theoryPeakMax)) {
      current.theoryPeakMax = Math.max(current.theoryPeakMax, row.theoryPeakMax);
    }
    if (compareTheoreticalRepresentative(row, current.representative) < 0) {
      current.representative = row;
      current.representativeIndex = idx;
    }
  });

  return Array.from(groups.values())
    .map((group) => ({
      displayLabel: `${group.ionType}${group.fragLen}`,
      ionType: group.ionType,
      ionSeries: group.ionSeries,
      fragLen: group.fragLen,
      fragmentIndex: group.fragmentIndex,
      charge: group.representative.charge,
      chargeStates: Array.from(group.chargeStates).sort((a, b) => a - b),
      chargeSummary: formatChargeSummary(Array.from(group.chargeStates)),
      obsMz: null,
      obsInt: null,
      css: 0,
      score: group.variantCount,
      anchorMz: group.representative.anchorMz,
      anchorMzMin: Number.isFinite(group.anchorMzMin) ? group.anchorMzMin : group.representative.anchorMz,
      anchorMzMax: Number.isFinite(group.anchorMzMax) ? group.anchorMzMax : group.representative.anchorMz,
      label: `${group.ionType}${group.fragLen}`,
      formula: '',
      theoryMz: group.representative.theoryMz,
      theoryIntensity: group.representative.theoryIntensity,
      theoryPeakMax: Number.isFinite(group.theoryPeakMax) ? group.theoryPeakMax : group.representative.theoryPeakMax,
      overlayColor: group.representative.overlayColor || '#8b5cf6',
      target: group.representative.target,
      theoryRowIndex: group.representativeIndex,
      variantCount: group.variantCount,
    }))
    .sort((a, b) => {
      return (
        getIonRank(a.ionType) - getIonRank(b.ionType) ||
        (a.fragLen ?? 0) - (b.fragLen ?? 0) ||
        (a.fragmentIndex ?? 0) - (b.fragmentIndex ?? 0)
      );
    });
};

const getTheoreticalTableRows = () => {
  return getFilteredTheoreticalRows();
};

const setCoverageRows = (rows, theoreticalOverride = null) => {
  coverageRows = Array.isArray(rows) ? rows : [];
  if (Array.isArray(theoreticalOverride) && theoreticalOverride.length) {
    theoreticalRows = normalizeTheoreticalRows(theoreticalOverride);
  } else {
    theoreticalRows = buildTheoreticalRows(coverageRows);
  }
  theoreticalDisplayRows = buildTheoreticalDisplayRows(theoreticalRows);
  syncTheoreticalIonFilterOptions();
  syncTheoreticalChargeFilterOptions();
  const tableData = activeTableView === 'theoretical' ? getTheoreticalTableRows() : coverageRows;
  updateResultsTable(tableData);
};

const applyFragments = (fragments, sequence, mode = 'fragments', theoreticalFragments = null) => {
  const normalized = fragments.map((frag) => ({
    ionType: frag.ion_type,
    ionSeries: normalizeIonSeries(frag.series || frag.ion_type),
    fragLen: frag.frag_len,
    fragmentIndex: frag.fragment_index,
    charge: frag.charge,
    obsMz: frag.obs_mz,
    obsInt: frag.obs_int,
    css: frag.css,
    anchorMz: frag.anchor_theory_mz,
    anchorPpm: frag.anchor_ppm,
    label: frag.label,
    theoryMz: Array.isArray(frag.theory_mz) ? frag.theory_mz : [],
    theoryIntensity: Array.isArray(frag.theory_intensity) ? frag.theory_intensity : [],
    overlayColor: '#f59e0b',
  }));
  setCoverageRows(normalized, theoreticalFragments);
  if (sequence && sequence !== peptideInput.value) {
    peptideInput.value = sequence;
  }
  const statusPrefix = mode === 'complex_fragments' ? 'Complex fragments' : 'Fragments';
  setCoverageStatus(`${statusPrefix}: ${normalized.length}`);
  rerenderCoverage();
};

const buildSpectrumOverlaysFromCandidates = (candidates, defaultColor = '#f59e0b') => {
  const source = Array.isArray(candidates) ? candidates : [];
  return source
    .map((cand) => {
      const mz = Array.isArray(cand?.theory_mz) ? cand.theory_mz : [];
      const intensity = Array.isArray(cand?.theory_intensity) ? cand.theory_intensity : [];
      if (!mz.length || !intensity.length) return null;
      return {
        mz,
        intensity,
        color: cand.color || defaultColor,
        label: cand.label || 'Theory',
        target: cand.target,
        status: cand.status,
      };
    })
    .filter(Boolean);
};

const applyPrecursorResults = (data) => {
  const sequence = data.sequence;
  if (sequence && sequence !== peptideInput.value) {
    peptideInput.value = sequence;
  }
  const summary = data.precursor || {};
  const acceptedCandidates = Array.isArray(data.candidates) ? data.candidates : [];
  const ambiguousCandidates = Array.isArray(data.ambiguous_candidates) ? data.ambiguous_candidates : [];
  const visibleCandidates = acceptedCandidates.length ? acceptedCandidates : ambiguousCandidates;
  const rows = visibleCandidates.map((cand) => {
    const isAccepted = cand.accepted !== false && cand.status !== 'ambiguous';
    const prefix = isAccepted ? '' : '? ';
    return {
      displayLabel: `${prefix}Precursor ${cand.charge}+`,
      ionType: 'p',
      fragLen: cand.charge,
      fragmentIndex: null,
      charge: cand.charge,
      obsMz: cand.obs_mz,
      obsInt: null,
      css: cand.css,
      score: cand.composite_score ?? cand.score ?? cand.css,
      anchorMz: cand.anchor_theory_mz,
      anchorPpm: cand.ppm,
      accepted: cand.accepted,
      status: cand.status || (isAccepted ? 'accepted' : 'ambiguous'),
      coverage: cand.coverage,
      ppmRmse: cand.ppm_rmse,
      label: `Precursor ${cand.charge}+`,
      theoryMz: Array.isArray(cand.theory_mz) ? cand.theory_mz : [],
      theoryIntensity: Array.isArray(cand.theory_intensity) ? cand.theory_intensity : [],
      overlayColor: cand.color || (isAccepted ? '#ef4444' : '#f59e0b'),
    };
  });
  setCoverageRows(rows, data.theoretical_fragments);
  hideCoveragePopover();
  rerenderCoverage();

  const bestCss = toNumberOrNull(summary.best_css);
  const bestScore = toNumberOrNull(summary.best_score ?? summary.best_css);
  const shiftPpm = toNumberOrNull(summary.shift_ppm);
  const blockReasons = Array.isArray(summary.calibration_block_reasons)
    ? summary.calibration_block_reasons.filter(Boolean)
    : [];
  if (summary.match_found) {
    const bestCharge = summary.best_charge ? `${summary.best_charge}+` : '?';
    const scoreText = Number.isFinite(bestScore) ? bestScore.toFixed(3) : '--';
    const cssText = Number.isFinite(bestCss) ? bestCss.toFixed(3) : '--';
    const shiftApplied = Number.isFinite(shiftPpm) ? -shiftPpm : null;
    const shiftText = shiftApplied != null ? `, shift ${formatTablePpm(shiftApplied, '--')} ppm` : '';
    let statusText = `Precursor ${bestCharge} score=${scoreText} (css=${cssText})${shiftText}`;
    if (summary.calibration_requested) {
      if (summary.calibration_applied) {
        statusText += ', calibrated';
      } else if (blockReasons.length) {
        statusText += `, calibration blocked: ${blockReasons.join(', ')}`;
      } else {
        statusText += ', calibration skipped';
      }
    }
    setCoverageStatus(statusText, !summary.calibration_applied && blockReasons.length > 0);
  } else if (summary.search_status === 'ambiguous' || ambiguousCandidates.length) {
    setCoverageStatus(`Precursor ambiguous: ${ambiguousCandidates.length} close candidates`, true);
  } else {
    setCoverageStatus('Precursor not found', true);
  }
};

const applyChargeReducedResults = (data) => {
  const sequence = data.sequence;
  if (sequence && sequence !== peptideInput.value) {
    peptideInput.value = sequence;
  }
  const acceptedCandidates = Array.isArray(data.candidates) ? data.candidates : [];
  const ambiguousCandidates = Array.isArray(data.ambiguous_candidates) ? data.ambiguous_candidates : [];
  const shadowedCandidates = Array.isArray(data.shadowed_candidates) ? data.shadowed_candidates : [];
  const visibleCandidates = acceptedCandidates.length ? acceptedCandidates : ambiguousCandidates;
  const rows = visibleCandidates.map((cand) => {
    const target = cand.target ? `${cand.target} ` : '';
    const state = cand.state && cand.state !== '0' ? ` ${cand.state}` : '';
    const isAccepted = (cand.status || 'accepted') === 'accepted';
    const prefix = isAccepted ? '' : '? ';
    return {
      displayLabel: `${prefix}CR ${target}${cand.charge}+${state}`.trim(),
      ionType: 'cr',
      fragLen: null,
      fragmentIndex: null,
      charge: cand.charge,
      obsMz: cand.obs_mz,
      obsInt: cand.obs_int,
      css: cand.css,
      score: cand.score ?? cand.css,
      anchorMz: cand.anchor_theory_mz,
      anchorPpm: cand.ppm,
      label: cand.label,
      status: cand.status || (isAccepted ? 'accepted' : 'ambiguous'),
      coverage: cand.coverage,
      ppmRmse: cand.ppm_rmse,
      matchCount: cand.match_count,
      strategy: cand.strategy,
      theoryMz: Array.isArray(cand.theory_mz) ? cand.theory_mz : [],
      theoryIntensity: Array.isArray(cand.theory_intensity) ? cand.theory_intensity : [],
      overlayColor: cand.color || (isAccepted ? '#0f172a' : '#f59e0b'),
      target: cand.target,
    };
  });
  setCoverageRows(rows, data.theoretical_fragments);
  hideCoveragePopover();
  rerenderCoverage();
  if (acceptedCandidates.length) {
    setCoverageStatus(
      `Charge reduced: ${acceptedCandidates.length} accepted, ${ambiguousCandidates.length} ambiguous, ${shadowedCandidates.length} shadowed`
    );
  } else if (ambiguousCandidates.length) {
    setCoverageStatus(
      `Charge reduced ambiguous: ${ambiguousCandidates.length} candidates (${shadowedCandidates.length} shadowed)`,
      true
    );
  } else {
    setCoverageStatus('Charge reduced: no matches', true);
  }
};

const applyRawResults = (data) => {
  const sequence = data.sequence;
  if (sequence && sequence !== peptideInput.value) {
    peptideInput.value = sequence;
  }
  setCoverageRows([], data.theoretical_fragments);
  hideCoveragePopover();
  rerenderCoverage();
  setCoverageStatus('Raw spectrum');
};

const applyDiagnoseResults = (data) => {
  const sequence = data.sequence;
  if (sequence && sequence !== peptideInput.value) {
    peptideInput.value = sequence;
  }
  const results = Array.isArray(data.results) ? data.results : [];
  const rows = results.map((r) => ({
    displayLabel: r.label || `${r.ion_type}${r.frag_len}^${r.charge}+`,
    ionType: r.ion_type,
    fragLen: r.frag_len,
    fragmentIndex: null,
    charge: r.charge,
    obsMz: r.anchor_obs_mz,
    obsInt: r.obs_int,
    css: r.isodec_css,
    anchorMz: r.anchor_theory_mz,
    anchorPpm: r.anchor_ppm,
    label: r.label,
    ok: r.ok,
    reason: r.reason,
    formula: r.formula,
    monoMass: r.mono_mass,
    rawCosine: r.raw_cosine,
    diagnosticSteps: r.diagnostic_steps || [],
    theoryMz: Array.isArray(r.theory_mz) ? r.theory_mz : [],
    theoryIntensity: Array.isArray(r.theory_int) ? r.theory_int : [],
    overlayColor: r.ok ? '#22c55e' : '#ef4444',
  }));
  setCoverageRows(rows, data.theoretical_fragments);
  hideCoveragePopover();
  rerenderCoverage();

  const best = data.best;
  const ionSpec = data.ion_spec || '';
  if (best && best.ok) {
    const cssText = Number.isFinite(best.isodec_css) ? best.isodec_css.toFixed(3) : '--';
    setCoverageStatus(`Diagnose ${ionSpec}: PASS (css=${cssText})`);
  } else if (best) {
    setCoverageStatus(`Diagnose ${ionSpec}: FAIL - ${best.reason || 'Unknown'}`, true);
  } else {
    setCoverageStatus(`Diagnose ${ionSpec}: No results`, true);
  }
};

const buildDiagnoseOverlays = (results) => {
  const rows = Array.isArray(results) ? results : [];
  const palette = ['#8b5cf6', '#06b6d4', '#f59e0b', '#22c55e', '#ef4444', '#0ea5e9'];
  let colorIndex = 0;
  return rows
    .map((r) => {
      const mz = Array.isArray(r?.theory_mz) ? r.theory_mz : [];
      const intensity = Array.isArray(r?.theory_int) ? r.theory_int : [];
      if (!mz.length || !intensity.length) return null;
      const label = r?.label || 'Theory';
      const color = palette[colorIndex % palette.length];
      colorIndex += 1;
      return { mz, intensity, color, label };
    })
    .filter(Boolean);
};

const applySpectrum = (spectrum, theory, options = {}) => {
  if (!spectrum || !Array.isArray(spectrum.mz) || !spectrum.mz.length) {
    return;
  }
  const theoryMz = theory && Array.isArray(theory.mz) ? theory.mz : [];
  const theoryInt = theory && Array.isArray(theory.intensity) ? theory.intensity : [];
  lastSpectrumState = {
    mz: spectrum.mz,
    intensity: spectrum.intensity,
    theoryMz,
    theoryInt,
  };
  lastSpectrumOptions = { ...options };
  renderSpectrumPlot(spectrum.mz, spectrum.intensity, theoryMz, theoryInt, lastSpectrumOptions);
};

const startRun = async () => {
  isRunning = true;
  runStart = Date.now();
  progress = 0;
  progressBar.style.width = '0%';
  statusPill.textContent = 'Running';
  statusPill.classList.remove('status-idle');
  statusPill.classList.add('status-running');
  runButton.disabled = true;
  stopButton.disabled = false;

  const selectedFile = fileInput.files && fileInput.files.length ? fileInput.files[0] : null;
  const filepathRaw = filePath.value.trim();
  const filepathValue = filepathRaw || (selectedFile ? selectedFile.name : '');
  const hasSelectedFile = Boolean(selectedFile);
  const manualPathActive = manualFilePathOverride && Boolean(filepathRaw);
  if (!filepathValue) {
    setWarnings(['File path is required to run the backend pipeline.']);
    stopRun();
    return;
  }

  runTimer = setInterval(() => {
    progress = Math.min(progress + 3, 92);
    progressBar.style.width = `${progress}%`;
  }, 300);

  try {
    const mode = modeSelect ? modeSelect.value : 'fragments';
    const isPrecursor = mode === 'precursor';
    const isChargeReduced = mode === 'charge_reduced';
    const isComplexFragments = mode === 'complex_fragments';
    const isDiagnose = mode === 'diagnose';
    const isRaw = mode === 'raw';

    // Build payload based on mode
    let payload;
    if (isDiagnose) {
      const ionSpec = diagnoseSpecInput ? diagnoseSpecInput.value.trim() : '';
      if (!ionSpec) {
        setWarnings(['Ion spec is required for diagnose mode (e.g., c7^2+, z12-2H2O^3+)']);
        stopRun();
        return;
      }
      payload = {
        filepath: filepathValue,
        scan: scanSelect ? Number(scanSelect.value) : 1,
        peptide: peptideInput.value,
        ion_spec: ionSpec,
        h_transfer: diagnoseHTransferSelect ? Number(diagnoseHTransferSelect.value) : 0,
        mz_min: mzMinInput ? toNumberOrNull(mzMinInput.value) : null,
        mz_max: mzMaxInput ? toNumberOrNull(mzMaxInput.value) : null,
        frag_min_charge: fragMinChargeInput ? Number(fragMinChargeInput.value) : null,
        frag_max_charge: fragMaxChargeInput ? Number(fragMaxChargeInput.value) : null,
        match_tol_ppm: matchTolInput ? Number(matchTolInput.value) : null,
        min_cosine: Number(cosineSlider.value),
        isodec_min_peaks: isodecMinPeaksInput ? Number(isodecMinPeaksInput.value) : null,
        isodec_min_area_covered: isodecMinAreaCoveredInput ? Number(isodecMinAreaCoveredInput.value) : null,
        isodec_mz_window_lb: isodecMzWindowLbInput ? Number(isodecMzWindowLbInput.value) : null,
        isodec_mz_window_ub: isodecMzWindowUbInput ? Number(isodecMzWindowUbInput.value) : null,
        isodec_plusone_int_window_lb: isodecPlusoneIntWindowLbInput ? Number(isodecPlusoneIntWindowLbInput.value) : null,
        isodec_plusone_int_window_ub: isodecPlusoneIntWindowUbInput ? Number(isodecPlusoneIntWindowUbInput.value) : null,
        isodec_minusone_as_zero: isodecMinusoneAsZeroInput ? isodecMinusoneAsZeroInput.checked : null,
        copies: copiesInput ? Number(copiesInput.value) : null,
        amidated: amidatedInput ? amidatedInput.checked : null,
        disulfide_bonds: disulfideBondsInput ? Number(disulfideBondsInput.value) : null,
        disulfide_map: disulfideMapInput ? disulfideMapInput.value : '',
        enable_isodec_rules: isoDecInput ? isoDecInput.checked : null,
        anchor_mode: anchorModeSelect ? anchorModeSelect.value : null,
      };
    } else {
      payload = {
        filepath: filepathValue,
        scan: scanSelect ? Number(scanSelect.value) : 1,
        mode,
        peptide: peptideInput.value,
        mz_min: mzMinInput ? toNumberOrNull(mzMinInput.value) : null,
        mz_max: mzMaxInput ? toNumberOrNull(mzMaxInput.value) : null,
        ion_types: getIonTypesFromChips(),
        frag_min_charge: fragMinChargeInput ? Number(fragMinChargeInput.value) : null,
        frag_max_charge: fragMaxChargeInput ? Number(fragMaxChargeInput.value) : null,
        match_tol_ppm: matchTolInput ? Number(matchTolInput.value) : null,
        precursor_match_tol_ppm: precursorMatchTolInput ? Number(precursorMatchTolInput.value) : null,
        min_cosine: Number(cosineSlider.value),
        isodec_css_thresh: Number(cosineSlider.value),
        isodec_min_peaks: isodecMinPeaksInput ? Number(isodecMinPeaksInput.value) : null,
        isodec_min_area_covered: isodecMinAreaCoveredInput ? Number(isodecMinAreaCoveredInput.value) : null,
        isodec_mz_window_lb: isodecMzWindowLbInput ? Number(isodecMzWindowLbInput.value) : null,
        isodec_mz_window_ub: isodecMzWindowUbInput ? Number(isodecMzWindowUbInput.value) : null,
        isodec_plusone_int_window_lb: isodecPlusoneIntWindowLbInput ? Number(isodecPlusoneIntWindowLbInput.value) : null,
        isodec_plusone_int_window_ub: isodecPlusoneIntWindowUbInput ? Number(isodecPlusoneIntWindowUbInput.value) : null,
        isodec_minusone_as_zero: isodecMinusoneAsZeroInput ? isodecMinusoneAsZeroInput.checked : null,
        enable_isodec_rules: isoDecInput ? isoDecInput.checked : null,
        enable_h_transfer: hTransferInput ? hTransferInput.checked : null,
        enable_neutral_losses: neutralLossInput ? neutralLossInput.checked : null,
        precursor_calibration: precursorCalibrationInput ? precursorCalibrationInput.checked : null,
        enable_centroid: enableCentroidInput ? enableCentroidInput.checked : null,
        copies: copiesInput ? Number(copiesInput.value) : null,
        amidated: amidatedInput ? amidatedInput.checked : null,
        disulfide_bonds: disulfideBondsInput ? Number(disulfideBondsInput.value) : null,
        disulfide_map: disulfideMapInput ? disulfideMapInput.value : '',
        anchor_mode: anchorModeSelect ? anchorModeSelect.value : null,
      };
    }

    // Prefer the browsed file unless the user explicitly types a path.
    const useUpload = hasSelectedFile && !manualPathActive;
    if (useUpload && selectedFile) {
      payload.filepath = selectedFile.name;
    }

    const runPath = isPrecursor
      ? '/api/run/precursor'
      : isChargeReduced
        ? '/api/run/charge_reduced'
        : isComplexFragments
          ? '/api/run/complex_fragments'
          : isDiagnose
            ? '/api/run/diagnose'
            : isRaw
              ? '/api/run/raw'
              : '/api/run/fragments';
    const runUploadPath = isPrecursor
      ? '/api/run/precursor/upload'
      : isChargeReduced
        ? '/api/run/charge_reduced/upload'
        : isComplexFragments
          ? '/api/run/complex_fragments/upload'
          : isDiagnose
            ? '/api/run/diagnose/upload'
            : isRaw
              ? '/api/run/raw/upload'
              : '/api/run/fragments/upload';

    let response;
    if (useUpload && selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('payload', JSON.stringify(payload));
      response = await fetch(`${API_BASE}${runUploadPath}`, {
        method: 'POST',
        body: formData,
      });
    } else {
      response = await fetch(`${API_BASE}${runPath}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    }

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || 'Run failed');
    }

    const data = await response.json();
    const isPrecursorMode = isPrecursor || data.mode === 'precursor';
    const isChargeReducedMode = isChargeReduced || data.mode === 'charge_reduced';
    const isComplexFragmentsMode = isComplexFragments || data.mode === 'complex_fragments';
    const isDiagnoseMode = isDiagnose || data.mode === 'diagnose';
    const isRawMode = isRaw || data.mode === 'raw';

    const parsePlotWindow = (window) => {
      if (!window || typeof window !== 'object') return null;
      const min = toNumberOrNull(window.min);
      const max = toNumberOrNull(window.max);
      return Number.isFinite(min) && Number.isFinite(max) && min < max ? [min, max] : null;
    };

    if (isPrecursorMode) {
      lastPrecursorWindow = parsePlotWindow(data.plot_window);
      applyPrecursorResults(data);
      const precursorCandidates = Array.isArray(data.candidates) && data.candidates.length
        ? data.candidates
        : (Array.isArray(data.ambiguous_candidates) ? data.ambiguous_candidates : []);
      applySpectrum(data.spectrum, data.theory, {
        rangeOverride: lastPrecursorWindow,
        overlays: buildSpectrumOverlaysFromCandidates(precursorCandidates, '#f59e0b'),
        theoryColor: '#ef4444',
        mode: 'precursor',
      });
      const precursorSummary = data.precursor || {};
      const warnings = [];
      const ambiguousCount = Array.isArray(data.ambiguous_candidates) ? data.ambiguous_candidates.length : 0;
      const blockReasons = Array.isArray(precursorSummary.calibration_block_reasons)
        ? precursorSummary.calibration_block_reasons.filter(Boolean)
        : [];
      if (precursorSummary.search_status === 'ambiguous' || ambiguousCount) {
        warnings.push(`Precursor ambiguous: ${ambiguousCount} close candidates.`);
      } else if (!precursorSummary.match_found) {
        warnings.push('Precursor not found. Try adjusting charge range or tolerance.');
      }
      if (precursorSummary.calibration_requested && !precursorSummary.calibration_applied && blockReasons.length) {
        warnings.push(`Precursor calibration blocked: ${blockReasons.join(', ')}`);
      }
      setWarnings(warnings);
    } else if (isChargeReducedMode) {
      lastPrecursorWindow = null;
      applyChargeReducedResults(data);
      const crCandidates = Array.isArray(data.candidates) && data.candidates.length
        ? data.candidates
        : (Array.isArray(data.ambiguous_candidates) ? data.ambiguous_candidates : []);
      applySpectrum(data.spectrum, data.theory, {
        overlays: Array.isArray(data.overlays) && data.overlays.length
          ? data.overlays
          : buildSpectrumOverlaysFromCandidates(crCandidates, '#f59e0b'),
        theoryColor: '#0f172a',
        mode: 'charge_reduced',
      });
      const acceptedCount = Number(data.count || 0);
      const ambiguousCount = Array.isArray(data.ambiguous_candidates) ? data.ambiguous_candidates.length : 0;
      const warnings = [];
      if (!acceptedCount && ambiguousCount) {
        warnings.push(`Only ambiguous charge-reduced candidates found (${ambiguousCount}).`);
      } else if (!acceptedCount) {
        warnings.push('No charge-reduced matches found.');
      }
      setWarnings(warnings);
    } else if (isDiagnoseMode) {
      lastPrecursorWindow = null;
      applyDiagnoseResults(data);
      const diagnoseOverlays = buildDiagnoseOverlays(data.results);
      applySpectrum(data.spectrum, data.theory, {
        overlays: diagnoseOverlays,
        theoryColor: '#8b5cf6',
        mode: 'diagnose',
      });
      const best = data.best;
      if (best && best.ok) {
        setWarnings([]);
      } else if (best) {
        setWarnings([`Diagnose: ${best.reason || 'Not matched'}`]);
      } else {
        setWarnings(['No diagnose results found.']);
      }
    } else if (isRawMode) {
      lastPrecursorWindow = null;
      applyRawResults(data);
      applySpectrum(data.spectrum, data.theory, { mode: 'raw', theoryColor: '#0f172a' });
      setWarnings([]);
    } else {
      lastPrecursorWindow = null;
      applyFragments(
        data.fragments || [],
        data.sequence,
        isComplexFragmentsMode ? 'complex_fragments' : 'fragments',
        data.theoretical_fragments
      );
      applySpectrum(data.spectrum, data.theory);
      const count = Number(data.count || 0);
      const emptyMessage = isComplexFragmentsMode
        ? 'No complex fragments matched. Check sequence and parameters.'
        : 'No fragments matched. Check sequence and parameters.';
      setWarnings(count === 0 ? [emptyMessage] : []);
    }
    progress = 100;
    progressBar.style.width = '100%';
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Run failed';
    setWarnings([message]);
    setCoverageStatus('Run failed', true);
  } finally {
    stopRun();
  }
};

fileBrowse.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) {
    const fileName = fileInput.files[0].name;
    filePath.value = fileName;
    fileStatus.textContent = `Loaded: ${fileName}`;
    manualFilePathOverride = false;
    setWarnings([]);
  }
});

filePath.addEventListener('input', () => {
  manualFilePathOverride = Boolean(filePath.value.trim());
  if (manualFilePathOverride) {
    fileStatus.textContent = 'Using manual path';
  }
});

cosineSlider.addEventListener('input', () => {
  cosineValue.textContent = Number(cosineSlider.value).toFixed(2);
});

const applyTableFilter = () => {
  const query = resultsFilter.value.trim().toLowerCase();
  tableRows.forEach((row) => {
    const matches = row.textContent.toLowerCase().includes(query);
    row.style.display = matches ? '' : 'none';
  });
};

resultsFilter.addEventListener('input', applyTableFilter);

if (resultsView) {
  resultsView.addEventListener('change', () => {
    activeTableView = resultsView.value === 'theoretical' ? 'theoretical' : 'results';
    if (resultsSort) {
      if (activeTableView === 'theoretical') {
        resultsSort.value = getTheoreticalModeValue() === 'all' ? 'mz' : 'charge';
      } else {
        resultsSort.value = 'score';
      }
    }
    syncResultsTableControls();
    const tableData = activeTableView === 'theoretical' ? getTheoreticalTableRows() : coverageRows;
    updateResultsTable(Array.isArray(tableData) ? tableData : []);
    rerenderCoverage();
  });
}

if (theoreticalMode) {
  theoreticalMode.addEventListener('change', () => {
    if (activeTableView !== 'theoretical') {
      return;
    }
    if (resultsSort) {
      resultsSort.value = getTheoreticalModeValue() === 'all' ? 'mz' : 'charge';
    }
    syncResultsTableControls();
    updateResultsTable(getTheoreticalTableRows());
    rerenderCoverage();
  });
}

if (theoreticalIonFilter) {
  theoreticalIonFilter.addEventListener('change', () => {
    if (activeTableView !== 'theoretical') {
      return;
    }
    updateResultsTable(getTheoreticalTableRows());
    rerenderCoverage();
  });
}

if (theoreticalChargeFilter) {
  theoreticalChargeFilter.addEventListener('change', () => {
    if (activeTableView !== 'theoretical') {
      return;
    }
    updateResultsTable(getTheoreticalTableRows());
    rerenderCoverage();
  });
}

resultsSort.addEventListener('change', () => {
  const key = resultsSort.value;
  const config = getResultsTableConfig();
  const sortMeta = config.sorts[key] || config.sorts.score;
  const datasetKey = key === 'mz' ? 'sortMz' : key === 'charge' ? 'sortCharge' : 'sortScore';
  const sorted = [...tableRows].sort((a, b) => {
    const safe = (value) => (Number.isFinite(value) ? value : -Infinity);
    const aValue = safe(parseFloat(a.dataset[datasetKey]));
    const bValue = safe(parseFloat(b.dataset[datasetKey]));
    if (sortMeta.direction === 'asc') {
      return aValue - bValue;
    }
    return bValue - aValue;
  });
  const tbody = resultsTable.querySelector('tbody');
  sorted.forEach((row) => tbody.appendChild(row));
  tableRows = sorted;
});

runButton.addEventListener('click', startRun);
stopButton.addEventListener('click', stopRun);

stopButton.disabled = true;
updateSession();
setInterval(updateSession, 1000);
setWarnings(['Load a data file to begin.']);

const plotLayout = {
  margin: { l: 36, r: 10, t: 10, b: 30 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { family: 'Fira Sans, sans-serif', color: '#1e3a8a' },
  xaxis: {
    title: 'm/z',
    gridcolor: 'rgba(30, 64, 175, 0.12)',
    zeroline: false,
  },
  yaxis: {
    title: 'Intensity',
    gridcolor: 'rgba(30, 64, 175, 0.12)',
    zeroline: false,
  },
  legend: { orientation: 'h', x: 0, y: 1.12 },
};

const buildDemoSpectrum = () => {
  const mz = [];
  const intensity = [];
  for (let i = 200; i <= 2000; i += 25) {
    const value = Math.max(0, 60 + 40 * Math.sin(i / 180) + 10 * Math.cos(i / 45));
    mz.push(i);
    intensity.push(Math.round(value));
  }
  const theoryMz = mz.filter((_, index) => index % 5 === 0);
  const theoryInt = theoryMz.map((x) => Math.max(10, 35 + 25 * Math.cos(x / 200)));
  return { mz, intensity, theoryMz, theoryInt };
};

const renderSpectrumPlot = (expMz, expInt, theoryMz, theoryInt, options = {}) => {
  const rangeMin = mzMinInput ? toNumberOrNull(mzMinInput.value) : null;
  const rangeMax = mzMaxInput ? toNumberOrNull(mzMaxInput.value) : null;
  const override = Array.isArray(options.rangeOverride) ? options.rangeOverride : null;
  const overrideMin = override ? toNumberOrNull(override[0]) : null;
  const overrideMax = override ? toNumberOrNull(override[1]) : null;
  let rangeUsed = null;
  if (Number.isFinite(overrideMin) && Number.isFinite(overrideMax) && overrideMin < overrideMax) {
    rangeUsed = [overrideMin, overrideMax];
  } else if (rangeMin != null && rangeMax != null && rangeMin < rangeMax) {
    rangeUsed = [rangeMin, rangeMax];
  }
  const buildStickData = (xs, ys, factor = 1, base = 0) => {
    const xlist = [];
    const ylist = [];
    const n = Math.min(xs.length, ys.length);
    for (let i = 0; i < n; i += 1) {
      const xVal = xs[i];
      const yVal = ys[i];
      if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) continue;
      xlist.push(xVal, xVal, null);
      ylist.push(base, base + factor * yVal, null);
    }
    return { xlist, ylist };
  };
  const maxFinite = (values) => {
    let max = 0;
    values.forEach((v) => {
      if (Number.isFinite(v) && v > max) max = v;
    });
    return max;
  };
  const maxFiniteInRange = (xs, ys, min, max) => {
    let maxVal = 0;
    let found = false;
    const n = Math.min(xs.length, ys.length);
    for (let i = 0; i < n; i += 1) {
      const xVal = xs[i];
      const yVal = ys[i];
      if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) continue;
      if (xVal < min || xVal > max) continue;
      found = true;
      if (yVal > maxVal) maxVal = yVal;
    }
    return found ? maxVal : maxFinite(ys);
  };
  const expMax = rangeUsed ? maxFiniteInRange(expMz, expInt, rangeUsed[0], rangeUsed[1]) : maxFinite(expInt);
  const theoryColor = options.theoryColor || '#f59e0b';
  const rawOverlays = Array.isArray(options.overlays) ? options.overlays : [];
  let overlayList = rawOverlays.filter(
    (spec) => spec && Array.isArray(spec.mz) && Array.isArray(spec.intensity) && spec.mz.length && spec.intensity.length
  );
  if (!overlayList.length && theoryMz.length && theoryInt.length) {
    overlayList = [{ mz: theoryMz, intensity: theoryInt, color: theoryColor, label: 'Theory' }];
  }
  let overlayMax = 0;
  overlayList.forEach((spec) => {
    const mzVals = spec.mz || [];
    const intVals = spec.intensity || [];
    const maxVal = rangeUsed
      ? maxFiniteInRange(mzVals, intVals, rangeUsed[0], rangeUsed[1])
      : maxFinite(intVals);
    if (Number.isFinite(maxVal) && maxVal > overlayMax) overlayMax = maxVal;
  });
  const yMax = Math.max(expMax, overlayMax, 1);
  const expStick = buildStickData(expMz, expInt, 1, 0);
  const layout = {
    ...plotLayout,
    xaxis: {
      ...plotLayout.xaxis,
      range: rangeUsed || undefined,
    },
    yaxis: {
      ...plotLayout.yaxis,
      range: [-1.1 * yMax, 1.1 * yMax],
      zeroline: true,
      zerolinecolor: '#0f172a',
      zerolinewidth: 1,
    },
  };
  const traces = [
    {
      x: expStick.xlist,
      y: expStick.ylist,
      type: 'scatter',
      mode: 'lines',
      name: 'Experimental',
      line: { color: '#0f172a', width: 1 },
    },
  ];
  const isChargeReducedMode = options.mode === 'charge_reduced';
  const normalizeOverlayTarget = (value) => {
    const text = String(value || '').toLowerCase();
    if (text.includes('complex')) return 'Complex';
    if (text.includes('monomer')) return 'Monomer';
    return value || 'Theory';
  };
  const showOverlayLegend = isChargeReducedMode ? true : overlayList.length > 1;
  const legendSeen = new Set();
  overlayList.forEach((spec) => {
    const stick = buildStickData(spec.mz || [], spec.intensity || [], -1, 0);
    if (!stick.xlist.length) return;
    const legendName = isChargeReducedMode ? normalizeOverlayTarget(spec.target) : spec.label || 'Theory';
    let showlegend = showOverlayLegend;
    if (isChargeReducedMode) {
      const key = String(legendName).toLowerCase();
      showlegend = !legendSeen.has(key);
      legendSeen.add(key);
    }
    traces.push({
      x: stick.xlist,
      y: stick.ylist,
      type: 'scatter',
      mode: 'lines',
      name: legendName,
      showlegend,
      hoverinfo: 'skip',
      line: { color: spec.color || theoryColor, width: 1 },
    });
  });
  Plotly.newPlot(
    'spectrumPlot',
    traces,
    layout,
    { displayModeBar: false, responsive: true }
  );
};

const demoSpectrum = buildDemoSpectrum();
renderSpectrumPlot(demoSpectrum.mz, demoSpectrum.intensity, demoSpectrum.theoryMz, demoSpectrum.theoryInt);

if (spectrumResetButton) {
  spectrumResetButton.addEventListener('click', () => {
    Plotly.relayout('spectrumPlot', {
      'xaxis.autorange': true,
      'yaxis.autorange': true,
    });
  });
}

if (spectrumZoomButton) {
  spectrumZoomButton.addEventListener('click', () => {
    const min = mzMinInput ? toNumberOrNull(mzMinInput.value) : null;
    const max = mzMaxInput ? toNumberOrNull(mzMaxInput.value) : null;
    if (min != null && max != null && min < max) {
      zoomSpectrumToRange(min, max);
    } else if (modeSelect && modeSelect.value === 'precursor' && lastPrecursorWindow) {
      zoomSpectrumToRange(lastPrecursorWindow[0], lastPrecursorWindow[1]);
    } else {
      Plotly.relayout('spectrumPlot', {
        'xaxis.autorange': true,
        'yaxis.autorange': true,
      });
    }
  });
}

const sanitizeSequence = (value) => {
  const cleaned = value.toUpperCase().replace(/[^A-Z]/g, '');
  return cleaned.length ? cleaned : 'PEPTIDE';
};

const setCoverageStatus = (text, isError = false) => {
  if (!coverageStatus) {
    return;
  }
  coverageStatus.textContent = text;
  coverageStatus.style.color = isError ? '#b91c1c' : '';
};

const parseCsv = (text) => {
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length);
  if (!lines.length) {
    return [];
  }
  const headers = lines[0].split(',').map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    const row = {};
    headers.forEach((header, idx) => {
      row[header] = (values[idx] || '').trim();
    });
    return row;
  });
};

const parseCoverageRows = (text) => {
  const rows = parseCsv(text);
  if (!rows.length || !('ion_type' in rows[0]) || !('frag_len' in rows[0])) {
    throw new Error('Expected columns ion_type and frag_len');
  }
  return rows
    .map((row) => ({
      ionType: normalizeIonType(row.ion_type),
      ionSeries: normalizeIonSeries(row.series || row.ion_type),
      fragLen: Number(row.frag_len),
      fragmentIndex: toNumberOrNull(row.fragment_index),
      charge: toNumberOrNull(row.charge),
      obsMz: toNumberOrNull(row.obs_mz),
      obsInt: toNumberOrNull(row.obs_int),
      css: toNumberOrNull(row.css),
      anchorMz: toNumberOrNull(row.anchor_theory_mz),
      anchorPpm: toNumberOrNull(row.anchor_ppm),
      label: row.label || '',
    }))
    .filter((row) => row.ionType && Number.isFinite(row.fragLen) && row.fragLen > 0);
};

const escapeHtml = (value) =>
  String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');

const clampValue = (value, min, max) => Math.min(Math.max(value, min), max);

const formatCharge = (value) => (Number.isFinite(value) ? `${value}+` : '--');
const formatMz = (value) => (Number.isFinite(value) ? value.toFixed(2) : '--');
const formatCss = (value) => (Number.isFinite(value) ? value.toFixed(3) : '--');
const formatIntensity = (value) => (Number.isFinite(value) ? Math.round(value).toLocaleString() : '--');

const sortMatchesByCss = (a, b) => {
  const aCss = Number.isFinite(a.css) ? a.css : -Infinity;
  const bCss = Number.isFinite(b.css) ? b.css : -Infinity;
  if (bCss !== aCss) return bCss - aCss;
  const aInt = Number.isFinite(a.obsInt) ? a.obsInt : -Infinity;
  const bInt = Number.isFinite(b.obsInt) ? b.obsInt : -Infinity;
  return bInt - aInt;
};

const computeFragmentIndex = (ionType, fragLen, length, fragmentIndex) => {
  const ionSeries = normalizeIonSeries(ionType);
  let index = fragmentIndex;
  if (!Number.isFinite(index)) {
    if (ionSeries === 'b' || ionSeries === 'c') {
      index = fragLen - 1;
    } else if (ionSeries === 'y' || ionSeries === 'z') {
      index = length - fragLen - 1;
    }
  }
  if (!Number.isFinite(index)) {
    return null;
  }
  return Math.floor(index);
};

const getCoverageSourceRows = () => {
  if (activeTableView === 'theoretical' && Array.isArray(theoreticalRows) && theoreticalRows.length) {
    return getFilteredRawTheoreticalRows();
  }
  return coverageRows;
};

const buildCoverageGroups = (length) => {
  const limit = Math.max(length - 1, 0);
  const groupsByIonType = { b: [], c: [], y: [], z: [] };
  const groupMap = new Map();
  const sourceRows = getCoverageSourceRows();

  if (sourceRows === null) {
    for (let index = 0; index < limit; index += 1) {
      const demo = (ionType) => {
        const isNterm = ionType === 'b' || ionType === 'c';
        const fragLen = isNterm ? index + 1 : length - index - 1;
        const key = `${ionType}:${fragLen}:${index}`;
        const group = {
          key,
          ionType,
          ionSeries: ionType,
          fragLen,
          index,
          matches: [
            {
              charge: 1,
              obsMz: null,
              obsInt: null,
              css: 1,
              anchorMz: null,
              anchorPpm: null,
              label: '',
            },
          ],
        };
        groupMap.set(key, group);
        groupsByIonType[ionType].push(group);
      };
      demo('b');
      demo('c');
      demo('y');
      demo('z');
    }
    return { groupsByIonType, groupMap };
  }

  sourceRows.forEach((row) => {
    const { ionType, fragLen, fragmentIndex } = row;
    const ionSeries = row.ionSeries || normalizeIonSeries(ionType);
    if (!ionSeries || !Number.isFinite(fragLen) || fragLen <= 0 || fragLen >= length) {
      return;
    }
    const index = computeFragmentIndex(ionSeries, fragLen, length, fragmentIndex);
    if (!Number.isFinite(index) || index < 0 || index >= limit) {
      return;
    }
    const key = `${ionType || ionSeries}:${fragLen}:${index}`;
    let group = groupMap.get(key);
    if (!group) {
      group = { key, ionType: ionType || ionSeries, ionSeries, fragLen, index, matches: [] };
      groupMap.set(key, group);
      groupsByIonType[ionSeries].push(group);
    }
    group.matches.push({
      charge: row.charge,
      obsMz: row.obsMz,
      obsInt: row.obsInt,
      css: row.css,
      anchorMz: row.anchorMz,
      anchorPpm: row.anchorPpm,
      label: row.label,
      ionType: ionType || ionSeries,
      ionSeries,
      fragLen,
      theoryMz: row.theoryMz,
      theoryIntensity: row.theoryIntensity,
    });
  });

  Object.values(groupsByIonType).forEach((groups) => {
    groups.forEach((group) => group.matches.sort(sortMatchesByCss));
    groups.sort((a, b) => a.index - b.index);
  });

  return { groupsByIonType, groupMap };
};

const getMatchCenterMz = (match) => (Number.isFinite(match.obsMz) ? match.obsMz : match.anchorMz);

const buildPopoverHtml = (group) => {
  const title = `${group.ionType}${group.fragLen}`;
  const subtitle = `${group.matches.length} match${group.matches.length === 1 ? '' : 'es'}`;
  if (!group.matches.length) {
    return `<div class="popover-title">${escapeHtml(title)}</div><div class="popover-empty">No matches</div>`;
  }
  const keyAttr = escapeHtml(group.key);
  const rows = group.matches
    .map(
      (match, idx) => {
        const centerMz = getMatchCenterMz(match);
        const mzText = formatMz(centerMz);
        const cssText = formatCss(match.css);
        return `<button type="button" class="popover-item" data-key="${keyAttr}" data-match-index="${idx}"><span>#${idx + 1} ${escapeHtml(
          formatCharge(match.charge)
        )}</span><span>css ${escapeHtml(cssText)} | m/z ${escapeHtml(mzText)}</span></button>`;
      }
    )
    .join('');
  const best = group.matches[0];
  const bestLine = `Best css ${formatCss(best.css)} | intensity ${formatIntensity(best.obsInt)}`;
  const hint = 'Click a match to zoom spectrum';
  return `<div class="popover-title">${escapeHtml(title)}</div><div class="popover-subtitle">${escapeHtml(
    subtitle
  )}</div><div class="popover-subtitle">${escapeHtml(bestLine)}</div><div class="popover-subtitle">${escapeHtml(
    hint
  )}</div><div class="popover-list">${rows}</div>`;
};

const clearCoveragePopoverTimer = () => {
  if (coveragePopoverHideTimer) {
    clearTimeout(coveragePopoverHideTimer);
    coveragePopoverHideTimer = null;
  }
};

const hideCoveragePopover = () => {
  if (!coveragePopover) return;
  clearCoveragePopoverTimer();
  coveragePopoverActiveKey = null;
  coveragePopover.hidden = true;
};

const scheduleCoveragePopoverHide = (delay = 180) => {
  if (!coveragePopover) return;
  clearCoveragePopoverTimer();
  coveragePopoverHideTimer = setTimeout(() => {
    if (coveragePopover.matches(':hover')) {
      return;
    }
    hideCoveragePopover();
  }, delay);
};

const ZOOM_WIDTH_RATIO = 0.02;
const ZOOM_WIDTH_MIN = 18;
const ZOOM_WIDTH_MAX = 160;

const computeZoomRange = (centerMz) => {
  const width = clampValue(centerMz * ZOOM_WIDTH_RATIO, ZOOM_WIDTH_MIN, ZOOM_WIDTH_MAX);
  const half = width / 2;
  const min = Math.max(centerMz - half, 0);
  const max = centerMz + half;
  return [min, max];
};

const computeSpectrumMaxAbs = (minMz, maxMz) => {
  const spectrumDiv = document.getElementById('spectrumPlot');
  if (!spectrumDiv || !Array.isArray(spectrumDiv.data)) {
    return 0;
  }
  let maxAbs = 0;
  spectrumDiv.data.forEach((trace) => {
    const xs = Array.isArray(trace.x) ? trace.x : [];
    const ys = Array.isArray(trace.y) ? trace.y : [];
    const n = Math.min(xs.length, ys.length);
    for (let i = 0; i < n; i += 1) {
      const xVal = xs[i];
      const yVal = ys[i];
      if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) continue;
      if (xVal < minMz || xVal > maxMz) continue;
      const absVal = Math.abs(yVal);
      if (absVal > maxAbs) maxAbs = absVal;
    }
  });
  return maxAbs;
};

const isFragmentsLikeMode = () => {
  const mode = lastSpectrumOptions.mode;
  return (
    !mode ||
    mode === 'fragments' ||
    mode === 'complex_fragments' ||
    mode === 'precursor' ||
    mode === 'charge_reduced' ||
    mode === 'diagnose' ||
    mode === 'raw'
  );
};

const buildOverlayFromMatch = (match) => {
  const mz = Array.isArray(match?.theoryMz) ? match.theoryMz : [];
  const intensity = Array.isArray(match?.theoryIntensity) ? match.theoryIntensity : [];
  if (!mz.length || !intensity.length) {
    return null;
  }
  const label =
    match.label ||
    (match.ionType && Number.isFinite(match.fragLen) ? `${match.ionType}${match.fragLen}` : 'Theory');
  const color = match.overlayColor || lastSpectrumOptions.theoryColor || '#f59e0b';
  return { mz, intensity, color, label, target: match.target };
};

const focusSpectrumTheoryForMatch = (match) => {
  if (!lastSpectrumState || !isFragmentsLikeMode()) {
    return;
  }
  const overlay = buildOverlayFromMatch(match);
  const options = overlay
    ? { ...lastSpectrumOptions, overlays: [overlay], mode: lastSpectrumOptions.mode || 'fragments' }
    : lastSpectrumOptions;
  renderSpectrumPlot(
    lastSpectrumState.mz,
    lastSpectrumState.intensity,
    lastSpectrumState.theoryMz,
    lastSpectrumState.theoryInt,
    options
  );
};

const zoomSpectrumToRange = (minMz, maxMz) => {
  if (!Number.isFinite(minMz) || !Number.isFinite(maxMz) || minMz >= maxMz) {
    return;
  }
  const maxAbs = computeSpectrumMaxAbs(minMz, maxMz) || 1;
  const yPad = maxAbs * 1.15;
  Plotly.relayout('spectrumPlot', {
    'xaxis.range': [minMz, maxMz],
    'yaxis.range': [-yPad, yPad],
  });
};

const zoomSpectrumToMatch = (match) => {
  const centerMz = getMatchCenterMz(match);
  if (!Number.isFinite(centerMz)) {
    return;
  }
  const [minMz, maxMz] = computeZoomRange(centerMz);
  zoomSpectrumToRange(minMz, maxMz);
};

const handleCoveragePopoverClick = (event) => {
  if (!coveragePopover) return;
  const item = event.target.closest('.popover-item');
  if (!item) return;
  event.preventDefault();
  const key = item.dataset.key || coveragePopoverActiveKey;
  const matchIndex = Number(item.dataset.matchIndex);
  if (!key || !Number.isFinite(matchIndex)) {
    return;
  }
  const group = coverageGroupMap.get(String(key));
  if (!group) return;
  const match = group.matches[matchIndex];
  if (!match) return;
  focusSpectrumTheoryForMatch(match);
  zoomSpectrumToMatch(match);
  hideCoveragePopover();
};

const setActiveResultRow = (row) => {
  if (activeResultRow && activeResultRow !== row) {
    activeResultRow.classList.remove('is-active');
  }
  activeResultRow = row;
  if (activeResultRow) {
    activeResultRow.classList.add('is-active');
  }
};

const handleResultsRowClick = (event) => {
  const row = event.target.closest('tr');
  if (!row || !row.dataset) return;
  const theoryIdx = Number(row.dataset.theoryIdx);
  const theoryMatch = Number.isFinite(theoryIdx) && Array.isArray(theoreticalRows) ? theoreticalRows[theoryIdx] : null;
  if (theoryMatch) {
    focusSpectrumTheoryForMatch(theoryMatch);
    zoomSpectrumToMatch({ obsMz: theoryMatch.anchorMz, anchorMz: theoryMatch.anchorMz });
    setActiveResultRow(row);
    hideCoveragePopover();
    return;
  }
  const matchIdx = Number(row.dataset.matchIdx);
  const match = Number.isFinite(matchIdx) && Array.isArray(coverageRows) ? coverageRows[matchIdx] : null;
  if (match) {
    focusSpectrumTheoryForMatch(match);
    zoomSpectrumToMatch(match);
  } else {
    const centerMz = toNumberOrNull(row.dataset.centerMz);
    if (!Number.isFinite(centerMz)) return;
    zoomSpectrumToMatch({ obsMz: centerMz, anchorMz: centerMz });
  }
  setActiveResultRow(row);
  hideCoveragePopover();
};

const positionCoveragePopover = (gd, point, dir) => {
  if (!coveragePopover) return;
  const fullLayout = gd?._fullLayout;
  const xaxis = fullLayout?.xaxis;
  const yaxis = fullLayout?.yaxis;
  if (!xaxis || !yaxis || typeof xaxis.l2p !== 'function' || typeof yaxis.l2p !== 'function') {
    return;
  }
  // Convert data coords to pixel coords within the graph div.
  const xPx = xaxis.l2p(point.x) + xaxis._offset;
  const yPx = yaxis.l2p(point.y) + yaxis._offset;
  const parent = coveragePopover.offsetParent || gd.parentElement;
  if (!parent) return;
  const parentRect = parent.getBoundingClientRect();
  const gdRect = gd.getBoundingClientRect();
  const relX = gdRect.left - parentRect.left + xPx;
  const relY = gdRect.top - parentRect.top + yPx;
  const offset = 12;
  const width = coveragePopover.offsetWidth;
  const height = coveragePopover.offsetHeight;
  let left = dir > 0 ? relX + offset : relX - width - offset;
  let top = dir > 0 ? relY - height - offset : relY + offset;
  left = clampValue(left, 8, parentRect.width - width - 8);
  top = clampValue(top, 8, parentRect.height - height - 8);
  coveragePopover.style.left = `${left}px`;
  coveragePopover.style.top = `${top}px`;
};

const showCoveragePopover = (gd, point) => {
  if (!coveragePopover) return;
  const rawKey = point?.customdata;
  const key = Array.isArray(rawKey) ? rawKey[0] : rawKey;
  if (!key) {
    hideCoveragePopover();
    return;
  }
  const group = coverageGroupMap.get(String(key));
  if (!group) {
    hideCoveragePopover();
    return;
  }
  clearCoveragePopoverTimer();
  coveragePopoverActiveKey = String(key);
  coveragePopover.innerHTML = buildPopoverHtml(group);
  coveragePopover.hidden = false;
  const dir = group.ionType === 'b' || group.ionType === 'c' ? 1 : -1;
  positionCoveragePopover(gd, point, dir);
};

const attachCoverageHoverHandlers = (gd) => {
  if (!gd || !coveragePopover) return;
  if (typeof gd.removeAllListeners === 'function') {
    gd.removeAllListeners('plotly_hover');
    gd.removeAllListeners('plotly_unhover');
    gd.removeAllListeners('plotly_doubleclick');
  }
  gd.on('plotly_hover', (eventData) => {
    const point = eventData?.points?.[0];
    if (!point?.customdata) return;
    showCoveragePopover(gd, point);
  });
  gd.on('plotly_unhover', () => scheduleCoveragePopoverHide());
  gd.on('plotly_doubleclick', hideCoveragePopover);
};

if (coveragePopover) {
  coveragePopover.addEventListener('mouseenter', clearCoveragePopoverTimer);
  coveragePopover.addEventListener('mouseleave', () => scheduleCoveragePopoverHide(60));
  coveragePopover.addEventListener('click', handleCoveragePopoverClick);
}

if (resultsTable) {
  resultsTable.addEventListener('click', handleResultsRowClick);
  syncResultsTableControls();
}

if (visualsStack) {
  applyVisualPanelOrder(loadVisualPanelOrder());
}

swapCoverageResultsButtons.forEach((button) => {
  button.addEventListener('click', () => {
    swapVisualPanels('coverage', 'results');
  });
});

const buildFragShape = (positions, index, length, ionType, fragLenOverride) => {
  const step = positions[index + 1] - positions[index];
  const xStart = positions[index] + step / 2;
  // Display position: y/z on top (dir=1), b/c on bottom (dir=-1)
  const displayOnTop = ionType === 'y' || ionType === 'z';
  const dir = displayOnTop ? 1 : -1;
  // Fragment length calculation: b/c count from N-term, y/z count from C-term
  const isNterm = ionType === 'b' || ionType === 'c';
  const L = 0.3;
  const verticalLen = L;
  const diagonalLen = 2 * L;
  const theta = Math.PI / 3;
  const xCorner = xStart;
  const yCorner = verticalLen * dir;
  const xPeak = xStart + diagonalLen * Math.sin(theta) * dir;
  const yPeak = yCorner + diagonalLen * Math.cos(theta) * dir;
  const isBase = ionType === 'b' || ionType === 'y';
  const labelIndex = Number.isFinite(fragLenOverride) ? fragLenOverride : (isNterm ? index + 1 : length - index);
  const color = isBase ? '#22c55e' : '#3b82f6';
  const markerX = isBase ? xCorner : xPeak;
  const markerY = isBase ? yCorner : yPeak;
  return {
    x: [xStart, xCorner, xPeak],
    y: [0, yCorner, yPeak],
    markerX,
    markerY,
    labelX: markerX,
    labelY: markerY + 0.15 * dir,
    color,
    label: `${ionType}${labelIndex}`,
    hasGlow: !isBase,
  };
};

const renderCoveragePlot = (sequence) => {
  hideCoveragePopover();
  const length = sequence.length;
  const positions = Array.from({ length: length + 1 }, (_, i) => i + 1);
  const coverage = buildCoverageGroups(length);
  coverageGroupMap = coverage.groupMap;
  const fragments = coverage.groupsByIonType;
  const lineTraces = [];
  const glowTraces = [];
  const markerTraces = [];
  const annotations = [];

  sequence.split('').forEach((aa, index) => {
    annotations.push({
      text: `<b>${aa}</b>`,
      x: positions[index],
      y: 0,
      showarrow: false,
      font: { size: 16, color: '#1e3a8a', family: 'Fira Code, monospace' },
      yshift: 2,
    });
  });

  Object.entries(fragments).forEach(([ionType, groups]) => {
    groups.forEach((group) => {
      const shape = buildFragShape(positions, group.index, length, ionType, group.fragLen);
      lineTraces.push({
        x: shape.x,
        y: shape.y,
        type: 'scatter',
        mode: 'lines',
        line: { color: '#000000', width: 2.5 },
        hoverinfo: 'skip',
        showlegend: false,
      });
      if (shape.hasGlow) {
        glowTraces.push({
          x: [shape.markerX],
          y: [shape.markerY],
          type: 'scatter',
          mode: 'markers',
          marker: {
            color: shape.color,
            size: 18,
            opacity: 0.2,
          },
          hoverinfo: 'skip',
          showlegend: false,
        });
      }
      markerTraces.push({
        x: [shape.markerX],
        y: [shape.markerY],
        type: 'scatter',
        mode: 'markers',
        customdata: [group.key],
        marker: {
          color: shape.color,
          size: 10,
          line: { color: shape.color, width: 2 },
        },
        hoverinfo: 'x+y',
        hovertemplate: '<extra></extra>',
        showlegend: false,
      });
      annotations.push({
        text: shape.label,
        x: shape.labelX,
        y: shape.labelY,
        showarrow: false,
        font: { size: 9, color: shape.color, family: 'Fira Code, monospace' },
      });
    });
  });

  const traces = [...lineTraces, ...glowTraces, ...markerTraces];

  Plotly.newPlot(
    'coveragePlot',
    traces,
    {
      ...plotLayout,
      margin: { l: 20, r: 20, t: 20, b: 20 },
      hovermode: 'closest',
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#ffffff',
      xaxis: {
        visible: false,
        range: [0.5, length + 0.5],
        fixedrange: true,
      },
      yaxis: {
        visible: false,
        range: [-1.3, 1.3],
        fixedrange: true,
      },
      annotations,
      showlegend: false,
    },
    { displayModeBar: false, responsive: true }
  ).then((gd) => {
    attachCoverageHoverHandlers(gd);
  });
};

const rerenderCoverage = () => {
  renderCoveragePlot(sanitizeSequence(peptideInput.value));
};

const handleCoverageFile = (file) => {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      setCoverageRows(parseCoverageRows(String(reader.result || '')));
      setCoverageStatus(`Loaded: ${file.name}`);
      rerenderCoverage();
    } catch (error) {
      setCoverageRows([]);
      setCoverageStatus('Invalid fragments CSV', true);
      rerenderCoverage();
    }
  };
  reader.onerror = () => {
    setCoverageRows([]);
    setCoverageStatus('Failed to read CSV', true);
    rerenderCoverage();
  };
  reader.readAsText(file);
};

if (coverageLoadButton && coverageFile) {
  coverageLoadButton.addEventListener('click', () => coverageFile.click());
  coverageFile.addEventListener('change', () => {
    const [file] = coverageFile.files || [];
    if (file) {
      handleCoverageFile(file);
    }
  });
}

loadConfig().finally(rerenderCoverage);
peptideInput.addEventListener('input', rerenderCoverage);
