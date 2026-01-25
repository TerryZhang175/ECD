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
const isoDecInput = document.getElementById('isoDec');
const hTransferInput = document.getElementById('hTransfer');
const neutralLossInput = document.getElementById('neutralLoss');
const coverageLoadButton = document.getElementById('coverageLoadButton');
const coverageFile = document.getElementById('coverageFile');
const coverageStatus = document.getElementById('coverageStatus');
const coveragePopover = document.getElementById('coveragePopover');
const spectrumResetButton = document.getElementById('spectrumResetButton');
const spectrumZoomButton = document.getElementById('spectrumZoomButton');
const cosineSlider = document.getElementById('minCosine');
const cosineValue = document.getElementById('cosineValue');
const resultsFilter = document.getElementById('resultsFilter');
const resultsSort = document.getElementById('resultsSort');
const resultsTable = document.getElementById('resultsTable');
const ionTypeChips = Array.from(document.querySelectorAll('.chip-group .chip'));

const sessionStart = Date.now();
let runStart = null;
let runTimer = null;
let progress = 0;
let isRunning = false;
let coverageRows = null;
let tableRows = Array.from(resultsTable.querySelectorAll('tbody tr'));
let manualFilePathOverride = false;
let coverageGroupMap = new Map();
let coveragePopoverHideTimer = null;
let coveragePopoverActiveKey = null;
let activeResultRow = null;

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
    const cosineDefault =
      data.isodec_css_thresh !== undefined ? data.isodec_css_thresh : data.min_cosine;
    if (cosineSlider && cosineDefault !== undefined) {
      cosineSlider.value = cosineDefault;
      cosineValue.textContent = Number(cosineDefault).toFixed(2);
    }
    if (isoDecInput && data.enable_isodec_rules !== undefined) isoDecInput.checked = Boolean(data.enable_isodec_rules);
    if (hTransferInput && data.enable_h_transfer !== undefined) hTransferInput.checked = Boolean(data.enable_h_transfer);
    if (neutralLossInput && data.enable_neutral_losses !== undefined) neutralLossInput.checked = Boolean(data.enable_neutral_losses);
    setChipSelection(data.ion_types || []);
    setCoverageStatus('Config loaded');
  } catch (error) {
    setCoverageStatus('Config unavailable', true);
  }
};

const updateResultsTable = (fragments) => {
  const tbody = resultsTable.querySelector('tbody');
  const ranked = [...fragments].sort((a, b) => (b.css || 0) - (a.css || 0));
  tbody.innerHTML = ranked
    .map((frag) => {
      const ion = `${frag.ionType}${frag.fragLen}`;
      const mz = frag.obsMz != null ? frag.obsMz.toFixed(2) : '';
      const centerMz = Number.isFinite(frag.obsMz) ? frag.obsMz : frag.anchorMz;
      const charge = frag.charge ? `${frag.charge}+` : '';
      const intensity = frag.obsInt != null ? Math.round(frag.obsInt).toLocaleString() : '';
      const score = frag.css != null ? frag.css.toFixed(3) : '';
      const centerAttr = Number.isFinite(centerMz) ? ` data-center-mz="${centerMz}"` : '';
      return `<tr${centerAttr}>\n        <td>${ion}</td>\n        <td>${mz}</td>\n        <td>${charge}</td>\n        <td>${intensity}</td>\n        <td>${score}</td>\n      </tr>`;
    })
    .join('');
  activeResultRow = null;
  refreshTableRows();
  applyTableFilter();
  resultsSort.dispatchEvent(new Event('change'));
};

const applyFragments = (fragments, sequence) => {
  const normalized = fragments.map((frag) => ({
    ionType: frag.ion_type,
    fragLen: frag.frag_len,
    fragmentIndex: frag.fragment_index,
    charge: frag.charge,
    obsMz: frag.obs_mz,
    obsInt: frag.obs_int,
    css: frag.css,
    anchorMz: frag.anchor_theory_mz,
    anchorPpm: frag.anchor_ppm,
    label: frag.label,
  }));
  coverageRows = normalized;
  if (sequence && sequence !== peptideInput.value) {
    peptideInput.value = sequence;
  }
  setCoverageStatus(`Run: ${normalized.length} fragments`);
  updateResultsTable(normalized);
  rerenderCoverage();
};

const applySpectrum = (spectrum, theory) => {
  if (!spectrum || !Array.isArray(spectrum.mz) || !spectrum.mz.length) {
    return;
  }
  const theoryMz = theory && Array.isArray(theory.mz) ? theory.mz : [];
  const theoryInt = theory && Array.isArray(theory.intensity) ? theory.intensity : [];
  renderSpectrumPlot(spectrum.mz, spectrum.intensity, theoryMz, theoryInt);
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
    const payload = {
      filepath: filepathValue,
      scan: scanSelect ? Number(scanSelect.value) : 1,
      peptide: peptideInput.value,
      mz_min: mzMinInput ? toNumberOrNull(mzMinInput.value) : null,
      mz_max: mzMaxInput ? toNumberOrNull(mzMaxInput.value) : null,
      ion_types: getIonTypesFromChips(),
      frag_min_charge: fragMinChargeInput ? Number(fragMinChargeInput.value) : null,
      frag_max_charge: fragMaxChargeInput ? Number(fragMaxChargeInput.value) : null,
      match_tol_ppm: matchTolInput ? Number(matchTolInput.value) : null,
      min_cosine: Number(cosineSlider.value),
      isodec_css_thresh: Number(cosineSlider.value),
      enable_isodec_rules: isoDecInput ? isoDecInput.checked : null,
      enable_h_transfer: hTransferInput ? hTransferInput.checked : null,
      enable_neutral_losses: neutralLossInput ? neutralLossInput.checked : null,
      copies: copiesInput ? Number(copiesInput.value) : null,
      amidated: amidatedInput ? amidatedInput.checked : null,
      disulfide_bonds: disulfideBondsInput ? Number(disulfideBondsInput.value) : null,
      disulfide_map: disulfideMapInput ? disulfideMapInput.value : '',
    };

    // Prefer the browsed file unless the user explicitly types a path.
    const useUpload = hasSelectedFile && !manualPathActive;
    if (useUpload && selectedFile) {
      payload.filepath = selectedFile.name;
    }

    let response;
    if (useUpload && selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('payload', JSON.stringify(payload));
      response = await fetch(`${API_BASE}/api/run/fragments/upload`, {
        method: 'POST',
        body: formData,
      });
    } else {
      response = await fetch(`${API_BASE}/api/run/fragments`, {
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
    applyFragments(data.fragments || [], data.sequence);
    applySpectrum(data.spectrum, data.theory);
    const count = Number(data.count || 0);
    setWarnings(count === 0 ? ['No fragments matched. Check sequence and parameters.'] : []);
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

resultsSort.addEventListener('change', () => {
  const key = resultsSort.value;
  const sorted = [...tableRows].sort((a, b) => {
    const aCells = a.querySelectorAll('td');
    const bCells = b.querySelectorAll('td');
    const safe = (value) => (Number.isFinite(value) ? value : -Infinity);
    const values = {
      score: [safe(parseFloat(aCells[4].textContent)), safe(parseFloat(bCells[4].textContent))],
      mz: [safe(parseFloat(aCells[1].textContent)), safe(parseFloat(bCells[1].textContent))],
      charge: [safe(parseInt(aCells[2].textContent, 10)), safe(parseInt(bCells[2].textContent, 10))],
    };
    return values[key][1] - values[key][0];
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
  margin: { l: 48, r: 24, t: 24, b: 44 },
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

const renderSpectrumPlot = (expMz, expInt, theoryMz, theoryInt) => {
  const rangeMin = mzMinInput ? toNumberOrNull(mzMinInput.value) : null;
  const rangeMax = mzMaxInput ? toNumberOrNull(mzMaxInput.value) : null;
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
  const expMax = maxFinite(expInt);
  const theoryMax = maxFinite(theoryInt);
  const yMax = Math.max(expMax, theoryMax, 1);
  const expStick = buildStickData(expMz, expInt, 1, 0);
  const theoryStick = buildStickData(theoryMz, theoryInt, -1, 0);
  const layout = {
    ...plotLayout,
    xaxis: {
      ...plotLayout.xaxis,
      range: rangeMin != null && rangeMax != null && rangeMin < rangeMax ? [rangeMin, rangeMax] : undefined,
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
  if (theoryStick.xlist.length) {
    traces.push({
      x: theoryStick.xlist,
      y: theoryStick.ylist,
      type: 'scatter',
      mode: 'lines',
      showlegend: false,
      hoverinfo: 'skip',
      line: { color: '#f59e0b', width: 1 },
    });
  }
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
      Plotly.relayout('spectrumPlot', {
        'xaxis.range': [min, max],
      });
    } else {
      Plotly.relayout('spectrumPlot', {
        'xaxis.autorange': true,
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

const normalizeIonType = (value) => {
  const ion = (value || '').toLowerCase();
  if (ion.startsWith('b')) return 'b';
  if (ion.startsWith('c')) return 'c';
  if (ion.startsWith('y')) return 'y';
  if (ion.startsWith('z')) return 'z';
  return null;
};

const parseCoverageRows = (text) => {
  const rows = parseCsv(text);
  if (!rows.length || !('ion_type' in rows[0]) || !('frag_len' in rows[0])) {
    throw new Error('Expected columns ion_type and frag_len');
  }
  return rows
    .map((row) => ({
      ionType: normalizeIonType(row.ion_type),
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
  let index = fragmentIndex;
  if (!Number.isFinite(index)) {
    if (ionType === 'b' || ionType === 'c') {
      index = fragLen - 1;
    } else if (ionType === 'y' || ionType === 'z') {
      index = length - fragLen - 1;
    }
  }
  if (!Number.isFinite(index)) {
    return null;
  }
  return Math.floor(index);
};

const buildCoverageGroups = (length) => {
  const limit = Math.max(length - 1, 0);
  const groupsByIonType = { b: [], c: [], y: [], z: [] };
  const groupMap = new Map();

  if (coverageRows === null) {
    for (let index = 0; index < limit; index += 1) {
      const demo = (ionType) => {
        const isNterm = ionType === 'b' || ionType === 'c';
        const fragLen = isNterm ? index + 1 : length - index - 1;
        const key = `${ionType}:${fragLen}:${index}`;
        const group = {
          key,
          ionType,
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

  coverageRows.forEach((row) => {
    const { ionType, fragLen, fragmentIndex } = row;
    if (!ionType || !Number.isFinite(fragLen) || fragLen <= 0 || fragLen >= length) {
      return;
    }
    const index = computeFragmentIndex(ionType, fragLen, length, fragmentIndex);
    if (!Number.isFinite(index) || index < 0 || index >= limit) {
      return;
    }
    const key = `${ionType}:${fragLen}:${index}`;
    let group = groupMap.get(key);
    if (!group) {
      group = { key, ionType, fragLen, index, matches: [] };
      groupMap.set(key, group);
      groupsByIonType[ionType].push(group);
    }
    group.matches.push({
      charge: row.charge,
      obsMz: row.obsMz,
      obsInt: row.obsInt,
      css: row.css,
      anchorMz: row.anchorMz,
      anchorPpm: row.anchorPpm,
      label: row.label,
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

const zoomSpectrumToMatch = (match) => {
  const centerMz = getMatchCenterMz(match);
  if (!Number.isFinite(centerMz)) {
    return;
  }
  const [minMz, maxMz] = computeZoomRange(centerMz);
  const maxAbs = computeSpectrumMaxAbs(minMz, maxMz) || 1;
  const yPad = maxAbs * 1.15;
  Plotly.relayout('spectrumPlot', {
    'xaxis.range': [minMz, maxMz],
    'yaxis.range': [-yPad, yPad],
  });
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
  const centerMz = toNumberOrNull(row.dataset.centerMz);
  if (!Number.isFinite(centerMz)) return;
  zoomSpectrumToMatch({ obsMz: centerMz, anchorMz: centerMz });
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
}

const buildFragShape = (positions, index, length, ionType, fragLenOverride) => {
  const step = positions[index + 1] - positions[index];
  const xStart = positions[index] + step / 2;
  const isNterm = ionType === 'b' || ionType === 'c';
  const dir = isNterm ? 1 : -1;
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
      coverageRows = parseCoverageRows(String(reader.result || ''));
      setCoverageStatus(`Loaded: ${file.name}`);
      updateResultsTable(coverageRows);
      rerenderCoverage();
    } catch (error) {
      coverageRows = null;
      setCoverageStatus('Invalid fragments CSV', true);
      updateResultsTable([]);
      rerenderCoverage();
    }
  };
  reader.onerror = () => {
    coverageRows = null;
    setCoverageStatus('Failed to read CSV', true);
    updateResultsTable([]);
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
