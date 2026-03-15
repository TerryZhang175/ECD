const report = window.Q10R_MISSED_REPORT || { summary: {}, items: [] };

const summaryGrid = document.getElementById('summaryGrid');
const searchInput = document.getElementById('searchInput');
const reFilter = document.getElementById('reFilter');
const ionFilter = document.getElementById('ionFilter');
const recurringOnly = document.getElementById('recurringOnly');
const recurringList = document.getElementById('recurringList');
const reasonList = document.getElementById('reasonList');
const reList = document.getElementById('reList');
const cardGrid = document.getElementById('cardGrid');
const emptyState = document.getElementById('emptyState');
const resultCount = document.getElementById('resultCount');
const generatedAt = document.getElementById('generatedAt');
const viewer = document.getElementById('viewer');
const viewerRe = document.getElementById('viewerRe');
const viewerTitle = document.getElementById('viewerTitle');
const viewerMeta = document.getElementById('viewerMeta');
const viewerChecks = document.getElementById('viewerChecks');
const viewerImage = document.getElementById('viewerImage');
const reviewLegend = document.getElementById('reviewLegend');

const items = Array.isArray(report.items) ? report.items : [];
const summary = report.summary || {};
const REVIEW_STORAGE_KEY = 'q10r-truthscore-fn-static-review-v1';
const REVIEW_STATES = {
  true_peak: {
    value: 'true_peak',
    icon: '★',
    label: 'true peak',
    title: '真峰 / True peak',
  },
  false_peak: {
    value: 'false_peak',
    icon: '✕',
    label: 'not true',
    title: '非真峰 / Not a true peak',
  },
};
let reviewStorageAvailable = false;
let reviewState = loadReviewState();

function formatMaybeNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 'n/a';
  }
  return Number(value).toFixed(digits);
}

function humanizeReason(reason) {
  return String(reason || 'n/a').replaceAll('_', ' ');
}

function loadReviewState() {
  try {
    const raw = window.localStorage.getItem(REVIEW_STORAGE_KEY);
    reviewStorageAvailable = true;
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (error) {
    return {};
  }
}

function persistReviewState() {
  try {
    window.localStorage.setItem(REVIEW_STORAGE_KEY, JSON.stringify(reviewState));
    reviewStorageAvailable = true;
    return true;
  } catch (error) {
    reviewStorageAvailable = false;
    return false;
  }
}

function getReviewValue(itemId) {
  return reviewState[itemId] || '';
}

function setReviewValue(itemId, nextValue) {
  if (!itemId || !REVIEW_STATES[nextValue]) {
    return;
  }
  if (reviewState[itemId] === nextValue) {
    delete reviewState[itemId];
  } else {
    reviewState[itemId] = nextValue;
  }
  persistReviewState();
}

function countReviewState(targetValue) {
  return Object.values(reviewState).filter((value) => value === targetValue).length;
}

function renderReviewLegend() {
  if (!reviewLegend) {
    return;
  }
  const trueCount = countReviewState(REVIEW_STATES.true_peak.value);
  const falseCount = countReviewState(REVIEW_STATES.false_peak.value);
  reviewLegend.innerHTML = `
    <span class="legend-pill is-true">${REVIEW_STATES.true_peak.icon} ${REVIEW_STATES.true_peak.label} ${trueCount}</span>
    <span class="legend-pill is-false">${REVIEW_STATES.false_peak.icon} ${REVIEW_STATES.false_peak.label} ${falseCount}</span>
    <span class="legend-note">${reviewStorageAvailable ? 'saved in this browser' : 'session only'}</span>
  `;
}

function buildSummaryCard(label, value, note) {
  return `
    <article class="summary-card">
      <p class="stat-label">${label}</p>
      <p class="stat-value">${value}</p>
      <p class="stat-note">${note}</p>
    </article>
  `;
}

function renderSummary() {
  const totalRows = summary.totalRows || 0;
  const uniqueBaseMisses = summary.uniqueBaseMisses || 0;
  const recurringPeak = (summary.recurring && summary.recurring[0]) || { label: 'n/a', count: 0 };
  const topFragmentGate = (summary.fragmentFailedChecks && summary.fragmentFailedChecks[0]) || { label: 'n/a', count: 0 };
  const fragmentStatuses = summary.fragmentStatuses || [];
  const notInRaw = (fragmentStatuses.find((row) => row.label === 'not_in_raw') || {}).count || 0;
  const rawOnly = (fragmentStatuses.find((row) => row.label === 'raw_only') || {}).count || 0;

  summaryGrid.innerHTML = [
    buildSummaryCard('FN rows', totalRows, 'manual true rows still missed under simplified gate + truth_score@0.80'),
    buildSummaryCard('unique base FN', uniqueBaseMisses, 'normalized by ion type, position and charge'),
    buildSummaryCard('top recurring FN', recurringPeak.label, `${recurringPeak.count} rows share this false-negative pattern`),
    buildSummaryCard('top fragments gate', topFragmentGate.label, `${topFragmentGate.count} rows blocked here | not-in-raw ${notInRaw}, raw-only ${rawOnly}`),
  ].join('');
}

function renderSidebarList(target, rows, formatter) {
  target.innerHTML = rows.map((row) => formatter(row)).join('');
}

function renderSidebars() {
  renderSidebarList(recurringList, summary.recurring || [], (row) => `
    <div class="stack-row">
      <div>
        <strong>${row.label}</strong>
        <small>repeated false-negative signature</small>
      </div>
      <strong>${row.count}</strong>
    </div>
  `);

  renderSidebarList(reasonList, summary.fragmentFailedChecks || [], (row) => `
    <div class="stack-row">
      <div>
        <strong>${row.label}</strong>
        <small>fragments reject gate count</small>
      </div>
      <strong>${row.count}</strong>
    </div>
  `);

  renderSidebarList(reList, summary.byRe || [], (row) => `
    <div class="stack-row">
      <div>
        <strong>${row.label}</strong>
        <small>FN rows in this scan</small>
      </div>
      <strong>${row.count}</strong>
    </div>
  `);
}

function hydrateFilters() {
  const options = ['<option value="all">All</option>'];
  (summary.byRe || []).forEach((row) => {
    options.push(`<option value="${row.label}">${row.label}</option>`);
  });
  reFilter.innerHTML = options.join('');
}

function buildRuleMetric(check, compact = false) {
  const state = check.state || 'info';
  const metaParts = [check.targetText, check.note].filter(Boolean);
  const meta = metaParts.length ? `<small>${metaParts.join(' | ')}</small>` : '';
  return `
    <div class="metric is-${state} ${compact ? 'is-compact' : ''}">
      <span>${check.label}</span>
      <strong>${check.valueText || 'n/a'}</strong>
      ${meta}
    </div>
  `;
}

function buildMetric(label, value, digits = 2, note = '') {
  const meta = note ? `<small>${note}</small>` : '';
  return `
    <div class="metric is-neutral">
      <span>${label}</span>
      <strong>${formatMaybeNumber(value, digits)}</strong>
      ${meta}
    </div>
  `;
}

function buildChip(label, kind = 'neutral') {
  return `<span class="chip chip-${kind}">${label}</span>`;
}

function buildReviewControls(itemId) {
  const currentValue = getReviewValue(itemId);
  return `
    <div class="review-controls" role="group" aria-label="manual review">
      ${Object.values(REVIEW_STATES).map((state) => `
        <button
          class="review-button review-button-${state.value} ${currentValue === state.value ? 'is-active' : ''}"
          type="button"
          data-item-id="${itemId}"
          data-review-value="${state.value}"
          aria-label="${state.title}"
          title="${state.title}"
        >
          <span aria-hidden="true">${state.icon}</span>
        </button>
      `).join('')}
    </div>
  `;
}

function matchesFilters(item) {
  const query = searchInput.value.trim().toLowerCase();
  const reValue = reFilter.value;
  const ionValue = ionFilter.value;
  const recurringValue = recurringOnly.checked;

  if (reValue !== 'all' && item.re !== reValue) {
    return false;
  }
  if (ionValue !== 'all' && item.ionType !== ionValue) {
    return false;
  }
  if (recurringValue && Number(item.recurringCount || 0) < 2) {
    return false;
  }
  if (!query) {
    return true;
  }

  const haystack = [
    item.re,
    item.ionLabel,
    item.ionSpec,
    item.ionType,
    item.diagnose.reason,
    item.diagnose.label,
    item.diagnose.variantType,
    item.diagnose.variantSuffix,
    ...(item.diagnose.unmetChecks || []),
    ...((item.diagnose.ruleChecks || []).map((check) => check.label)),
    item.fragments?.statusLabel,
    item.fragments?.reason,
    item.fragments?.candidateLabel,
    ...(item.fragments?.unmetChecks || []),
    ...((item.fragments?.ruleChecks || []).map((check) => check.label)),
  ]
    .join(' ')
    .toLowerCase();
  return haystack.includes(query);
}

function buildCard(item) {
  const currentReview = getReviewValue(item.id);
  const reviewClass = currentReview ? `is-reviewed-${currentReview}` : '';
  const recurringBadge = Number(item.recurringCount || 0) > 1
    ? `<span class="card-badge is-hot">repeat x${item.recurringCount}</span>`
    : '';
  const diagnoseUnmet = item.diagnose.unmetChecks || [];
  const diagnoseAlert = diagnoseUnmet.length
    ? `<p class="card-alert">IsoDec unmet: ${diagnoseUnmet.join(', ')}</p>`
    : '';
  const variantChip = item.diagnose.label ? `<span class="chip chip-variant">${item.diagnose.label}</span>` : '';
  const fragments = item.fragments || {};
  const fragmentUnmet = fragments.unmetChecks || [];
  const fragmentAlert = `Fragments ${fragments.statusLabel || 'n/a'} | ${fragments.reason || 'n/a'}`;
  const fragmentChips = [
    buildChip(fragments.candidateLabel || 'n/a', 'variant'),
    ...fragmentUnmet.map((label) => buildChip(label, 'fail')),
  ].join('');

  return `
    <article class="card ${reviewClass}" data-id="${item.id}">
      <div class="card-figure">
        <img src="${item.image}" alt="${item.ionLabel} diagnose plot" loading="lazy" />
        ${buildReviewControls(item.id)}
        <div class="card-overlay">
          <span class="card-badge">${item.re}</span>
          ${recurringBadge}
        </div>
      </div>
      <div class="card-body">
        <div class="card-title-row">
          <div>
            <h3>${item.ionLabel}</h3>
            <div class="chip-row">
              <span class="chip">${item.ionSpec}</span>
              <span class="chip">series ${item.ionType}</span>
              <span class="chip">pos ${item.pos}</span>
              <span class="chip">charge ${item.charge}+</span>
              ${variantChip}
            </div>
          </div>
        </div>
        ${diagnoseAlert}
        <div class="section-label">IsoDec checks</div>
        <div class="metric-grid metric-grid--checks">
          ${(item.diagnose.ruleChecks || []).map((check) => buildRuleMetric(check, true)).join('')}
        </div>
        <div class="section-label">Fragments trace</div>
        <p class="card-alert card-alert-fragments">${fragmentAlert}</p>
        <div class="chip-row">${fragmentChips}</div>
        <div class="metric-grid">
          ${buildMetric('raw count', fragments.rawCount, 0)}
          ${buildMetric('best count', fragments.bestCount, 0)}
          ${buildMetric('manual ppm', item.manual.avgPpmError, 2)}
          ${buildMetric('manual score', item.manual.ionScore, 3)}
        </div>
      </div>
    </article>
  `;
}

function buildViewerSections(item) {
  const diagnoseChecks = (item.diagnose.ruleChecks || []).map((check) => buildRuleMetric(check, false)).join('');
  const fragmentChecks = ((item.fragments && item.fragments.ruleChecks) || []).map((check) => buildRuleMetric(check, false)).join('');
  return `
    <section class="viewer-section">
      <div class="section-label">IsoDec checks</div>
      <div class="viewer-check-grid">${diagnoseChecks}</div>
    </section>
    <section class="viewer-section">
      <div class="section-label">Fragments trace</div>
      <p class="viewer-fragment-summary">${item.fragments?.statusLabel || 'n/a'} | ${item.fragments?.reason || 'n/a'}</p>
      <div class="viewer-check-grid">${fragmentChecks}</div>
    </section>
  `;
}

function renderCards() {
  const filtered = items.filter(matchesFilters);
  resultCount.textContent = `${filtered.length} / ${items.length} cards`;
  emptyState.hidden = filtered.length !== 0;
  cardGrid.innerHTML = filtered.map((item) => buildCard(item)).join('');

  cardGrid.querySelectorAll('.review-button').forEach((button) => {
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      setReviewValue(button.dataset.itemId, button.dataset.reviewValue);
      renderReviewLegend();
      renderCards();
    });
  });

  cardGrid.querySelectorAll('.card').forEach((card) => {
    card.addEventListener('click', () => {
      const item = items.find((entry) => entry.id === card.dataset.id);
      if (!item) {
        return;
      }
      viewerRe.textContent = item.re;
      viewerTitle.textContent = `${item.ionLabel} | ${item.ionSpec}`;
      viewerMeta.textContent = [
        `variant ${item.diagnose.label || item.ionSpec}`,
        `diagnose reason ${humanizeReason(item.diagnose.reason)}`,
        `fragments ${item.fragments?.statusLabel || 'n/a'}`,
      ].join(' | ');
      viewerChecks.innerHTML = buildViewerSections(item);
      viewerImage.src = item.image;
      viewerImage.alt = `${item.ionLabel} diagnose image`;
      viewer.showModal();
    });
  });
}

function boot() {
  generatedAt.textContent = report.generatedAt || 'n/a';
  renderSummary();
  renderSidebars();
  hydrateFilters();
  renderReviewLegend();
  renderCards();

  [searchInput, reFilter, ionFilter, recurringOnly].forEach((node) => {
    node.addEventListener('input', renderCards);
    node.addEventListener('change', renderCards);
  });
}

boot();
