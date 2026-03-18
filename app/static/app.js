const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('fileInput');
const fileHelp = document.getElementById('fileHelp');
const submitButton = document.getElementById('submitButton');
const versionBadge = document.getElementById('versionBadge');
const statusMessage = document.getElementById('statusMessage');
const statusDetail = document.getElementById('statusDetail');
const stageLabel = document.getElementById('stageLabel');
const jobLabel = document.getElementById('jobLabel');
const timeLabel = document.getElementById('timeLabel');
const progressPercent = document.getElementById('progressPercent');
const progressFill = document.getElementById('progressFill');
const progressTrack = document.querySelector('.progress-track');
const errorBox = document.getElementById('errorBox');
const resultsCard = document.getElementById('resultsCard');
const dxfActions = document.getElementById('dxfActions');
const pdfActions = document.getElementById('pdfActions');
const imageActions = document.getElementById('imageActions');
const summaryStrip = document.getElementById('summaryStrip');

let pollingTimer = null;
let currentJobId = null;
let pollingStartedAt = null;

const ACTION_GROUPS = {
  dxf: [
    { key: 'abrir_dxf', label: 'Abrir DXF' },
    { key: 'dxf', label: 'Descargar DXF vectorizado', download: true },
    { key: 'dxf_nube_puntos', label: 'Descargar DXF nube de puntos', download: true },
  ],
  pdf: [
    { key: 'abrir_pdf', label: 'Abrir PDF' },
    { key: 'report', label: 'Descargar PDF', download: true },
  ],
  image: [
    { key: 'jpg', label: 'Descargar JPG', download: true },
    { key: 'png', label: 'Descargar PNG', download: true },
  ],
};

function setProgress(value) {
  const safeValue = Math.max(0, Math.min(100, Number(value) || 0));
  progressPercent.textContent = `${safeValue}%`;
  progressFill.style.width = `${safeValue}%`;
  progressTrack.setAttribute('aria-valuenow', String(safeValue));
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? 'Procesando…' : 'Procesar';
  fileInput.disabled = isBusy;
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove('hidden');
}

function clearError() {
  errorBox.textContent = '';
  errorBox.classList.add('hidden');
}

function resetResults() {
  resultsCard.classList.add('hidden');
  dxfActions.innerHTML = '';
  pdfActions.innerHTML = '';
  imageActions.innerHTML = '';
  summaryStrip.innerHTML = '';
}

function formatSeconds(value) {
  const seconds = Math.max(0, Math.round(Number(value) || 0));
  if (!seconds) return '—';
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `${minutes}m ${rest}s`;
}

function createActionLink({ href, label, download = false }) {
  const link = document.createElement('a');
  link.className = 'action-link';
  link.href = href;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  if (download) link.setAttribute('download', '');
  link.textContent = label;
  return link;
}

function createSummaryChip(label, value) {
  const chip = document.createElement('div');
  chip.className = 'summary-chip';
  const strong = document.createElement('strong');
  strong.textContent = label;
  const span = document.createElement('span');
  span.textContent = value;
  chip.append(strong, span);
  return chip;
}

function populateActions(container, definitions, downloads) {
  container.innerHTML = '';
  definitions.forEach((item) => {
    const href = downloads[item.key];
    if (!href) return;
    container.appendChild(createActionLink({ href, label: item.label, download: item.download }));
  });
}

function renderSummary(summary) {
  summaryStrip.innerHTML = '';
  if (!summary) return;
  const chips = [
    ['Hoja', summary.hoja || '—'],
    ['Orientación', summary.orientacion_documento || '—'],
    ['Líneas', String(summary.lineas ?? '—')],
    ['Contornos', String(summary.contornos ?? '—')],
    ['Precisión', summary.precision_clase || '—'],
  ];
  chips.forEach(([label, value]) => summaryStrip.appendChild(createSummaryChip(label, value)));
}

function renderResults(result) {
  const downloads = result?.downloads || {};
  populateActions(dxfActions, ACTION_GROUPS.dxf, downloads);
  populateActions(pdfActions, ACTION_GROUPS.pdf, downloads);
  populateActions(imageActions, ACTION_GROUPS.image, downloads);
  renderSummary(result?.summary);
  resultsCard.classList.remove('hidden');
}

function updateStatus(payload) {
  const state = payload?.state || 'queued';
  const stage = payload?.stage || 'cola';
  statusMessage.textContent = payload?.message || 'Sin novedades por el momento.';
  statusDetail.textContent = state === 'done'
    ? 'El proceso terminó correctamente. Ya podés abrir o descargar los archivos.'
    : state === 'error'
      ? 'La tarea terminó con error. Revisá el mensaje mostrado arriba.'
      : 'Procesando el archivo en segundo plano.';
  stageLabel.textContent = `Etapa: ${stage.replaceAll('_', ' ')}`;
  jobLabel.textContent = `Job: ${payload?.job_id || '—'}`;
  const elapsed = payload?.elapsed_seconds || (pollingStartedAt ? (Date.now() - pollingStartedAt) / 1000 : 0);
  timeLabel.textContent = `Tiempo: ${formatSeconds(elapsed)}`;
  setProgress(payload?.progress || 0);
}

async function fetchVersion() {
  try {
    const response = await fetch('/version');
    if (!response.ok) return;
    const payload = await response.json();
    if (payload?.version) versionBadge.textContent = `Versión ${payload.version}`;
  } catch (_error) {
    // no bloquea interfaz
  }
}

async function pollStatus(jobId) {
  try {
    const response = await fetch(`/api/process/${jobId}/status`, { cache: 'no-store' });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.detail || 'No se pudo consultar el estado del proceso.');

    updateStatus(payload);

    if (payload.state === 'done') {
      clearInterval(pollingTimer);
      pollingTimer = null;
      setBusy(false);
      clearError();
      renderResults(payload.result);
      return;
    }

    if (payload.state === 'error') {
      clearInterval(pollingTimer);
      pollingTimer = null;
      setBusy(false);
      resetResults();
      showError(payload.error || 'La tarea terminó con error.');
    }
  } catch (error) {
    clearInterval(pollingTimer);
    pollingTimer = null;
    setBusy(false);
    resetResults();
    showError(error.message || 'No se pudo consultar el estado del proceso.');
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearError();
  resetResults();

  const file = fileInput.files?.[0];
  if (!file) {
    showError('Seleccioná un archivo antes de procesar.');
    return;
  }

  const formData = new FormData(form);
  setBusy(true);
  setProgress(2);
  pollingStartedAt = Date.now();
  statusMessage.textContent = 'Enviando el archivo al servidor…';
  statusDetail.textContent = 'La tarea se va a ejecutar en segundo plano.';
  stageLabel.textContent = 'Etapa: cola';
  jobLabel.textContent = 'Job: preparando';
  timeLabel.textContent = 'Tiempo: 0s';

  try {
    const response = await fetch('/api/process', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload?.detail || 'No se pudo iniciar el proceso.');

    currentJobId = payload.job_id;
    updateStatus(payload);
    if (pollingTimer) clearInterval(pollingTimer);
    pollingTimer = setInterval(() => pollStatus(currentJobId), 2500);
    pollStatus(currentJobId);
  } catch (error) {
    setBusy(false);
    setProgress(0);
    showError(error.message || 'No se pudo iniciar el proceso.');
    statusMessage.textContent = 'No se pudo iniciar el proceso.';
    statusDetail.textContent = 'Revisá el archivo y volvé a intentar.';
    stageLabel.textContent = 'Etapa: sin iniciar';
    jobLabel.textContent = 'Job: —';
    timeLabel.textContent = 'Tiempo: —';
  }
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  fileHelp.textContent = file
    ? `Archivo seleccionado: ${file.name} · ${(file.size / (1024 * 1024)).toFixed(2)} MB`
    : 'Formatos compatibles: PNG, JPG, BMP y TIFF. Límite recomendado: 25 MB.';
});

fetchVersion();
