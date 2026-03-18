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
const summaryStrip = document.getElementById('summaryStrip');
const dxfActions = document.getElementById('dxfActions');
const pdfActions = document.getElementById('pdfActions');
const imageActions = document.getElementById('imageActions');

const notesInput = document.getElementById('notesInput');
const noteTheme = document.getElementById('noteTheme');
const noteAction = document.getElementById('noteAction');
const notesFree = document.getElementById('notesFree');

const NOTE_PRESETS = {
  fidelidad: [
    { value: 'Preservar el plano completo y evitar limpieza agresiva en áreas débiles.', label: 'Preservar plano completo' },
    { value: 'Reconstruir trazos finos y punteados sin deformar la geometría principal.', label: 'Reforzar trazos finos y punteados' },
    { value: 'Priorizar fidelidad visual por encima de simplificación extrema.', label: 'Priorizar fidelidad visual' },
  ],
  rotulo: [
    { value: 'Preservar y reforzar el rótulo inferior derecho y las referencias.', label: 'Preservar rótulo inferior derecho' },
    { value: 'Intentar OCR dirigido sobre rótulo, referencias y notas.', label: 'OCR dirigido a rótulo y notas' },
    { value: 'No recortar ni limpiar agresivamente el cuadro de rótulo.', label: 'Evitar limpieza agresiva del rótulo' },
  ],
  cotas: [
    { value: 'Priorizar cotas, ejes y textos numéricos aunque aumente el tiempo de proceso.', label: 'Priorizar cotas y ejes' },
    { value: 'Intentar OCR dirigido sobre cotas y textos técnicos.', label: 'OCR dirigido a cotas' },
    { value: 'Conservar líneas de cota y flechas aunque estén débiles.', label: 'Conservar líneas de cota' },
  ],
  geometria: [
    { value: 'Reconstruir perímetros, arcos y contornos incompletos del dibujo.', label: 'Reconstruir perímetros y arcos' },
    { value: 'Cerrar pequeños cortes de línea sin deformar el plano.', label: 'Cerrar cortes pequeños de línea' },
    { value: 'Priorizar ejes estructurales, marcos y contornos dominantes.', label: 'Priorizar ejes y contornos dominantes' },
  ],
};

function refreshNoteActions() {
  if (!noteTheme || !noteAction) return;
  const theme = noteTheme.value;
  const items = NOTE_PRESETS[theme] || [];
  noteAction.innerHTML = '<option value="">Sin acción extra</option>';
  items.forEach((item) => {
    const option = document.createElement('option');
    option.value = item.value;
    option.textContent = item.label;
    noteAction.appendChild(option);
  });
}

function composeNotes() {
  const parts = [];
  if (noteTheme?.value) parts.push(`Temática: ${noteTheme.options[noteTheme.selectedIndex].text}`);
  if (noteAction?.value) parts.push(`Acción sugerida: ${noteAction.value}`);
  const free = notesFree?.value?.trim();
  if (free) parts.push(`Instrucción adicional: ${free}`);
  if (notesInput) notesInput.value = parts.join(' | ');
}


const CURRENT_JOB_STORAGE_KEY = 'trazocad.currentJobId';
const MAX_UPLOAD_MB = 25;
const POLL_INTERVAL_MS = 2500;
const MAX_POLL_FAILURES = 6;
const ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'];

let currentJobId = null;
let pollHandle = null;
let pollFailureCount = 0;
let isBusy = false;
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

function formatSeconds(seconds) {
  const safeSeconds = Math.max(0, Math.round(Number(seconds) || 0));
  const mins = Math.floor(safeSeconds / 60);
  const secs = safeSeconds % 60;
  return mins > 0 ? `${mins} min ${secs.toString().padStart(2, '0')} s` : `${secs} s`;
}

function setBusy(nextBusy) {
  isBusy = nextBusy;
  submitButton.disabled = nextBusy;
  submitButton.textContent = nextBusy ? 'Procesando…' : 'Procesar';
  fileInput.disabled = nextBusy;
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
  summaryStrip.innerHTML = '';
  dxfActions.innerHTML = '';
  pdfActions.innerHTML = '';
  imageActions.innerHTML = '';
}

function stopPolling() {
  if (pollHandle) {
    clearTimeout(pollHandle);
    pollHandle = null;
  }
}

function persistCurrentJob(jobId) {
  currentJobId = jobId || null;
  if (currentJobId) {
    window.localStorage.setItem(CURRENT_JOB_STORAGE_KEY, currentJobId);
  } else {
    window.localStorage.removeItem(CURRENT_JOB_STORAGE_KEY);
  }
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
    ['Textos OCR', String(summary.textos_ocr ?? '—')],
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
      : state === 'recovering'
        ? 'El servidor está reconstruyendo el estado de la tarea. No cierres esta pestaña.'
        : state === 'missing'
          ? 'La tarea se interrumpió y ya no quedó disponible en el servidor. Conviene volver a procesar el archivo.'
          : 'Procesando el archivo en segundo plano.';
  stageLabel.textContent = `Etapa: ${stage.replaceAll('_', ' ')}`;
  jobLabel.textContent = `Job: ${payload?.job_id || '—'}`;
  const elapsed = payload?.elapsed_seconds || (pollingStartedAt ? (Date.now() - pollingStartedAt) / 1000 : 0);
  timeLabel.textContent = `Tiempo: ${formatSeconds(elapsed)}`;
  setProgress(payload?.progress || 0);
}

function validateSelectedFile(file) {
  if (!file) return 'Seleccioná un archivo antes de procesar.';
  const extension = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : '';
  if (!ALLOWED_EXTENSIONS.includes(extension)) return 'Formato no soportado. Usá PNG, JPG, BMP o TIFF.';
  if (file.size <= 0) return 'El archivo seleccionado está vacío.';
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `El archivo supera el máximo recomendado de ${MAX_UPLOAD_MB} MB.`;
  return null;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }
  if (!response.ok) {
    throw new Error(payload?.detail || payload?.message || 'La operación no se pudo completar.');
  }
  return payload;
}

async function fetchVersion() {
  try {
    const payload = await fetchJson('/version');
    if (payload?.version) versionBadge.textContent = `Versión ${payload.version}`;
  } catch (_error) {
    // No bloquea la interfaz.
  }
}

function scheduleNextPoll(delay = POLL_INTERVAL_MS) {
  stopPolling();
  if (!currentJobId) return;
  pollHandle = window.setTimeout(() => pollStatus(currentJobId), delay);
}

async function pollStatus(jobId) {
  if (!jobId) return;

  try {
    const payload = await fetchJson(`/api/process/${jobId}/status`, { cache: 'no-store' });
    pollFailureCount = 0;
    updateStatus(payload);

    if (payload.state === 'done') {
      stopPolling();
      setBusy(false);
      clearError();
      renderResults(payload.result);
      persistCurrentJob(null);
      return;
    }

    if (payload.state === 'error' || payload.state === 'missing') {
      stopPolling();
      setBusy(false);
      resetResults();
      showError(payload.error || (payload.state === 'missing' ? 'La tarea se perdió durante el proceso.' : 'La tarea terminó con error.'));
      persistCurrentJob(null);
      return;
    }

    setBusy(true);
    scheduleNextPoll(payload.state === 'recovering' ? POLL_INTERVAL_MS + 2000 : POLL_INTERVAL_MS);
  } catch (error) {
    pollFailureCount += 1;
    if (pollFailureCount >= 2) {
      statusDetail.textContent = 'Se perdió contacto momentáneamente con el servidor. Reintentando…';
    }

    if (pollFailureCount >= MAX_POLL_FAILURES) {
      stopPolling();
      setBusy(false);
      resetResults();
      showError(error.message || 'No se pudo consultar el estado del proceso.');
      return;
    }

    scheduleNextPoll(POLL_INTERVAL_MS + pollFailureCount * 1500);
  }
}

async function restoreJobIfNeeded() {
  const storedJobId = window.localStorage.getItem(CURRENT_JOB_STORAGE_KEY);
  if (!storedJobId) return;

  persistCurrentJob(storedJobId);
  pollingStartedAt = Date.now();
  setBusy(true);
  statusMessage.textContent = 'Recuperando tarea previa…';
  statusDetail.textContent = 'Se encontró una tarea anterior y se está consultando su estado.';
  stageLabel.textContent = 'Etapa: recuperando';
  jobLabel.textContent = `Job: ${storedJobId}`;
  timeLabel.textContent = 'Tiempo: 0 s';
  setProgress(5);
  await pollStatus(storedJobId);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (isBusy) return;

  clearError();
  resetResults();

  const file = fileInput.files?.[0];
  const validationError = validateSelectedFile(file);
  if (validationError) {
    showError(validationError);
    return;
  }

  composeNotes();
  const formData = new FormData(form);
  setBusy(true);
  setProgress(2);
  pollingStartedAt = Date.now();
  statusMessage.textContent = 'Enviando el archivo al servidor…';
  statusDetail.textContent = 'La tarea se va a ejecutar en segundo plano.';
  stageLabel.textContent = 'Etapa: cola';
  jobLabel.textContent = 'Job: preparando';
  timeLabel.textContent = 'Tiempo: 0 s';

  try {
    const payload = await fetchJson('/api/process', { method: 'POST', body: formData });
    persistCurrentJob(payload.job_id);
    updateStatus(payload);
    pollFailureCount = 0;
    scheduleNextPoll(500);
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
restoreJobIfNeeded();


function syncNotesUi() {
  composeNotes();
}

if (noteTheme) noteTheme.addEventListener('change', () => {
  refreshNoteActions();
  syncNotesUi();
});
if (noteAction) noteAction.addEventListener('change', syncNotesUi);
if (notesFree) notesFree.addEventListener('input', syncNotesUi);
refreshNoteActions();
syncNotesUi();
