// Every application URL is relative, so this page works below /session/.../proxy/5001/.
const pageBase = window.location.pathname.endsWith('/') ? window.location.pathname : `${window.location.pathname}/`;
const socket = io({ path: `${pageBase}socket.io` });
const $ = id => document.getElementById(id);
const source = $('src'), mask = $('mask'), output = $('out'), sourceSelect = $('video');
const inputCanvas = $('input-canvas');
const inputContext = inputCanvas.getContext('2d', { willReadFrequently: true });
const maskCanvas = document.createElement('canvas');
const maskContext = maskCanvas.getContext('2d', { willReadFrequently: true });
const players = [source, mask, output];
let selectedLatent = 21;
let playing = false;
let rafId = 0;
let pendingAutoplay = false;
let lastInputDraw = 0;

function ensureInputSize() {
  const width = source.videoWidth || 832;
  const height = source.videoHeight || 480;
  if (inputCanvas.width !== width || inputCanvas.height !== height) {
    inputCanvas.width = width; inputCanvas.height = height;
    maskCanvas.width = width; maskCanvas.height = height;
  }
  return { width, height };
}

function clearInputCanvas() {
  const { width, height } = ensureInputSize();
  lastInputDraw = 0;
  inputContext.fillStyle = '#0a0b0a';
  inputContext.fillRect(0, 0, width, height);
}

function drawInputOverlay(force = false) {
  if (!source.videoWidth || source.readyState < 2) return;
  const now = performance.now();
  // The source clips are 16 FPS.  Avoid doing a full 832x480 pixel blend on
  // every browser animation tick while keeping the overlay visually smooth.
  if (!force && now - lastInputDraw < 45) return;
  lastInputDraw = now;
  const { width, height } = ensureInputSize();
  inputContext.drawImage(source, 0, 0, width, height);
  if (mask.readyState < 2 || !mask.videoWidth) return;

  // Convert the white area of the mask video into a translucent orange layer.
  // The source frame remains visible underneath so the model condition is
  // understandable in a single, presentation-ready input view.
  maskContext.drawImage(mask, 0, 0, width, height);
  const frame = inputContext.getImageData(0, 0, width, height);
  const maskFrame = maskContext.getImageData(0, 0, width, height);
  const pixels = frame.data;
  const maskPixels = maskFrame.data;
  const orange = [255, 155, 74];
  for (let i = 0; i < pixels.length; i += 4) {
    const luminance = (maskPixels[i] + maskPixels[i + 1] + maskPixels[i + 2]) / 3;
    if (luminance < 20) continue;
    const alpha = Math.min(0.62, (luminance / 255) * 0.62);
    pixels[i] = pixels[i] * (1 - alpha) + orange[0] * alpha;
    pixels[i + 1] = pixels[i + 1] * (1 - alpha) + orange[1] * alpha;
    pixels[i + 2] = pixels[i + 2] * (1 - alpha) + orange[2] * alpha;
  }
  inputContext.putImageData(frame, 0, 0);
}

function setMedia() {
  pauseAll();
  const name = encodeURIComponent(sourceSelect.value);
  source.src = `long_input/video/${name}`;
  mask.src = `long_input/mask/${name}`;
  output.removeAttribute('src'); output.load();
  $('result-card').classList.remove('has-video');
  clearInputCanvas();
  source.load(); mask.load();
  $('play').disabled = true; $('reset').disabled = false; $('scrub').disabled = false;
}
sourceSelect.addEventListener('change', setMedia);
setMedia();

[source, mask].forEach(item => {
  item.addEventListener('loadedmetadata', drawInputOverlay);
  item.addEventListener('loadeddata', drawInputOverlay);
  item.addEventListener('timeupdate', drawInputOverlay);
});

document.querySelectorAll('.length-button').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.length-button').forEach(item => item.classList.remove('selected'));
  button.classList.add('selected'); selectedLatent = Number(button.dataset.latent);
}));

function duration() {
  const values = players.filter(item => item.src && Number.isFinite(item.duration) && item.duration > 0).map(item => item.duration);
  return values.length ? Math.min(...values) : 0;
}
function safeTime(item, value) { if (item.readyState > 0 && Number.isFinite(item.duration)) { try { item.currentTime = Math.min(Math.max(value, 0), item.duration); } catch (_) {} } }
function setTime(value) { players.forEach(item => safeTime(item, value)); drawInputOverlay(true); updatePlayback(); }
function align() {
  const t = source.currentTime || 0, d = duration();
  if (d && t >= d - .04) { setTime(0); return; }
  players.forEach(item => { if (item !== source && Math.abs((item.currentTime || 0) - t) > .05) safeTime(item, t); });
}
function frame() { if (!playing) return; align(); drawInputOverlay(); updatePlayback(); rafId = requestAnimationFrame(frame); }
function playAll() {
  const ready = players.filter(item => item.src && item.readyState > 0);
  if (!ready.length) return;
  playing = true; $('play').innerHTML = '<span>❚❚</span><b>Pause all</b>';
  ready.forEach(item => item.play().catch(() => {}));
  cancelAnimationFrame(rafId); rafId = requestAnimationFrame(frame);
}
function pauseAll() { playing = false; players.forEach(item => item.pause()); cancelAnimationFrame(rafId); $('play').innerHTML = '<span>▶</span><b>Play all</b>'; }
function updatePlayback() {
  const t = source.currentTime || 0, d = duration(), ratio = d ? Math.min(t / d, 1) : 0;
  $('scrub').value = Math.round(ratio * 1000); $('clock').textContent = `${t.toFixed(1)} / ${d.toFixed(1)} s`;
  $('playline').style.width = `${ratio * 100}%`; $('playback-label').textContent = `${Math.round(ratio * 100)}%`;
}

$('play').onclick = () => playing ? pauseAll() : playAll();
$('reset').onclick = () => { pauseAll(); setTime(0); };
$('scrub').oninput = event => setTime(duration() * Number(event.target.value) / 1000);
source.ontimeupdate = updatePlayback;
source.onended = () => { pauseAll(); setTime(0); };
output.onloadedmetadata = () => { if (pendingAutoplay) { pendingAutoplay = false; setTime(0); playAll(); } };

$('start').onclick = () => {
  $('start').disabled = true; $('status').textContent = 'Generating…'; $('generation-label').textContent = 'Starting';
  $('progress').textContent = '0%'; $('bar').style.width = '0%'; $('result-label').textContent = 'GENERATING';
  pauseAll(); output.removeAttribute('src'); output.load(); $('result-card').classList.remove('has-video');
  socket.emit('start', { video: sourceSelect.value, latent_frames: selectedLatent });
};
socket.on('started', () => { $('status').textContent = 'Generating…'; $('generation-label').textContent = 'Running'; $('progress').textContent = '25%'; $('bar').style.width = '25%'; });
socket.on('progress', data => { $('status').textContent = data.message; $('generation-label').textContent = data.message; $('progress').textContent = `${data.progress}%`; $('bar').style.width = `${data.progress}%`; });
socket.on('complete', data => {
  output.src = `${data.url}?t=${Date.now()}`; output.load(); $('result-card').classList.add('has-video'); $('result-label').textContent = 'READY TO PLAY';
  $('status').textContent = 'Complete · synchronized playback'; $('generation-label').textContent = 'Complete'; $('progress').textContent = '100%'; $('bar').style.width = '100%';
  $('elapsed').textContent = `${data.elapsed}s`; $('fps').textContent = data.fps; $('speed').textContent = `${data.realtime}×`; $('frames').textContent = data.frames;
  $('start').disabled = false; $('play').disabled = false; pendingAutoplay = true;
  if (output.readyState > 0) { pendingAutoplay = false; setTime(0); playAll(); }
});
socket.on('job_error', data => { $('status').textContent = `Error · ${data.message}`; $('generation-label').textContent = 'Failed'; $('result-label').textContent = 'GENERATION FAILED'; $('start').disabled = false; });

async function pollStatus() {
  try {
    const data = await (await fetch('api/status', { cache: 'no-store' })).json();
    if (data.model_ready) { $('start').disabled = false; $('start').innerHTML = '<span>✦</span> Start generation'; $('status').textContent = `GPU ready${data.preparation_seconds ? ` · prepared in ${data.preparation_seconds}s` : ''}`; $('status-dot').classList.add('ready'); return; }
    if (data.model_error) { $('status').textContent = `Model error · ${data.model_error}`; $('start').innerHTML = '<span>↻</span> Retry'; $('start').disabled = false; return; }
    setTimeout(pollStatus, 1000);
  } catch (_) { setTimeout(pollStatus, 1500); }
}
pollStatus();
