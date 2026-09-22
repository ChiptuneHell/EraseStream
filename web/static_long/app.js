// Every application URL is relative, so this page works below /session/.../proxy/5001/.
const pageBase = window.location.pathname.endsWith('/') ? window.location.pathname : `${window.location.pathname}/`;
const socket = io({ path: `${pageBase}socket.io` });
const $ = id => document.getElementById(id);
const source = $('src'), output = $('out'), sourceSelect = $('video');
const streamCanvas = $('stream-canvas');
const streamContext = streamCanvas.getContext('2d');
const resultCard = $('result-card');
const players = [source, output];

let selectedLatent = 21;
let playing = false;
let rafId = 0;
let pendingAutoplay = false;
let currentJobId = null;

// The stream is an ordered JPEG frame queue. The backend emits frames after
// each causal VAE block, while a short buffer keeps playback smooth between
// two comparatively slow blocks.
let streaming = false;
let streamFinished = false;
let streamJobId = null;
let streamExpectedFrames = 0;
let streamReceivedFrames = 0;
let streamFps = 16;
let streamNextFrame = 0;
let streamPlaying = false;
let streamClockOrigin = 0;
let streamRafId = 0;
const streamFrameMap = new Map();

function setMedia() {
  pauseAll();
  resetStream();
  const name = encodeURIComponent(sourceSelect.value);
  source.src = `long_preview/${name}`;
  source.poster = `long_preview/poster/${name}`;
  output.removeAttribute('src'); output.load();
  resultCard.classList.remove('has-video', 'streaming');
  $('result-label').textContent = 'WAITING FOR GENERATION';
  source.load();
  $('play').disabled = true; $('reset').disabled = false; $('scrub').disabled = false;
}
sourceSelect.addEventListener('change', setMedia);
setMedia();

document.querySelectorAll('.length-button').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.length-button').forEach(item => item.classList.remove('selected'));
  button.classList.add('selected'); selectedLatent = Number(button.dataset.latent);
}));

function duration() {
  const values = players.filter(item => item.src && Number.isFinite(item.duration) && item.duration > 0).map(item => item.duration);
  return values.length ? Math.min(...values) : 0;
}
function safeTime(item, value) { if (item.readyState > 0 && Number.isFinite(item.duration)) { try { item.currentTime = Math.min(Math.max(value, 0), item.duration); } catch (_) {} } }
function setTime(value) { players.forEach(item => safeTime(item, value)); updatePlayback(); }
function align() {
  const t = source.currentTime || 0, d = duration();
  if (d && t >= d - .04) { setTime(0); return; }
  players.forEach(item => { if (item !== source && Math.abs((item.currentTime || 0) - t) > .05) safeTime(item, t); });
}
function frame() {
  if (!playing) return;
  if (!streaming) { align(); updatePlayback(); }
  rafId = requestAnimationFrame(frame);
}

function updatePlayback() {
  if (streaming) { updateStreamPlayback(); return; }
  const t = source.currentTime || 0, d = duration(), ratio = d ? Math.min(t / d, 1) : 0;
  $('scrub').value = Math.round(ratio * 1000); $('clock').textContent = `${t.toFixed(1)} / ${d.toFixed(1)} s`;
  $('playline').style.width = `${ratio * 100}%`; $('playback-label').textContent = `${Math.round(ratio * 100)}%`;
}

function updateStreamTelemetry() {
  const expected = streamExpectedFrames || '—';
  $('frames').textContent = `${streamReceivedFrames} / ${expected}`;
}

function updateStreamPlayback() {
  const durationSeconds = streamExpectedFrames / Math.max(streamFps, 1);
  const currentSeconds = Math.min(streamNextFrame / Math.max(streamFps, 1), durationSeconds);
  const ratio = durationSeconds ? Math.min(currentSeconds / durationSeconds, 1) : 0;
  $('scrub').value = Math.round(ratio * 1000);
  $('clock').textContent = `${currentSeconds.toFixed(1)} / ${durationSeconds.toFixed(1)} s`;
  $('playline').style.width = `${ratio * 100}%`;
  $('playback-label').textContent = `${Math.round(ratio * 100)}%`;
}

function drawStreamFrame(index) {
  const image = streamFrameMap.get(index);
  if (!image) return false;
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (streamCanvas.width !== width || streamCanvas.height !== height) {
    streamCanvas.width = width; streamCanvas.height = height;
  }
  streamContext.drawImage(image, 0, 0, width, height);
  streamFrameMap.delete(index);
  return true;
}

function contiguousStreamFrames() {
  let count = 0;
  while (streamFrameMap.has(streamNextFrame + count)) count += 1;
  return count;
}

function streamFrameLoop(now) {
  if (!streamPlaying) return;
  const framePeriod = 1000 / Math.max(streamFps, 1);
  const target = Math.floor((now - streamClockOrigin) / framePeriod);
  while (streamNextFrame <= target) {
    if (!drawStreamFrame(streamNextFrame)) {
      // A causal block may take longer than the playback period. Hold the
      // last frame and restart the local clock when the next frame arrives.
      streamClockOrigin = now - streamNextFrame * framePeriod;
      $('result-label').textContent = streamFinished ? 'STREAM READY' : 'BUFFERING';
      break;
    }
    streamNextFrame += 1;
  }
  if (streamNextFrame > 0) $('result-label').textContent = streamFinished ? 'STREAM READY' : 'PLAYING · LIVE';
  // The source preview is the timing reference for the final MP4. During
  // live canvas playback, correct it from the generated frame index so a
  // slow block or a source loop never causes the two views to drift apart.
  const sourceTime = Math.min(streamNextFrame / Math.max(streamFps, 1), source.duration || 0);
  if (source.readyState > 0 && Number.isFinite(source.duration) && Math.abs(source.currentTime - sourceTime) > .06) {
    safeTime(source, sourceTime);
  }
  updateStreamPlayback();
  if (streamFinished && streamNextFrame >= streamExpectedFrames) {
    streamPlaying = false;
    return;
  }
  streamRafId = requestAnimationFrame(streamFrameLoop);
}

function startStreamPlayback() {
  if (!streaming || streamPlaying) return;
  const threshold = Math.min(8, streamExpectedFrames || 8);
  if (!streamFinished && contiguousStreamFrames() < threshold) return;
  streamPlaying = true; playing = true;
  $('play').innerHTML = '<span>❚❚</span><b>Pause all</b>';
  streamClockOrigin = performance.now() - streamNextFrame * (1000 / Math.max(streamFps, 1));
  source.play().catch(() => {});
  cancelAnimationFrame(streamRafId);
  streamRafId = requestAnimationFrame(streamFrameLoop);
  cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(frame);
}

function resetStream() {
  streamPlaying = false; streaming = false; streamFinished = false;
  streamJobId = null; streamExpectedFrames = 0; streamReceivedFrames = 0;
  streamNextFrame = 0; streamFrameMap.clear();
  cancelAnimationFrame(streamRafId);
  streamContext.clearRect(0, 0, streamCanvas.width, streamCanvas.height);
  resultCard.classList.remove('streaming');
  $('empty').style.display = '';
  updateStreamTelemetry();
}

function playAll() {
  if (streaming) { startStreamPlayback(); return; }
  const ready = players.filter(item => item.src && item.readyState > 0);
  if (!ready.length) return;
  playing = true; $('play').innerHTML = '<span>❚❚</span><b>Pause all</b>';
  ready.forEach(item => item.play().catch(() => {}));
  cancelAnimationFrame(rafId); rafId = requestAnimationFrame(frame);
}
function pauseAll() {
  playing = false; streamPlaying = false;
  players.forEach(item => item.pause());
  cancelAnimationFrame(rafId); cancelAnimationFrame(streamRafId);
  $('play').innerHTML = '<span>▶</span><b>Play all</b>';
}

$('play').onclick = () => playing ? pauseAll() : playAll();
$('reset').onclick = () => { pauseAll(); if (streaming) resetStream(); setTime(0); };
$('scrub').oninput = event => {
  if (streaming) {
    streamNextFrame = Math.round(streamExpectedFrames * Number(event.target.value) / 1000);
    updateStreamPlayback();
    return;
  }
  setTime(duration() * Number(event.target.value) / 1000);
};
source.ontimeupdate = () => { if (!streaming) updatePlayback(); };
source.onended = () => { pauseAll(); setTime(0); };
source.onloadeddata = () => { if (streamPlaying) source.play().catch(() => {}); };

function showFinalOutput() {
  pendingAutoplay = false;
  streamPlaying = false; streaming = false;
  cancelAnimationFrame(streamRafId);
  resultCard.classList.remove('streaming');
  resultCard.classList.add('has-video');
  $('result-label').textContent = 'READY TO PLAY';
  setTime(0); playAll();
}

$('start').onclick = () => {
  $('start').disabled = true; $('status').textContent = 'Generating…'; $('generation-label').textContent = 'Starting';
  $('progress').textContent = '0%'; $('bar').style.width = '0%'; $('result-label').textContent = 'GENERATING';
  pauseAll(); resetStream(); output.removeAttribute('src'); output.load(); resultCard.classList.remove('has-video');
  currentJobId = null;
  socket.emit('start', { video: sourceSelect.value, latent_frames: selectedLatent });
};

socket.on('started', data => {
  currentJobId = data.job_id;
  $('status').textContent = 'Generating…'; $('generation-label').textContent = 'Running';
  $('progress').textContent = '25%'; $('bar').style.width = '25%';
});

socket.on('stream_started', data => {
  if (currentJobId && data.job_id !== currentJobId) return;
  streamJobId = data.job_id; streaming = true; streamFinished = false;
  streamExpectedFrames = Number(data.total_frames) || 0; streamFps = Number(data.fps) || 16;
  streamReceivedFrames = 0; streamNextFrame = 0; streamFrameMap.clear();
  resultCard.classList.add('streaming'); resultCard.classList.remove('has-video');
  $('empty').style.display = 'none'; $('result-label').textContent = 'BUFFERING';
  $('play').disabled = false; updateStreamTelemetry();
});

socket.on('stream_frame', data => {
  if (data.job_id !== streamJobId) return;
  const index = Number(data.frame_index);
  if (!Number.isFinite(index) || !data.jpeg) return;
  streamReceivedFrames = Math.max(streamReceivedFrames, index + 1);
  updateStreamTelemetry();
  const image = new Image();
  image.onload = () => {
    if (data.job_id !== streamJobId || !streaming) return;
    streamFrameMap.set(index, image);
    startStreamPlayback();
  };
  image.onerror = () => { $('result-label').textContent = 'STREAM FRAME ERROR'; };
  image.src = `data:image/jpeg;base64,${data.jpeg}`;
});

socket.on('stream_finished', data => {
  if (data.job_id !== streamJobId) return;
  streamFinished = true;
  $('result-label').textContent = 'FINALIZING OUTPUT';
  startStreamPlayback();
});

socket.on('progress', data => {
  if (currentJobId && data.job_id && data.job_id !== currentJobId) return;
  $('status').textContent = data.message; $('generation-label').textContent = data.message;
  $('progress').textContent = `${data.progress}%`; $('bar').style.width = `${data.progress}%`;
});

output.onloadedmetadata = () => { if (pendingAutoplay) showFinalOutput(); };
socket.on('complete', data => {
  if (currentJobId && data.job_id && data.job_id !== currentJobId) return;
  streamFinished = true;
  output.src = `${data.url}?t=${Date.now()}`; output.load();
  $('result-label').textContent = 'LOADING FINAL VIDEO';
  $('status').textContent = 'Complete · synchronized playback'; $('generation-label').textContent = 'Complete'; $('progress').textContent = '100%'; $('bar').style.width = '100%';
  $('elapsed').textContent = `${data.elapsed}s`; $('fps').textContent = data.fps; $('speed').textContent = `${data.realtime}×`; $('frames').textContent = data.frames;
  $('start').disabled = false; $('play').disabled = false; pendingAutoplay = true;
  if (output.readyState > 0) showFinalOutput();
});

socket.on('job_error', data => {
  resetStream(); output.removeAttribute('src'); output.load(); resultCard.classList.remove('has-video');
  $('status').textContent = `Error · ${data.message}`; $('generation-label').textContent = 'Failed'; $('result-label').textContent = 'GENERATION FAILED'; $('start').disabled = false;
});

async function pollStatus() {
  try {
    const data = await (await fetch('api/status', { cache: 'no-store' })).json();
    if (data.model_ready) { $('start').disabled = false; $('start').innerHTML = '<span>✦</span> Start generation'; $('status').textContent = `GPU ready${data.preparation_seconds ? ` · prepared in ${data.preparation_seconds}s` : ''}`; $('status-dot').classList.add('ready'); return; }
    if (data.model_error) { $('status').textContent = `Model error · ${data.model_error}`; $('start').innerHTML = '<span>↻</span> Retry'; $('start').disabled = false; return; }
    setTimeout(pollStatus, 1000);
  } catch (_) { setTimeout(pollStatus, 1500); }
}
pollStatus();
