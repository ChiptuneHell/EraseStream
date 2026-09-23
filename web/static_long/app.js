// Every application URL stays below the GPU platform's proxy prefix.
const pageBase = window.location.pathname.endsWith('/') ? window.location.pathname : window.location.pathname + '/';
const socket = io({ path: pageBase + 'socket.io' });
const $ = id => document.getElementById(id);
const source = $('src'), output = $('out'), sourceSelect = $('video');
const canvas = $('stream-canvas'), context = canvas.getContext('2d');
const bufferSelect = $('buffer-seconds');
const lengthButtons = [...document.querySelectorAll('.length-button')];

let selectedLatent = 21, modelReady = false, busy = false, awaitingStart = false;
let jobId = null, revision = 0, mode = 'idle', fps = 16, totalFrames = 0;
let nextFrame = 0, receivedFrames = 0, streamDone = false, startedOnce = false;
let requestedPlay = false, running = false, starting = false, ended = false;
let streamFailed = false, finalUrl = null, rafId = 0, stalls = 0;
let generationProgress = 0, startBufferSeconds = 2;
let clockRevision = 0;
const frames = new Map(), received = new Set();

function showView(view) {
  // Exactly one surface occupies the same fixed media box in every state.
  output.hidden = view !== 'video';
  canvas.hidden = view !== 'stream';
  $('empty').hidden = view !== 'empty';
  $('result-card').dataset.view = view;
}

function seek(media, seconds) {
  if (media.readyState < 1 || !Number.isFinite(media.duration)) return;
  const time = Math.max(0, Math.min(seconds, media.duration));
  if (Math.abs(media.currentTime - time) > .001) media.currentTime = time;
}

function stopClock() {
  clockRevision++;
  running = false;
  starting = false;
  source.pause();
  output.pause();
  cancelAnimationFrame(rafId);
}

function contiguousFrames() {
  let count = 0;
  while (frames.has(nextFrame + count)) count++;
  return count;
}

function updateControls() {
  sourceSelect.disabled = busy;
  bufferSelect.disabled = busy;
  lengthButtons.forEach(button => { button.disabled = busy; });
  $('start').disabled = !modelReady || busy || awaitingStart || !socket.connected;
  $('play').disabled = mode === 'idle' || (ended && mode === 'stream') || streamFailed;
  // Consumed JPEG frames are released; seeking/replaying becomes available
  // only once the complete MP4 is ready.
  $('reset').disabled = mode !== 'video';
  $('scrub').disabled = mode !== 'video';
  const label = ended ? 'Replay' : requestedPlay ? 'Pause all' : 'Play all';
  $('play').innerHTML = '<span>' + (requestedPlay ? '❚❚' : '▶') + '</span><b>' + label + '</b>';
}

function updatePlayback() {
  const duration = totalFrames / fps;
  const time = ended ? duration : mode === 'video' ? output.currentTime : Math.min(nextFrame / fps, duration);
  const ratio = duration ? Math.max(0, Math.min(time / duration, 1)) : 0;
  $('clock').textContent = time.toFixed(1) + ' / ' + duration.toFixed(1) + ' s';
  $('scrub').value = Math.round(ratio * 1000);
  $('playline').style.width = ratio * 100 + '%';
  $('playback-label').textContent = Math.round(ratio * 100) + '%';
  $('played').textContent = String(ended ? totalFrames : mode === 'video' ? Math.min(totalFrames, Math.floor(time * fps)) : nextFrame);
  const buffer = mode === 'stream' ? contiguousFrames() / fps : 0;
  $('buffered').textContent = buffer.toFixed(1) + 's';
  $('buffered').title = 'Automatic buffering pauses: ' + stalls;
}

function updateFrames() {
  $('frames').textContent = receivedFrames + ' / ' + totalFrames;
  updatePlayback();
}

function setProgress(value, message) {
  generationProgress = Math.max(generationProgress, Math.min(100, Number(value) || 0));
  $('bar').style.width = generationProgress + '%';
  $('progress').textContent = generationProgress + '%';
  $('generation-label').textContent = message;
}

function resetPlayback() {
  revision++;
  stopClock();
  jobId = null;
  mode = 'idle';
  totalFrames = nextFrame = receivedFrames = stalls = 0;
  requestedPlay = streamDone = startedOnce = ended = streamFailed = false;
  finalUrl = null;
  frames.clear();
  received.clear();
  output.removeAttribute('src');
  output.load();
  source.loop = output.loop = false;
  seek(source, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  showView('empty');
  updatePlayback();
}

function setMedia() {
  if (busy) return;
  resetPlayback();
  const name = encodeURIComponent(sourceSelect.value);
  source.src = 'long_preview/' + name;
  source.poster = 'long_preview/poster/' + name;
  source.load(); // Remains paused until generated frames are buffered.
  $('result-label').textContent = 'WAITING FOR GENERATION';
  updateControls();
}

function draw(image) {
  if (canvas.width !== image.naturalWidth || canvas.height !== image.naturalHeight) {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
  }
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  showView('stream');
}

function bufferStream() {
  const wasRunning = running;
  stopClock();
  if (wasRunning) stalls++;
  // Keep the source on the same frame as the canvas while the queue refills.
  seek(source, Math.max(0, nextFrame - 1) / fps);
  $('result-label').textContent = startedOnce ? 'REBUFFERING' : 'BUFFERING';
  updatePlayback();
  updateControls();
}

function finishPlayback() {
  stopClock();
  ended = true;
  requestedPlay = false;
  if (mode === 'stream') nextFrame = totalFrames;
  seek(source, Math.max(0, totalFrames - 1) / fps);
  $('result-label').textContent = finalUrl ? 'PLAYBACK COMPLETE' : 'SAVING FINAL VIDEO';
  updatePlayback();
  updateControls();
  tryFinalVideo();
}

function tick() {
  if (!running) return;
  if (mode === 'stream') {
    // The decoded source media clock drives the canvas, rather than a second
    // wall clock plus repeated seeks of the source (which caused jitter).
    if (!source.seeking && source.readyState >= 2) {
      const target = Math.min(totalFrames - 1, Math.floor((source.currentTime + .0001) * fps));
      while (nextFrame <= target) {
        const image = frames.get(nextFrame);
        if (!image) { bufferStream(); return; }
        draw(image);
        frames.delete(nextFrame++);
      }
      if (nextFrame === totalFrames && source.currentTime >= totalFrames / fps - .005) {
        finishPlayback();
        return;
      }
    }
  } else if (mode === 'video') {
    if (output.ended || output.currentTime >= totalFrames / fps - .005) {
      finishPlayback();
      return;
    }
    if (!output.seeking && !source.seeking && Math.abs(source.currentTime - output.currentTime) > .12) {
      seek(source, output.currentTime);
    }
  }
  updatePlayback();
  rafId = requestAnimationFrame(tick);
}

async function startClocks(media) {
  const token = revision;
  const clockToken = ++clockRevision;
  starting = true;
  try {
    await Promise.all(media.map(item => item.play()));
    if (token !== revision || clockToken !== clockRevision || !requestedPlay || ended || streamFailed || !starting) return;
    starting = false;
    running = true;
    startedOnce = true;
    $('result-label').textContent = mode === 'stream' ? 'PLAYING · LIVE' : 'SYNCHRONIZED PLAYBACK';
    updateControls();
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(tick);
  } catch (_) {
    if (token !== revision || clockToken !== clockRevision) return;
    stopClock();
    requestedPlay = false;
    $('result-label').textContent = 'PRESS PLAY TO CONTINUE';
    updateControls();
  }
}

function maybePlay() {
  if (!requestedPlay || running || starting || ended || streamFailed || mode === 'idle') return;
  if (source.readyState < 2 || source.seeking) return;
  if (mode === 'stream') {
    const count = contiguousFrames(), remaining = totalFrames - nextFrame;
    if (remaining <= 0) { finishPlayback(); return; }
    const seconds = startedOnce ? Math.min(startBufferSeconds, 1) : startBufferSeconds;
    const threshold = Math.min(remaining, Math.max(1, Math.ceil(seconds * fps)));
    // stream_finished means all frames were sent, not that JPEGs have all
    // decoded. Only the complete contiguous tail may bypass the threshold.
    if (count < threshold && !(streamDone && count === remaining)) {
      $('result-label').textContent = startedOnce ? 'REBUFFERING' : 'BUFFERING · ' + startBufferSeconds + 's';
      return;
    }
    startClocks([source]);
  } else if (output.readyState >= 2 && !output.seeking) {
    startClocks([source, output]);
  }
}

function tryFinalVideo() {
  // Never interrupt the first streaming pass just because MP4 writing ended.
  if (mode !== 'stream' || !finalUrl || (!ended && !streamFailed) || output.readyState < 2 || output.seeking) return;
  const target = ended ? Math.max(0, totalFrames - 1) / fps : Math.max(0, nextFrame - 1) / fps;
  if (Math.abs(output.currentTime - target) > .5 / fps) { seek(output, target); return; }
  stopClock();
  requestedPlay = false;
  streamFailed = false;
  mode = 'video';
  frames.clear();
  showView('video');
  seek(source, target);
  $('result-label').textContent = ended ? 'PLAYBACK COMPLETE' : 'READY TO PLAY';
  updatePlayback();
  updateControls();
}

function failStream(message) {
  stopClock();
  requestedPlay = false;
  streamFailed = true;
  $('status').textContent = message;
  $('result-label').textContent = 'STREAM INTERRUPTED';
  updateControls();
  tryFinalVideo();
}

function accepts(data) { return Boolean(jobId && data.job_id === jobId); }

sourceSelect.addEventListener('change', setMedia);
lengthButtons.forEach(button => button.addEventListener('click', () => {
  if (busy) return;
  lengthButtons.forEach(item => item.classList.remove('selected'));
  button.classList.add('selected');
  selectedLatent = Number(button.dataset.latent);
}));

$('play').onclick = () => {
  if (requestedPlay) {
    requestedPlay = false;
    stopClock();
    $('result-label').textContent = 'PAUSED';
  } else {
    if (ended && mode === 'video') {
      ended = false;
      seek(source, 0);
      seek(output, 0);
    }
    requestedPlay = true;
    maybePlay();
  }
  updateControls();
};

$('reset').onclick = () => {
  if (mode !== 'video') return;
  requestedPlay = false;
  ended = false;
  stopClock();
  seek(source, 0);
  seek(output, 0);
  updatePlayback();
  updateControls();
};
$('scrub').oninput = event => {
  if (mode !== 'video') return;
  const time = totalFrames / fps * Number(event.target.value) / 1000;
  stopClock();
  ended = false;
  seek(source, time);
  seek(output, time);
  updatePlayback();
  maybePlay();
};

for (const name of ['loadeddata', 'canplay', 'seeked']) {
  source.addEventListener(name, maybePlay);
  output.addEventListener(name, () => { tryFinalVideo(); maybePlay(); });
}
source.addEventListener('ended', () => {
  if (mode === 'stream' && nextFrame >= totalFrames - 1 && frames.has(totalFrames - 1)) {
    draw(frames.get(totalFrames - 1));
    frames.delete(totalFrames - 1);
    nextFrame = totalFrames;
  }
  if (mode === 'stream' && nextFrame === totalFrames) finishPlayback();
  else if (mode === 'stream') failStream('Source preview ended before the generated video. Rebuild the preview.');
});
output.addEventListener('ended', () => { if (mode === 'video') finishPlayback(); });
for (const media of [source, output]) {
  media.addEventListener('waiting', () => {
    if (!running || (mode === 'stream' && media === output)) return;
    stopClock();
    $('result-label').textContent = 'BUFFERING VIDEO';
  });
  media.addEventListener('error', () => {
    if (!media.getAttribute('src')) return;
    if (media === source) failStream('Source preview could not be loaded.');
    else {
      $('status').textContent = 'Final video could not be loaded. The live preview is still available.';
      finalUrl = null;
    }
  });
}

$('start').onclick = () => {
  if (busy || !modelReady || !socket.connected) return;
  resetPlayback();
  busy = awaitingStart = true;
  generationProgress = 0;
  startBufferSeconds = Number(bufferSelect.value) || 2;
  for (const id of ['fps', 'speed', 'elapsed', 'frames']) $(id).textContent = '—';
  setProgress(0, 'Starting');
  $('status').textContent = 'Preparing input…';
  $('result-label').textContent = 'PREPARING INPUT';
  updateControls();
  socket.emit('start', { video: sourceSelect.value, latent_frames: selectedLatent });
};

socket.on('started', data => {
  if (!awaitingStart) return;
  awaitingStart = false;
  jobId = data.job_id;
  $('status').textContent = 'Generating…';
  setProgress(0, 'Preparing input');
  updateControls();
});
function handleStreamStarted(data) {
  if (!accepts(data)) return;
  mode = 'stream';
  totalFrames = Number(data.total_frames);
  fps = Number(data.fps) || 16;
  requestedPlay = true;
  $('result-label').textContent = 'BUFFERING · ' + startBufferSeconds + 's';
  updateFrames();
  updateControls();
}
function handleStreamFrame(data) {
  if (!accepts(data) || mode !== 'stream' || streamFailed) return;
  const index = Number(data.frame_index), token = revision;
  if (!Number.isInteger(index) || index < 0 || index >= totalFrames || received.has(index)) return;
  received.add(index);
  receivedFrames = received.size;
  updateFrames();
  const image = new Image();
  image.onload = () => {
    if (token !== revision || !accepts(data) || mode !== 'stream' || streamFailed) return;
    frames.set(index, image);
    if (index === 0 && !startedOnce) draw(image); // Still image until buffer is ready.
    updatePlayback();
    maybePlay();
  };
  image.onerror = () => {
    if (token === revision && accepts(data)) failStream('A streamed frame could not be decoded. Waiting for the final video.');
  };
  image.src = 'data:image/jpeg;base64,' + data.jpeg;
}
function handleStreamFinished(data) {
  if (!accepts(data)) return;
  streamDone = true;
  if (Number(data.frames) !== totalFrames) {
    failStream('Stream frame count differs from the selected length. Waiting for the final video.');
    return;
  }
  maybePlay();
}
socket.on('stream_started', handleStreamStarted);
socket.on('stream_frame', handleStreamFrame);
socket.on('stream_finished', handleStreamFinished);
socket.on('progress', data => {
  if (!accepts(data)) return;
  $('status').textContent = data.message;
  setProgress(data.progress, data.message);
  if (Number.isFinite(data.fps)) $('fps').textContent = data.fps.toFixed(2);
  if (Number.isFinite(data.realtime)) $('speed').textContent = data.realtime.toFixed(2) + '×';
  if (Number.isFinite(data.elapsed)) $('elapsed').textContent = data.elapsed.toFixed(2) + 's';
});
socket.on('complete', data => {
  if (!accepts(data)) return;
  busy = false;
  streamDone = true;
  finalUrl = data.url;
  $('status').textContent = 'Generation complete';
  setProgress(100, 'Complete');
  $('elapsed').textContent = data.elapsed + 's';
  $('fps').textContent = data.fps;
  $('speed').textContent = data.realtime + '×';
  $('frames').textContent = data.frames + ' / ' + data.frames;
  output.src = finalUrl + '?t=' + Date.now();
  output.load(); // Preload hidden; switch only when this streaming pass ends.
  updateControls();
  maybePlay();
  tryFinalVideo();
});
socket.on('job_error', data => {
  if (data.job_id ? !accepts(data) : !awaitingStart) return;
  busy = awaitingStart = false;
  failStream('Error · ' + data.message);
  jobId = null; // Ignore in-flight JPEG image decodes after a failed job.
  $('generation-label').textContent = 'Failed';
  updateControls();
});
socket.on('disconnect', () => {
  if (busy) {
    busy = awaitingStart = false;
    failStream('Connection lost. Reconnect and start a new generation.');
    jobId = null;
  }
  updateControls();
});
socket.on('connect', () => { updateControls(); pollStatus(); });

let statusPolling = false;
async function pollStatus() {
  if (statusPolling) return;
  statusPolling = true;
  let retry = false;
  try {
    const data = await (await fetch('api/status', { cache: 'no-store' })).json();
    modelReady = Boolean(data.model_ready);
    if (!busy && mode === 'idle') {
      $('status').textContent = modelReady ? 'GPU ready' : data.model_error ? 'Model error · ' + data.model_error : 'Preparing GPU';
      $('start').innerHTML = modelReady ? '<span>✦</span> Start generation' : 'Preparing GPU…';
    }
    $('status-dot').classList.toggle('ready', modelReady);
    retry = !modelReady && !data.model_error;
    updateControls();
  } catch (_) { retry = true; }
  statusPolling = false;
  if (retry) setTimeout(pollStatus, 1500);
}

setMedia();
pollStatus();
