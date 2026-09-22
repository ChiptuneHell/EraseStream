// All URLs stay relative so the app also works behind /session/.../proxy/5001/.
const pageBase = window.location.pathname.endsWith('/') ? window.location.pathname : window.location.pathname + '/';
const socket = io({path: pageBase + 'socket.io'});
const $ = id => document.getElementById(id);
const video = $('video'), src = $('src'), mask = $('mask'), out = $('out');
const players = [src, mask, out];
let isPlaying = false, rafId = 0, pendingAutoplay = false;

function setMedia() {
  pauseAll();
  src.src = 'test_input/video/' + encodeURIComponent(video.value);
  mask.src = 'test_input/mask/' + encodeURIComponent(video.value);
  out.removeAttribute('src'); out.load();
  $('result-card').classList.remove('has-video');
  src.load(); mask.load();
  $('play').disabled = true; $('reset').disabled = false; $('scrub').disabled = false;
}
video.addEventListener('change', setMedia); setMedia();

function safeTime(player, value) {
  if (player.readyState > 0 && Number.isFinite(player.duration)) {
    try { player.currentTime = Math.min(Math.max(value, 0), player.duration); } catch (_) {}
  }
}
function setTime(value) { players.forEach(player => safeTime(player, value)); updatePlaybackUi(); }
function alignPlayers() {
  const master = src.currentTime || 0;
  const duration = sharedDuration();
  if (duration && master >= duration - 0.035) { setTime(0); return; }
  players.forEach(player => {
    if (player !== src && Math.abs((player.currentTime || 0) - master) > 0.045) safeTime(player, master);
  });
}
function sharedDuration() {
  const durations = players.filter(player => player.src && Number.isFinite(player.duration) && player.duration > 0).map(player => player.duration);
  return durations.length ? Math.min(...durations) : 0;
}
function tick() {
  if (isPlaying) { alignPlayers(); updatePlaybackUi(); rafId = requestAnimationFrame(tick); }
}
function playAll() {
  const playable = players.filter(player => player.src && player.readyState > 0);
  if (!playable.length) return;
  isPlaying = true; $('play').innerHTML = '<span>❚❚</span><b>Pause all</b>';
  playable.forEach(player => player.play().catch(() => {}));
  cancelAnimationFrame(rafId); rafId = requestAnimationFrame(tick);
}
function pauseAll() {
  isPlaying = false; players.forEach(player => player.pause());
  $('play').innerHTML = '<span>▶</span><b>Play all</b>'; cancelAnimationFrame(rafId);
}
function updatePlaybackUi() {
  const current = src.currentTime || 0, duration = sharedDuration();
  const ratio = duration ? Math.min(current / duration, 1) : 0;
  $('scrub').value = Math.round(ratio * 1000); $('clock').textContent = `${current.toFixed(1)} / ${duration.toFixed(1)} s`;
  $('playline').style.width = ratio * 100 + '%'; $('playback-label').textContent = Math.round(ratio * 100) + '%';
}

$('play').onclick = () => (isPlaying ? pauseAll() : playAll());
$('reset').onclick = () => { pauseAll(); setTime(0); };
$('scrub').oninput = event => { const duration = sharedDuration(); setTime(duration * Number(event.target.value) / 1000); };
src.ontimeupdate = updatePlaybackUi; src.onended = () => { pauseAll(); setTime(0); };
out.onloadedmetadata = () => { if (pendingAutoplay) { pendingAutoplay = false; setTime(0); playAll(); } };

$('start').onclick = () => {
  $('start').disabled = true; $('status').textContent = 'Generating…'; $('generation-label').textContent = 'Starting';
  $('out').removeAttribute('src'); $('result-card').classList.remove('has-video'); $('bar').style.width = '0%'; $('progress').textContent = '0%';
  pauseAll(); socket.emit('start', {video: video.value});
};
socket.on('started', () => { $('status').textContent = 'Generating…'; $('generation-label').textContent = 'Running'; $('progress').textContent = '25%'; $('bar').style.width = '25%'; });
socket.on('progress', data => { $('status').textContent = data.message; $('generation-label').textContent = data.message; $('progress').textContent = data.progress + '%'; $('bar').style.width = data.progress + '%'; });
socket.on('complete', data => {
  out.src = data.url + '?t=' + Date.now(); out.load(); $('result-card').classList.add('has-video');
  $('status').textContent = 'Complete · synchronized playback'; $('generation-label').textContent = 'Complete'; $('progress').textContent = '100%'; $('bar').style.width = '100%';
  $('elapsed').textContent = data.elapsed + 's'; $('fps').textContent = data.fps; $('speed').textContent = data.realtime + '×'; $('frames').textContent = data.frames;
  $('start').disabled = false; $('play').disabled = false; pendingAutoplay = true;
  if (out.readyState > 0) { pendingAutoplay = false; setTime(0); playAll(); }
});
socket.on('job_error', data => { $('status').textContent = 'Error · ' + data.message; $('generation-label').textContent = 'Failed'; $('start').disabled = false; });

async function pollModelStatus() {
  try {
    const data = await (await fetch('api/status', {cache: 'no-store'})).json();
    if (data.model_ready) { $('start').disabled = false; $('start').innerHTML = '<span>✦</span> Start generation'; $('status').textContent = `GPU ready${data.preparation_seconds ? ` · prepared in ${data.preparation_seconds}s` : ''}`; return; }
    if (data.model_error) { $('status').textContent = 'Model error · ' + data.model_error; $('start').innerHTML = '<span>↻</span> Retry'; $('start').disabled = false; return; }
    setTimeout(pollModelStatus, 1000);
  } catch (_) { setTimeout(pollModelStatus, 1500); }
}
pollModelStatus();
