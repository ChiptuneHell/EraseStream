// Keep requests relative so reverse-proxy prefixes are preserved.
const pageBase = window.location.pathname.endsWith('/') ? window.location.pathname : window.location.pathname + '/';
const socket = io({path: pageBase + 'socket.io'});
const $ = id => document.getElementById(id);
const video = $('video'), src = $('src'), mask = $('mask'), out = $('out');
const players = [src, mask, out];
let syncing = false;

function setMedia() {
  src.src = 'test_input/video/' + encodeURIComponent(video.value);
  mask.src = 'test_input/mask/' + encodeURIComponent(video.value);
  out.removeAttribute('src');
  src.load(); mask.load(); out.load();
  $('play').disabled = true;
  $('reset').disabled = false;
  $('scrub').disabled = false;
}
video.addEventListener('change', setMedia);
setMedia();

function setTime(value) {
  players.forEach(player => {
    if (Number.isFinite(player.duration)) player.currentTime = Math.min(value, player.duration);
  });
  updatePlaybackUi();
}
function playAll() {
  Promise.all(players.filter(player => player.src).map(player => player.play().catch(() => {})))
    .then(() => { $('play').textContent = '❚❚ 暂停'; });
}
function pauseAll() {
  players.forEach(player => player.pause());
  $('play').textContent = '▶ 播放';
}
function updatePlaybackUi() {
  const current = src.currentTime || 0;
  const duration = src.duration || out.duration || 0;
  if (!syncing) {
    syncing = true;
    players.forEach(player => {
      if (player !== src && Number.isFinite(player.duration) && Math.abs(player.currentTime - current) > 0.08)
        player.currentTime = Math.min(current, player.duration);
    });
    syncing = false;
  }
  $('scrub').value = duration ? Math.round(current / duration * 1000) : 0;
  $('clock').textContent = `${current.toFixed(1)} / ${duration.toFixed(1)} 秒`;
  $('playline').style.width = duration ? (current / duration * 100) + '%' : '0%';
}
$('play').onclick = () => (src.paused ? playAll() : pauseAll());
$('reset').onclick = () => { pauseAll(); setTime(0); };
$('scrub').oninput = event => {
  const duration = src.duration || out.duration || 0;
  setTime(duration * Number(event.target.value) / 1000);
};
src.ontimeupdate = updatePlaybackUi;
src.onended = pauseAll;

$('start').onclick = () => {
  $('start').disabled = true;
  $('status').textContent = '读取视频…';
  $('out').removeAttribute('src');
  $('bar').style.width = '0%'; $('progress').textContent = '0%';
  pauseAll(); socket.emit('start', {video: video.value});
};
socket.on('started', () => {
  $('status').textContent = '生成中…'; $('progress').textContent = '25%'; $('bar').style.width = '25%';
});
socket.on('progress', data => {
  $('status').textContent = data.message; $('progress').textContent = data.progress + '%';
  $('bar').style.width = data.progress + '%'; $('genline').style.width = data.progress + '%';
});
socket.on('complete', data => {
  out.src = data.url + '?t=' + Date.now(); out.load();
  $('status').textContent = '完成，三路视频已同步'; $('progress').textContent = '100%'; $('bar').style.width = '100%';
  $('elapsed').textContent = data.elapsed + 's'; $('fps').textContent = data.fps; $('speed').textContent = data.realtime + '×'; $('frames').textContent = data.frames;
  $('start').disabled = false; $('play').disabled = false; setTime(0); playAll();
});
socket.on('job_error', data => { $('status').textContent = '错误：' + data.message; $('start').disabled = false; });

async function pollModelStatus() {
  try {
    const response = await fetch('api/status', {cache: 'no-store'}); const data = await response.json();
    if (data.model_ready) {
      $('start').disabled = false; $('start').textContent = '开始生成';
      $('status').textContent = `GPU 已就绪${data.preparation_seconds ? `（准备 ${data.preparation_seconds}s）` : ''}`; return;
    }
    if (data.model_error) { $('status').textContent = '模型准备失败：' + data.model_error; $('start').textContent = '重试'; $('start').disabled = false; return; }
    setTimeout(pollModelStatus, 1000);
  } catch (_) { setTimeout(pollModelStatus, 1500); }
}
pollModelStatus();
