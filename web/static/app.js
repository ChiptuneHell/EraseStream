// Keep requests relative so reverse-proxy prefixes are preserved.
const pageBase=window.location.pathname.endsWith('/')?window.location.pathname:window.location.pathname+'/';
const socket=io({path:pageBase+'socket.io'}); const $=id=>document.getElementById(id); const video=$('video'), src=$('src'), mask=$('mask'), out=$('out');
function setMedia(){src.src='test_input/video/'+encodeURIComponent(video.value); mask.src='test_input/mask/'+encodeURIComponent(video.value); src.load();mask.load()} video.addEventListener('change',setMedia); setMedia();
$('start').onclick=()=>{ $('start').disabled=true; $('status').textContent='正在加载模型…'; $('out').removeAttribute('src'); $('bar').style.width='0%'; socket.emit('start',{video:video.value}) };
socket.on('started',d=>{$('status').textContent='生成中…';$('progress').textContent='25%';$('bar').style.width='25%'})
socket.on('progress',d=>{$('status').textContent=d.message; $('progress').textContent=d.progress+'%';$('bar').style.width=d.progress+'%';$('genline').style.width=d.progress+'%'})
socket.on('complete',d=>{out.src=d.url+'?t='+Date.now();out.load();$('status').textContent='完成';$('progress').textContent='100%';$('bar').style.width='100%';$('elapsed').textContent=d.elapsed+'s';$('fps').textContent=d.fps;$('speed').textContent=d.realtime+'×';$('frames').textContent=d.frames;$('start').disabled=false})
socket.on('job_error',d=>{$('status').textContent='错误：'+d.message;$('start').disabled=false})
function track(v){if(!v.duration)return;$('playline').style.width=(v.currentTime/v.duration*100)+'%'} src.ontimeupdate=()=>track(src);out.ontimeupdate=()=>track(out);
