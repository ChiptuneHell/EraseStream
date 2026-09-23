// Run: node --test tests/web_long_playback.test.cjs
// Exercise real page handlers with delayed image decoding and a controllable
// media clock. GPU inference is independent of these playback regressions.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const script = fs.readFileSync(path.join(__dirname, '../web/static_long/app.js'), 'utf8');

async function page() {
  const events = {}, elements = {}, pendingImages = [], rafs = new Map();
  let raf = 0;
  class Element {
    constructor(id) {
      this.id = id;
      this.value = id === 'buffer-seconds' ? '2' : id === 'video' ? 'video.mp4' : '0';
      this.style = {}; this.dataset = {}; this.hidden = false;
      this.classList = { add() {}, remove() {}, toggle() {} };
      this.listeners = {}; this.currentTime = 0; this.readyState = 2;
      this.duration = id === 'src' ? 249 / 16 : 81 / 16;
      this.seeking = false; this.ended = false; this.paused = true;
      this.src = ''; this.plays = 0; this.drawn = [];
    }
    addEventListener(name, fn) { (this.listeners[name] ??= []).push(fn); }
    event(name) { for (const fn of this.listeners[name] || []) fn(); }
    pause() { this.paused = true; }
    async play() { this.plays++; this.paused = false; }
    load() { this.currentTime = 0; this.ended = false; }
    removeAttribute(name) { this[name] = ''; }
    getAttribute(name) { return this[name]; }
    getContext() { return { clearRect() {}, drawImage: img => this.drawn.push(img.index) }; }
  }
  const get = id => elements[id] ??= new Element(id);
  const context = vm.createContext({
    window: { location: { pathname: '/session/test/proxy/5001/' } },
    document: { getElementById: get, querySelectorAll: () => [] },
    io: options => {
      assert.equal(options.path, '/session/test/proxy/5001/socket.io');
      return { connected: true, on: (name, fn) => { events[name] = fn; }, emit() {} };
    },
    Image: class {
      naturalWidth = 832; naturalHeight = 480;
      set src(value) { this.index = Number(value.split(',')[1]); pendingImages.push(this); }
    },
    requestAnimationFrame: fn => { rafs.set(++raf, fn); return raf; },
    cancelAnimationFrame: id => rafs.delete(id),
    setTimeout() {},
    fetch: async () => ({ json: async () => ({ model_ready: true }) }),
    console,
  });
  vm.runInContext(script, context);
  const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
  await flush();
  return {
    get, events, pendingImages, flush,
    start(id = 'one', count = 81) {
      get('start').onclick();
      events.started({ job_id: id });
      events.stream_started({ job_id: id, fps: 16, total_frames: count });
    },
    async deliver(start, end, id = 'one', decode = true) {
      for (let i = start; i < end; i++) events.stream_frame({ job_id: id, frame_index: i, jpeg: String(i) });
      if (decode) { for (const image of pendingImages.splice(0)) image.onload(); }
      await flush();
    },
    async tick(time) {
      get('src').currentTime = time;
      const scheduled = [...rafs.values()]; rafs.clear();
      for (const fn of scheduled) fn();
      await flush();
    },
    complete(id = 'one', count = 81) {
      events.complete({ job_id: id, url: 'long_results/' + id + '.mp4', frames: count, fps: 24, realtime: 1.5, elapsed: 4 });
      get('out').event('canplay');
    },
  };
}

test('initial source stays paused; two seconds of decoded contiguous frames start playback', async () => {
  const p = await page();
  assert.equal(p.get('src').plays, 0);
  assert.equal(p.get('stream-canvas').hidden, true);
  assert.equal(p.get('out').hidden, true);
  p.start();
  await p.deliver(0, 31);
  assert.equal(p.get('src').plays, 0);
  assert.equal(p.get('stream-canvas').hidden, false);
  await p.deliver(31, 32, 'one', false);
  assert.equal(p.get('src').plays, 0, 'network receipt alone must not start playback');
  p.pendingImages.splice(0).forEach(img => img.onload()); await p.flush();
  assert.equal(p.get('src').paused, false);
  assert.equal(p.get('out').paused, true);
});

test('out of order image decodes do not count as contiguous buffer', async () => {
  const p = await page(); p.start();
  await p.deliver(1, 40);
  assert.equal(p.get('src').plays, 0);
  assert.equal(p.get('buffered').textContent, '0.0s');
  await p.deliver(0, 1);
  assert.equal(p.get('src').paused, false);
});

test('underrun freezes both views; one-second refill resumes; manual pause survives arriving frames', async () => {
  const p = await page(); p.start();
  await p.deliver(0, 32); await p.tick(0); await p.tick(2);
  assert.equal(p.get('src').paused, true);
  assert.equal(p.get('src').currentTime, 31 / 16);
  assert.equal(p.get('result-label').textContent, 'REBUFFERING');
  await p.deliver(32, 47);
  assert.equal(p.get('src').paused, true);
  await p.deliver(47, 48);
  assert.equal(p.get('src').paused, false);
  p.get('play').onclick();
  await p.deliver(48, 60);
  assert.equal(p.get('src').paused, true);
  p.get('play').onclick(); await p.flush();
  assert.equal(p.get('src').paused, false);
});

test('short final tail waits for all JPEG decodes after stream_finished', async () => {
  const p = await page(); p.start('one', 9);
  await p.deliver(0, 9, 'one', false);
  p.events.stream_finished({ job_id: 'one', frames: 9 });
  assert.equal(p.get('src').plays, 0);
  const images = p.pendingImages.splice(0);
  images.slice(0, 8).forEach(img => img.onload()); await p.flush();
  assert.equal(p.get('src').plays, 0);
  images[8].onload(); await p.flush();
  assert.equal(p.get('src').paused, false);
});

test('early MP4 does not interrupt streaming; end reaches 100% and exposes only final video', async () => {
  const p = await page(); p.start(); await p.deliver(0, 81);
  await p.tick(1.5); const t = p.get('src').currentTime;
  p.complete(); await p.flush();
  assert.equal(p.get('src').currentTime, t);
  assert.equal(p.get('out').hidden, true);
  assert.equal(p.get('stream-canvas').hidden, false);
  assert.equal(p.get('scrub').disabled, true);
  await p.tick(81 / 16);
  p.get('out').event('seeked');
  assert.equal(p.get('playback-label').textContent, '100%');
  assert.equal(p.get('playline').style.width, '100%');
  assert.equal(p.get('out').hidden, false);
  assert.equal(p.get('stream-canvas').hidden, true);
  assert.equal(p.get('src').paused, true);
  p.get('play').onclick(); await p.flush();
  assert.equal(p.get('src').currentTime, 0);
  assert.equal(p.get('out').currentTime, 0);
  assert.equal(p.get('src').paused, false);
  assert.equal(p.get('out').paused, false);
});

test('late MP4 preserves final streamed frame until ready', async () => {
  const p = await page(); p.start(); await p.deliver(0, 81);
  await p.tick(81 / 16);
  assert.equal(p.get('stream-canvas').hidden, false);
  assert.equal(p.get('playback-label').textContent, '100%');
  p.complete(); p.get('out').event('seeked');
  assert.equal(p.get('stream-canvas').hidden, true);
  assert.equal(p.get('out').hidden, false);
});

test('new job ignores old image decodes, old events and unrelated errors', async () => {
  const p = await page(); p.start(); await p.deliver(0, 81, 'one', false);
  const oldImages = p.pendingImages.splice(0);
  p.complete(); p.start('two');
  oldImages.forEach(img => img.onload());
  p.events.job_error({ job_id: 'one', message: 'old error' });
  p.complete('one'); await p.flush();
  assert.equal(p.get('out').hidden, true);
  assert.equal(p.get('src').paused, true);
  assert.equal(p.get('frames').textContent, '0 / 81');
  await p.deliver(0, 32, 'two');
  assert.equal(p.get('src').paused, false);
});

test('disconnect freezes playback and stale JPEG decodes cannot restart it', async () => {
  const p = await page(); p.start(); await p.deliver(0, 32, 'one', false);
  p.events.disconnect();
  p.pendingImages.splice(0).forEach(img => img.onload()); await p.flush();
  assert.equal(p.get('src').paused, true);
  assert.match(p.get('status').textContent, /Connection lost/);
});
