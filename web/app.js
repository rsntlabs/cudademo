import { ParticleRenderer } from './renderer.js';

const $ = id => document.getElementById(id);
const source = $('cuda');
const output = $('wgsl');
const presets = { galaxy: ['particles.cu', 'Spiral galaxy'], vortex: ['vortex.cu', 'Orbital vortex'], wave: ['wave.cu', 'Wave field'] };
const cache = new Map();
let renderer;
let worker;
let compilerReady = false;
let currentRequest = 0;
let presetRequest = 0;
let pending = new Map();
let appliedSource = '';
let compiling = false;
let initializing = true;
let revision = '';

function status(message, state = 'loading') {
  $('status').textContent = message;
  $('status-bar').dataset.state = state;
  $('status-symbol').textContent = { error: '!', success: '✓', dirty: '○', loading: '◌' }[state];
}

function lines(textarea) {
  const gutter = $(textarea.id + '-lines');
  gutter.textContent = Array.from({ length: textarea.value.split('\n').length }, (_, i) => i + 1).join('\n');
  gutter.scrollTop = textarea.scrollTop;
}

function showStageError(message) {
  const box = $('stage-message');
  box.replaceChildren();
  const heading = document.createElement('strong');
  heading.textContent = 'The particle field is taking a break.';
  const detail = document.createElement('span');
  detail.textContent = message;
  box.append(heading, detail);
  box.hidden = false;
  $('preview-state').textContent = 'PREVIEW UNAVAILABLE';
  $('live-dot').classList.remove('active');
}

function startCompiler() {
  worker?.terminate();
  compilerReady = false;
  worker = new Worker(new URL('./compiler-worker.js', import.meta.url), { type: 'module' });
  const ready = new Promise((resolve, reject) => {
    const bootTimer = setTimeout(() => reject(new Error('The WebAssembly compiler took too long to load. Reload the page to retry.')), 20000);
    worker.onmessage = ({ data }) => {
      if (data.type === 'ready') {
        clearTimeout(bootTimer); compilerReady = true; revision = data.revision;
        $('revision-link').href = `https://github.com/rsntlabs/wgpu/tree/${revision}`;
        $('revision-link').title = `dev/cuda · ${revision}`;
        resolve();
      } else if (data.type === 'fatal') {
        clearTimeout(bootTimer); reject(new Error(data.error));
      } else if (data.type === 'result') {
        const request = pending.get(data.id);
        if (!request) return;
        clearTimeout(request.timer); pending.delete(data.id);
        data.error ? request.reject(new Error(data.error)) : request.resolve(data);
      }
    };
    worker.onerror = event => {
      clearTimeout(bootTimer);
      const error = new Error(event.message || 'The WebAssembly compiler stopped unexpectedly.');
      compilerReady = false;
      for (const request of pending.values()) { clearTimeout(request.timer); request.reject(error); }
      pending.clear(); reject(error);
    };
  });
  return ready;
}

function translate(sourceText, id) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id); compilerReady = false; worker.terminate();
      reject(new Error('Translation exceeded 12 seconds. Simplify the kernel and try again.'));
    }, 12000);
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ source: sourceText, id });
  });
}

function playbackState() {
  const paused = renderer?.paused;
  $('pause').ariaLabel = paused ? 'Resume animation' : 'Pause animation';
  $('pause').title = $('pause').ariaLabel;
  $('pause-icon').innerHTML = paused ? '<path d="M5 3l8 5-8 5z"/>' : '<path d="M5 3v10M11 3v10"/>';
  $('preview-state').textContent = paused ? 'PAUSED' : 'LIVE PREVIEW';
  $('live-dot').classList.toggle('active', !paused);
  if (paused) $('fps').textContent = 'PAUSED';
}

async function compile() {
  if (compiling || initializing) return;
  compiling = true;
  $('compile').disabled = true;
  const id = ++currentRequest;
  const text = source.value;
  status('Translating CUDA → Naga IR → WGSL…');
  try {
    if (!compilerReady) await startCompiler();
    const result = await translate(text, id);
    if (id !== currentRequest) return;
    output.value = result.wgsl; lines(output); $('copy').disabled = false;
    $('compile-time').textContent = `${result.ms.toFixed(1)} ms`;
    if (renderer) {
      const pipeline = await renderer.compile(result.wgsl);
      if (id !== currentRequest) return;
      await renderer.usePipeline(pipeline);
      $('stage-message').hidden = true;
      $('pause').disabled = false; $('reset').disabled = false;
      playbackState();
    }
    appliedSource = text;
    if (source.value === text) {
      status(renderer ? `Compiled successfully · ${result.ms.toFixed(1)} ms in WASM · running on WebGPU` : `Translated successfully · ${result.ms.toFixed(1)} ms in WASM · WebGPU preview unavailable in this browser`, 'success');
    } else {
      status('Source changed during compilation. Translate & run to apply your latest edits.', 'dirty');
    }
  } catch (error) {
    if (id !== currentRequest) return;
    status(error.message || String(error), 'error');
    if (!renderer?.pipeline) showStageError(renderer ? 'The shader could not be compiled. Check the error below the editors and try again.' : 'The compiler could not start. Check the error below the editors.');
  } finally {
    compiling = false;
    $('compile').disabled = false;
  }
}

async function loadPreset(name) {
  const request = ++presetRequest;
  status(`Loading ${presets[name][1]}…`);
  try {
    if (!cache.has(name)) {
      const response = await fetch(new URL(`./shaders/${presets[name][0]}`, import.meta.url));
      if (!response.ok) throw new Error(`Could not load the ${presets[name][1]} preset (${response.status}).`);
      cache.set(name, await response.text());
    }
    if (request !== presetRequest) return;
    source.value = cache.get(name); lines(source);
    $('effect-label').textContent = presets[name][1].toUpperCase();
    if (compiling) {
      // Let the current compile finish before applying the newly selected source.
      const wait = setInterval(() => { if (!compiling) { clearInterval(wait); if (request === presetRequest) compile(); } }, 50);
    } else if (compilerReady && !initializing) await compile();
  } catch (error) { status(error.message, 'error'); }
}

source.addEventListener('input', () => {
  lines(source);
  status(source.value === appliedSource ? 'Source matches the running shader.' : 'You have changes. Translate & run to bring them to life.', source.value === appliedSource ? 'success' : 'dirty');
});
for (const textarea of [source, output]) textarea.addEventListener('scroll', () => { $(textarea.id + '-lines').scrollTop = textarea.scrollTop; });
source.addEventListener('keydown', event => {
  if (event.key === 'Tab') {
    event.preventDefault(); source.setRangeText('    ', source.selectionStart, source.selectionEnd, 'end'); source.dispatchEvent(new Event('input'));
  }
});
document.addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); compile(); } });
$('compile').addEventListener('click', compile);
$('preset').addEventListener('change', () => loadPreset($('preset').value));
$('restore').addEventListener('click', () => loadPreset($('preset').value));
$('count').addEventListener('change', () => renderer?.setCount(Number($('count').value)));
$('pause').addEventListener('click', () => { renderer.togglePause(); playbackState(); });
$('reset').addEventListener('click', () => renderer.restart());
$('fullscreen').addEventListener('click', async () => {
  try { document.fullscreenElement ? await document.exitFullscreen() : await $('stage').requestFullscreen(); }
  catch { status('Fullscreen is unavailable in this browser.', 'error'); }
});
if (!$('stage').requestFullscreen) $('fullscreen').hidden = true;
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(output.value); $('copy').textContent = 'Copied ✓'; setTimeout(() => { $('copy').textContent = 'Copy ⧉'; }, 1600); }
  catch { status('Clipboard access is unavailable. Select the WGSL text to copy it.', 'error'); output.focus(); output.select(); }
});
window.addEventListener('pagehide', () => { renderer?.dispose(); worker?.terminate(); });

const gpu = ParticleRenderer.create($('particles'), message => { showStageError(message); status(message, 'error'); }, fps => { $('fps').textContent = renderer?.paused ? 'PAUSED' : `${fps} FPS`; })
  .then(value => { renderer = value; renderer.setCount(Number($('count').value)); })
  .catch(error => showStageError(error.message));
const compiler = startCompiler().catch(error => { status(error.message, 'error'); showStageError(error.message); });
await Promise.all([gpu, compiler, loadPreset($('preset').value)]);
initializing = false;
$('compile').disabled = false;
if (compilerReady) await compile();
