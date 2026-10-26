// Dependency-free end-to-end checks against Chromium's WebGPU implementation.
// Usage: node scripts/test-browser.mjs [https://deployed.example/path/]
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { resolve, extname, join } from 'node:path';
import { tmpdir } from 'node:os';
import assert from 'node:assert/strict';

const artifacts = resolve('.test-artifacts');
await mkdir(artifacts, { recursive: true });
const profile = await mkdtemp(join(tmpdir(), 'cuda-webgpu-test-'));
const prefix = '/cudademo.rsntlabs.com/';
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.cu': 'text/plain' };
const server = createServer(async (req, res) => {
  try {
    const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    if (!pathname.startsWith(prefix)) { res.writeHead(404).end(); return; }
    const relative = pathname.slice(prefix.length) || 'index.html';
    const file = resolve('dist', relative);
    if (!file.startsWith(resolve('dist') + '/')) throw new Error('Invalid path');
    const data = await readFile(file);
    res.writeHead(200, { 'Content-Type': mime[extname(file)] || 'application/octet-stream' }).end(data);
  } catch { res.writeHead(404).end(); }
});
if (!process.argv[2]) await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = process.argv[2] || `http://127.0.0.1:${server.address().port}${prefix}`;
const graphicsFlags = process.env.CHROMIUM_FLAGS?.split(/\s+/).filter(Boolean) || ['--enable-features=Vulkan', '--use-angle=vulkan', '--use-vulkan=swiftshader', '--disable-vulkan-surface', '--enable-unsafe-webgpu'];
const browser = spawn(process.env.CHROMIUM || 'chromium', ['--headless', '--no-sandbox', '--disable-dev-shm-usage', ...graphicsFlags, '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: ['ignore', 'ignore', 'pipe'] });
let browserLog = '';
browser.stderr.on('data', chunk => { browserLog += chunk; });
let socket;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = '';
    const timer = setTimeout(() => reject(new Error('Chromium did not start')), 20000);
    browser.stderr.on('data', chunk => { stderr += chunk; const match = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/); if (match) { clearTimeout(timer); resolve(match[1]); } });
    browser.on('error', reject);
    browser.on('exit', code => { clearTimeout(timer); reject(new Error(`Chromium exited ${code}: ${stderr}`)); });
  });
  socket = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  const requests = new Map();
  const exceptions = [];
  let seq = 0;
  socket.onmessage = ({ data }) => {
    const result = JSON.parse(data);
    if (result.method === 'Runtime.exceptionThrown') exceptions.push(result.params.exceptionDetails);
    if (result.id) {
      const pending = requests.get(result.id); if (!pending) return;
      clearTimeout(pending.timer); requests.delete(result.id);
      result.error ? pending.reject(new Error(JSON.stringify(result.error))) : pending.resolve(result.result);
    }
  };
  const call = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++seq;
    const timer = setTimeout(() => { requests.delete(id); reject(new Error(`CDP timed out: ${method}`)); }, 45000);
    requests.set(id, { resolve, reject, timer }); socket.send(JSON.stringify({ id, method, params, sessionId }));
  });
  const { targetId } = await call('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await call('Target.attachToTarget', { targetId, flatten: true });
  const tab = (method, params) => call(method, params, sessionId);
  const evaluate = async expression => {
    const result = await tab('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true, userGesture: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  const wait = async expression => {
    const start = Date.now();
    while (Date.now() - start < 30000) {
      if (await evaluate(expression)) return;
      await new Promise(resolve => setTimeout(resolve, 150));
    }
    throw new Error(`Timed out: ${expression}\n${await evaluate('document.body.innerText')}`);
  };
  const screenshot = async name => {
    const result = await tab('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    await writeFile(join(artifacts, name), Buffer.from(result.data, 'base64'));
    return result.data;
  };
  const setSource = text => evaluate(`document.getElementById('cuda').value=${JSON.stringify(text)};document.getElementById('cuda').dispatchEvent(new Event('input'));document.getElementById('compile').click()`);
  await tab('Page.enable'); await tab('Runtime.enable');
  await tab('Page.addScriptToEvaluateOnNewDocument', { source: `
    if(navigator.gpu){
      const getAdapter=navigator.gpu.requestAdapter.bind(navigator.gpu);
      navigator.gpu.requestAdapter=async(...args)=>{
        const adapter=await getAdapter(...args); if(!adapter)return adapter;
        const getDevice=adapter.requestDevice.bind(adapter);
        adapter.requestDevice=async(...args)=>{
          const device=await getDevice(...args);globalThis.testDevice=device;
          const create=device.createBuffer.bind(device);
          device.createBuffer=desc=>{const buffer=create(desc);if(desc.label==='CUDA float4 particle buffer')globalThis.testPoints=buffer;return buffer;};
          return device;
        };return adapter;
      };
    }
  ` });
  await tab('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1700, deviceScaleFactor: 1, mobile: false });
  await tab('Page.navigate', { url: base });
  await call('Target.activateTarget', { targetId });
  await tab('Page.bringToFront');
  await wait(`document.getElementById('status-bar')?.dataset.state === 'success' && document.getElementById('preview-state').textContent === 'LIVE PREVIEW'`);
  console.log('PASS: WASM translated CUDA and the WebGPU pipeline is running');
  const pointSample = await evaluate(`(async()=>{const d=testDevice;const b=d.createBuffer({size:256,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=d.createCommandEncoder();e.copyBufferToBuffer(testPoints,0,b,0,256);d.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);const values=Array.from(new Float32Array(b.getMappedRange()));b.unmap();b.destroy();return values})()`);
  assert.ok(pointSample.every(Number.isFinite) && pointSample.some(v=>v!==0), 'Compute output must contain finite particle data');
  console.log('PASS: actual GPU compute buffer contains valid particle positions');
  await wait(`parseInt(document.getElementById('fps').textContent, 10) > 0`);
  assert.match(await evaluate(`document.getElementById('wgsl').value`), /@compute[\s\S]*@workgroup_size\(64/);
  assert.equal(await evaluate(`document.querySelectorAll('textarea').length`), 2);
  assert.match(await evaluate(`document.getElementById('revision-link').href`), /02f26c369996e6204d38069048be9051c2ea4e74/);
  await screenshot('desktop-live.png');
  await evaluate(`document.getElementById('pause').click()`);
  await wait(`document.getElementById('preview-state').textContent === 'PAUSED'`);
  const original = await evaluate(`document.getElementById('cuda').value`);
  const originalWgsl = await evaluate(`document.getElementById('wgsl').value`);
  await screenshot('desktop.png');
  // Read back the actual canvas, proving that the compute output produces lit pixels.
  const pixels = await evaluate(`(async()=>{document.getElementById('reset').click();const canvas=document.getElementById('particles');const bitmap=await createImageBitmap(canvas);const probe=document.createElement('canvas');probe.width=canvas.width;probe.height=canvas.height;const ctx=probe.getContext('2d');ctx.drawImage(bitmap,0,0);const p=ctx.getImageData(0,0,probe.width,probe.height).data;let lit=0;for(let i=0;i<p.length;i+=4)if(p[i]>40 || p[i+1]>40 || p[i+2]>40)lit++;bitmap.close();return lit})()`);
  assert.ok(pixels > 2000, `Expected thousands of visible particle pixels, found ${pixels}`);
  console.log(`PASS: particle image has ${pixels} lit pixels`);
  await setSource(original.replace('points[i].x = x * 0.86f;', 'points[i].x = 0.0f;'));
  await wait(`document.getElementById('status-bar').dataset.state === 'success' && !document.getElementById('compile').disabled`);
  const editedWgsl = await evaluate(`document.getElementById('wgsl').value`);
  assert.notEqual(editedWgsl, originalWgsl);
  const bounds = await evaluate(`(async()=>{const c=document.getElementById('particles');const b=await createImageBitmap(c);const p=document.createElement('canvas');p.width=c.width;p.height=c.height;const x=p.getContext('2d');x.drawImage(b,0,0);const d=x.getImageData(0,0,p.width,p.height).data;let lit=0;for(let i=0;i<d.length;i+=4)if(d[i]>40 || d[i+1]>40 || d[i+2]>40)lit++;b.close();return lit})()`);
  assert.ok(bounds < pixels * 0.5, 'Changing CUDA must visibly collapse the galaxy into a line');
  console.log('PASS: editing CUDA changes generated WGSL and the rendered particle positions');
  await setSource('__global__ void broken( {');
  await wait(`document.getElementById('status-bar').dataset.state === 'error'`);
  assert.match(await evaluate(`document.getElementById('status').textContent`), /line 1/);
  assert.equal(await evaluate(`document.getElementById('stage-message').hidden`), true);
  console.log('PASS: malformed CUDA reports its source location and preserves the previous preview');
  await setSource(original.replace('void particles(', 'void renamed('));
  await wait(`document.getElementById('status-bar').dataset.state === 'error' && !document.getElementById('compile').disabled`);
  assert.match(await evaluate(`document.getElementById('status').textContent`), /WebGPU/);
  console.log('PASS: incompatible GPU entry points are rejected without replacing the running pipeline');
  for (const preset of ['vortex', 'wave', 'galaxy']) {
    await evaluate(`document.getElementById('preset').value=${JSON.stringify(preset)};document.getElementById('preset').dispatchEvent(new Event('change'))`);
    await wait(`document.getElementById('status-bar').dataset.state === 'success' && !document.getElementById('compile').disabled`);
  }
  console.log('PASS: all three CUDA presets translate and run');
  await evaluate(`document.getElementById('count').value='131072';document.getElementById('count').dispatchEvent(new Event('change'));document.getElementById('reset').click()`);
  await evaluate(`document.getElementById('pause').click()`);
  await wait(`document.getElementById('preview-state').textContent === 'LIVE PREVIEW'`);
  await evaluate(`document.getElementById('pause').click()`);
  console.log('PASS: particle count, restart, pause, and resume controls');
  await tab('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await new Promise(resolve => setTimeout(resolve, 300));
  assert.ok(await evaluate(`document.documentElement.scrollWidth <= window.innerWidth`), 'Mobile page must not overflow horizontally');
  await screenshot('mobile.png');
  assert.deepEqual(exceptions, [], 'No uncaught browser exceptions');
  console.log('PASS: mobile layout and no uncaught browser exceptions');
  // Prove translation still works and an actionable message appears without WebGPU.
  await tab('Page.addScriptToEvaluateOnNewDocument', { source: `Object.defineProperty(navigator, 'gpu', { value: undefined })` });
  await tab('Page.reload');
  await wait(`document.getElementById('status-bar')?.dataset.state === 'success'`);
  assert.match(await evaluate(`document.getElementById('stage-message').textContent`), /WebGPU is unavailable/);
  assert.match(await evaluate(`document.getElementById('wgsl').value`), /@compute/);
  console.log('PASS: browsers without WebGPU can translate and see a clear compatibility message');
  console.log(`Screenshots: ${artifacts}`);
} finally {
  await writeFile(join(artifacts, 'chromium.log'), browserLog);
  socket?.close(); browser.kill(); server.close();
  await new Promise(resolve => { if (browser.exitCode !== null) resolve(); else { browser.once('exit', resolve); setTimeout(resolve, 3000); } });
  await rm(profile, { recursive: true, force: true, maxRetries: 3, retryDelay: 200 }).catch(() => {});
}
