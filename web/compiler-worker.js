import init, { translate, compiler_revision } from './pkg/cuda_browser.js';

try {
  await init();
  self.postMessage({ type: 'ready', revision: compiler_revision() });
} catch (error) {
  self.postMessage({ type: 'fatal', error: `Could not load the WebAssembly compiler: ${error.message || error}` });
}

self.onmessage = ({ data }) => {
  const start = performance.now();
  try {
    const wgsl = translate(data.source);
    self.postMessage({ type: 'result', id: data.id, wgsl, ms: performance.now() - start });
  } catch (error) {
    self.postMessage({ type: 'result', id: data.id, error: String(error) });
  }
};
