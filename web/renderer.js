const MAX_PARTICLES = 131072;

const DRAW_SHADER = /* wgsl */ `
struct View { resolution: vec2f, time: f32, count: f32 };
@group(0) @binding(0) var<storage, read> points: array<vec4f>;
@group(0) @binding(1) var<uniform> view: View;
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f, @location(1) color: vec3f, @location(2) intensity: f32 };
@vertex fn vertex_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> Vertex {
    let corners = array<vec2f, 6>(vec2f(-1,-1),vec2f(1,-1),vec2f(-1,1),vec2f(-1,1),vec2f(1,-1),vec2f(1,1));
    let corner = corners[v];
    let p = points[i];
    let fit = min(view.resolution.x, view.resolution.y * 1.6);
    let rotation = -0.23;
    let rotated = vec2f(p.x * cos(rotation) - p.y * sin(rotation), p.x * sin(rotation) + p.y * cos(rotation));
    let center = rotated * vec2f(1.0, 0.68) * fit / view.resolution;
    let size = (2.0 + p.w * 3.8) * clamp(view.resolution.y / 420.0, 0.8, 2.0);
    var out: Vertex;
    out.position = vec4f(center + corner * size / view.resolution, 0.0, 1.0);
    out.uv = corner;
    let radius = clamp(p.z, 0.0, 1.0);
    let inner = mix(vec3f(1.0,0.88,0.61), vec3f(0.92,0.52,0.22), smoothstep(0.0,0.48,radius));
    out.color = mix(inner,vec3f(0.28,0.72,0.73),smoothstep(0.35,0.95,radius));
    out.intensity = (0.24 + p.w * 0.52) * sqrt(32768.0 / view.count);
    return out;
}
@fragment fn fragment_main(in: Vertex) -> @location(0) vec4f {
    let d = dot(in.uv, in.uv);
    let glow = exp(-d * 4.5) * max(1.0 - d, 0.0);
    return vec4f(in.color * glow * in.intensity, 1.0);
}`;

export class ParticleRenderer {
  static async create(canvas, onError, onStats) {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser. Open this page in a browser with WebGPU enabled, using HTTPS or localhost.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('A WebGPU device could not be found. Check that hardware acceleration is enabled in your browser.');
    const device = await adapter.requestDevice();
    const renderer = new ParticleRenderer(canvas, device, onError, onStats);
    try { await renderer.initialize(); }
    catch (error) { renderer.dispose(); throw error; }
    return renderer;
  }

  constructor(canvas, device, onError, onStats) {
    Object.assign(this, { canvas, device, onError, onStats, count: 32768, time: 9, paused: matchMedia('(prefers-reduced-motion: reduce)').matches, pipeline: null, disposed: false });
    this.lastFrame = 0;
    this.frames = 0;
    this.statsStart = 0;
    device.addEventListener('uncapturederror', event => { event.preventDefault(); this.fail(event.error.message); });
    device.lost.then(info => { if (!this.disposed) this.fail(`The graphics device was lost: ${info.message || info.reason}. Reload the page to reconnect.`); });
  }

  async initialize() {
    const { device } = this;
    this.context = this.canvas.getContext('webgpu');
    if (!this.context) throw new Error('This browser could not create a WebGPU canvas.');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.points = device.createBuffer({ label: 'CUDA float4 particle buffer', size: MAX_PARTICLES * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.params = device.createBuffer({ label: 'CUDA frame parameters', size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.view = device.createBuffer({ label: 'Particle camera', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.computeLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    ] });
    this.computeGroup = device.createBindGroup({ layout: this.computeLayout, entries: [
      { binding: 0, resource: { buffer: this.points } }, { binding: 1, resource: { buffer: this.params } },
    ] });
    const drawModule = device.createShaderModule({ label: 'Particle sprites', code: DRAW_SHADER });
    this.drawPipeline = await device.createRenderPipelineAsync({
      label: 'Additive particle glow', layout: 'auto',
      vertex: { module: drawModule, entryPoint: 'vertex_main' },
      fragment: { module: drawModule, entryPoint: 'fragment_main', targets: [{ format: this.format, blend: {
        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
      } }] }, primitive: { topology: 'triangle-list' },
    });
    this.drawGroup = device.createBindGroup({ layout: this.drawPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.points } }, { binding: 1, resource: { buffer: this.view } },
    ] });
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(this.canvas);
    this.tick = this.tick.bind(this);
    this.animation = requestAnimationFrame(this.tick);
  }

  async compile(wgsl) {
    const { device } = this;
    device.pushErrorScope('validation');
    let result;
    let failure;
    try {
      const module = device.createShaderModule({ label: 'Translated CUDA kernel', code: wgsl });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => `WGSL · line ${m.lineNum}: ${m.message}`).join('\n'));
      result = await device.createComputePipelineAsync({ label: 'CUDA → WGSL particles', layout: device.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] }), compute: { module, entryPoint: 'particles' } });
    } catch (error) { failure = error; }
    const validation = await device.popErrorScope();
    if (failure || validation) throw new Error(`WebGPU: ${(failure || validation).message}\nKeep the particles(float4 *points, const float *params) interface and 64-thread workgroup.`);
    return result;
  }

  async usePipeline(pipeline) {
    if (this.framePending) await this.framePending;
    this.pipeline = pipeline;
    this.draw();
    await this.framePending;
  }
  setCount(count) { this.count = Math.max(64, Math.min(MAX_PARTICLES, count)); this.draw(); }
  restart() { this.time = 0; this.draw(); }
  togglePause() { this.paused = !this.paused; this.lastFrame = 0; return this.paused; }
  fail(message) { this.pipeline = null; this.onError(message); }

  tick(now) {
    if (this.disposed) return;
    if (document.visibilityState === 'visible') {
      if (!this.paused && this.pipeline) {
        if (this.lastFrame) this.time += Math.min((now - this.lastFrame) / 1000, 0.05);
        if (this.draw()) this.frames++;
      }
      if (now - this.statsStart >= 1000) {
        this.onStats(this.paused ? 0 : Math.round(this.frames * 1000 / (now - this.statsStart)));
        this.frames = 0; this.statsStart = now;
      }
      this.lastFrame = now;
    } else { this.lastFrame = 0; this.statsStart = now; this.frames = 0; }
    this.animation = requestAnimationFrame(this.tick);
  }

  draw() {
    if (!this.pipeline || this.disposed) return;
    if (this.framePending) { this.needsDraw = true; return; }
    this.needsDraw = false;
    const { canvas, device } = this;
    const scale = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.min(device.limits.maxTextureDimension2D, Math.round(canvas.clientWidth * scale)));
    const height = Math.max(1, Math.min(device.limits.maxTextureDimension2D, Math.round(canvas.clientHeight * scale)));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    device.queue.writeBuffer(this.params, 0, new Float32Array([this.time, this.count, 0, 0, 0, 0, 0, 0]));
    device.queue.writeBuffer(this.view, 0, new Float32Array([width, height, this.time, this.count]));
    const encoder = device.createCommandEncoder();
    const compute = encoder.beginComputePass();
    compute.setPipeline(this.pipeline); compute.setBindGroup(0, this.computeGroup); compute.dispatchWorkgroups(Math.ceil(this.count / 64)); compute.end();
    const render = encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), clearValue: { r: 0.024, g: 0.041, b: 0.043, a: 1 }, loadOp: 'clear', storeOp: 'store' }] });
    render.setPipeline(this.drawPipeline); render.setBindGroup(0, this.drawGroup); render.draw(6, this.count); render.end();
    device.queue.submit([encoder.finish()]);
    this.framePending = device.queue.onSubmittedWorkDone().then(() => {
      this.framePending = null;
      if (this.needsDraw) this.draw();
    }).catch(error => { this.framePending = null; this.fail(error.message); });
    return true;
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.animation);
    this.observer?.disconnect();
    this.device.destroy();
  }
}
