# cudademo · CUDA → WebGPU

A static shader playground for rsnt labs. Edit a CUDA kernel, translate it to WGSL
inside a Web Worker using Naga compiled to WebAssembly, then run that exact WGSL
as a WebGPU compute shader. No translation service, CUDA toolkit, NVIDIA driver,
JavaScript transpiler, or pretranslated fallback is involved.

Three editable CUDA presets generate a spiral galaxy, orbital vortex, and wave
field. The renderer draws the compute buffer as additive particle sprites. The
page includes particle count, pause/resume, restart, fullscreen, WGSL copy, source
diagnostics, and a usable translation view when WebGPU is unavailable.

## Compiler provenance

`Cargo.toml` pins Naga from [`rsntlabs/wgpu`, `dev/cuda`](https://github.com/rsntlabs/wgpu/tree/dev/cuda)
at [`02f26c369996e6204d38069048be9051c2ea4e74`](https://github.com/rsntlabs/wgpu/commit/02f26c369996e6204d38069048be9051c2ea4e74).
`Cargo.lock` pins the full dependency graph. The small Rust wrapper calls
`naga::front::cuda::parse_str`, validates the IR, and calls Naga's WGSL backend.
The worker loads the resulting WASM module and returns generated WGSL, compilation
time, and error locations. Translation has a 64 KiB source limit and a 12-second
worker timeout. CUDA input stays in the browser.

The frontend supports a [documented CUDA subset](https://github.com/rsntlabs/wgpu/blob/02f26c369996e6204d38069048be9051c2ea4e74/docs/cuda.md).
Presets use a CUDA sine approximation for inexpensive visual animation; their
helper code is also translated by Naga.

## Local build

Install Rust (at least the upstream workspace's Rust 1.95 requirement), the WASM
target, and the matching wasm-bindgen CLI:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.128 --locked
cargo test --locked
bash scripts/build.sh
python3 -m http.server 4173 --directory dist
```

Open `http://localhost:4173`. The build produces only static files in `dist/`.
The browser needs WebGPU for the particle view, and HTTPS or localhost. Translation
still works without a WebGPU adapter. Reduced-motion preferences start the effect
paused. No npm packages are needed.

## Kernel interface

Keep this entry point and parameter order for the renderer:

```cuda
extern "C" __global__ void particles(float4 *points, const float *params)
```

- Group 0, binding 0: writable storage buffer of `float4` particles (16-byte stride).
  `x` and `y` are positions near `[-1, 1]`, `z` selects the radial color, and `w`
  controls sprite brightness/size near `[0, 1]`.
- Group 0, binding 1: read-only storage buffer of eight floats. `params[0]` is time
  in seconds and `params[1]` is particle count. Remaining entries are reserved.
- Workgroup size is `[64, 1, 1]`. Guard indices against `params[1]`.

The host allocates space for 131,072 particles and dispatches the selected count.
The generated shader goes straight to `createShaderModule` and an asynchronously
validated compute pipeline. Invalid edits leave the previous working pipeline
intact. A separate WGSL render pipeline draws the generated particle positions.

## Browser checks

With Node 24+ and Chromium installed:

```sh
node scripts/test-browser.mjs
# Check the published site using the same tests:
node scripts/test-browser.mjs https://rsntlabs.github.io/cudademo/
```

Set `CHROMIUM` to another Chromium executable when needed (Chrome for Testing
156 was verified locally). Set `CHROMIUM_FLAGS` to override the graphics launch
flags when testing a particular native driver. Checks use a temporary
browser profile and software Vulkan, so they do not require a physical GPU. They
verify actual lit pixels, a visible position change after a CUDA edit, generated
WGSL, all presets, compiler and GPU errors, playback controls, mobile sizing, and
the no-WebGPU experience. Screenshots and logs go in `.test-artifacts/`.
Local tests serve under the repository subpath to exercise GitHub Pages paths.

## GitHub Pages

The workflow runs Rust tests, builds the WASM compiler, uploads `dist/`, and
deploys to GitHub Pages on every push to `main`. Browser checks are run manually
using the command above and are intentionally excluded from CI. Pull requests
run the Rust tests and build without deploying.

For first-time setup, in the repository's **Settings → Pages → Build and
deployment**, select **GitHub Actions**. GitHub requires an eligible plan to
publish Pages from a private repository. The workflow does not change repository
visibility. After setup, rerun the workflow or push to `main`.

Default URL: `https://rsntlabs.github.io/cudademo/`.
All asset URLs are relative and also work at a custom domain's root if a custom
domain is configured in GitHub Pages.
