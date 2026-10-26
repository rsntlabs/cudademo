#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --locked --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg --out-name cuda_browser target/wasm32-unknown-unknown/release/cuda_browser.wasm
rm -rf dist
mkdir -p dist
cp -R web/. dist/
touch dist/.nojekyll
