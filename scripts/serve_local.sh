#!/usr/bin/env bash
set -euo pipefail
MODEL_BUNDLE="${1:-runs/latest/model_bundle}"
PORT="${PORT:-8080}"
./build/gpt2cpp serve --model "${MODEL_BUNDLE}" --port "${PORT}" --web-root web
