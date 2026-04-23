#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: scripts/setup_macos.sh is intended for macOS." >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/tiny.toml}"
PROMPT="${PROMPT:-To be}"
ENABLE_TRAIN="${ENABLE_TRAIN:-1}"
ENABLE_GENERATE="${ENABLE_GENERATE:-1}"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install torch

CMAKE_PREFIX_PATH="$("${VENV_DIR}/bin/python" -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake -S . -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DGPT2CPP_ENABLE_TORCH=ON \
  -DGPT2CPP_ENABLE_CUDA=OFF \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"

cmake --build "${BUILD_DIR}" -j

"./${BUILD_DIR}/gpt2cpp" tokenizer train \
  --input examples/tiny_shakespeare/sample.txt \
  --output artifacts/tokenizer \
  --vocab-size 512 \
  --special "<|system|>,<|user|>,<|assistant|>,<|endoftext|>"

"./${BUILD_DIR}/gpt2cpp" data prepare \
  --tokenizer artifacts/tokenizer \
  --input examples/tiny_shakespeare \
  --output artifacts/dataset \
  --val-ratio 0.1

if [[ "${ENABLE_TRAIN}" == "1" ]]; then
  "./${BUILD_DIR}/gpt2cpp" train --config "${TRAIN_CONFIG}"
fi

LATEST_MODEL="$(find runs -maxdepth 2 -type d -name model_bundle | sort | tail -n 1)"

if [[ -z "${LATEST_MODEL}" ]]; then
  echo "error: no model_bundle found under runs/ after setup." >&2
  exit 1
fi

if [[ "${ENABLE_GENERATE}" == "1" ]]; then
  "./${BUILD_DIR}/gpt2cpp" generate --model "${LATEST_MODEL}" --prompt "${PROMPT}"
fi

echo
echo "Setup complete."
echo "Latest model: ${LATEST_MODEL}"
echo "Serve locally with:"
echo "./${BUILD_DIR}/gpt2cpp serve --model \"${LATEST_MODEL}\" --port 8080 --web-root web"
