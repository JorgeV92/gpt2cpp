# gpt2cpp

`gpt2cpp` 

- tokenizer training and encode/decode,
- dataset packing into binary corpora,
- GPT-2-style decoder-only transformer modules,
- training/evaluation/generation workflows,
- terminal chat,
- a minimal local web demo,
- CPU-first layout with optional LibTorch CUDA acceleration,
- production-minded repo structure, artifacts, logging, and tests.

> Honest status: the tokenizer/data/CLI/serving pieces are fully owned in C++. The model/training/inference path is built around **LibTorch** for a realistic CPU/CUDA path. Some advanced performance features such as a full KV cache, fused kernels, and AMP scaler integration are explicitly marked as extension points rather than faked.

## Build

### Minimal tokenizer/data/CLI build

```bash
cmake -S . -B build -DGPT2CPP_ENABLE_TORCH=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### CPU LibTorch build

```bash
cmake -S . -B build   -DCMAKE_PREFIX_PATH=/path/to/libtorch   -DGPT2CPP_ENABLE_TORCH=ON   -DGPT2CPP_ENABLE_CUDA=OFF   -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### macOS fresh-clone CPU setup

This path uses a local Python virtual environment to provide Torch and its CMake package, then builds the full training/inference stack on CPU.

Shortcut:

```bash
./scripts/setup_macos.sh
```

Equivalent manual steps:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torch

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPT2CPP_ENABLE_TORCH=ON \
  -DGPT2CPP_ENABLE_CUDA=OFF \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake --build build -j

./scripts/run_demo.sh
./build/gpt2cpp train --config configs/tiny.toml

LATEST_MODEL="$(ls -td runs/*/model_bundle | head -n1)"
./build/gpt2cpp generate --model "$LATEST_MODEL" --prompt "To be"
```

Optional local web demo:

```bash
LATEST_MODEL="$(ls -td runs/*/model_bundle | head -n1)"
./build/gpt2cpp serve --model "$LATEST_MODEL" --port 8080 --web-root web
```

### CUDA LibTorch build

```bash
cmake -S . -B build   -DCMAKE_PREFIX_PATH=/path/to/libtorch   -DGPT2CPP_ENABLE_TORCH=ON   -DGPT2CPP_ENABLE_CUDA=ON   -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Quickstart

Train a tokenizer:

```bash
./build/gpt2cpp tokenizer train   --input examples/tiny_shakespeare/sample.txt   --output artifacts/tokenizer   --vocab-size 512   --special "<|system|>,<|user|>,<|assistant|>,<|endoftext|>"
```

Prepare packed data:

```bash
./build/gpt2cpp data prepare   --tokenizer artifacts/tokenizer   --input examples/tiny_shakespeare   --output artifacts/dataset   --val-ratio 0.1
```

Train:

```bash
./build/gpt2cpp train --config configs/tiny.toml
```

Generate:

```bash
./build/gpt2cpp generate   --model runs/.../model_bundle   --prompt "Once upon a time"   --temperature 0.8   --top-k 40
```

Chat:

```bash
./build/gpt2cpp chat --model runs/.../model_bundle
```

Serve:

```bash
./build/gpt2cpp serve --model runs/.../model_bundle --port 8080 --web-root web
```

Then open:

```text
http://127.0.0.1:8080/
```

## Architecture

```text
raw text / jsonl
   │
   ▼
byte-level BPE tokenizer
   │
   ├── vocab.tsv
   ├── merges.tsv
   └── special_tokens.tsv
   │
   ▼
packed dataset writer
   │
   ├── train.bin
   ├── val.bin
   └── dataset.meta
   │
   ▼
GPT-2-style transformer (LibTorch)
   │
   ├── checkpoints
   ├── metrics
   ├── samples
   └── model_bundle/
   │
   ├── generate
   ├── terminal chat
   └── local web server + UI
```

## Artifact bundle

```text
model_bundle/
  model.pt
  config.snapshot.toml
  tokenizer/
    tokenizer.meta
    vocab.tsv
    merges.tsv
    special_tokens.tsv
  manifest.txt
```

## Included tests

- tokenizer round-trip
- packed dataset integrity
- model shape smoke test

## Phased roadmap

### Phase 1
- tokenizer + packing + CPU-only tooling
- tiny generation path
- terminal UX

### Phase 2
- training + evaluation + checkpoints
- model bundle export
- local web demo

### Phase 3
- CUDA polish
- mixed precision integration
- cleaner inference hot path

### Phase 4
- KV cache
- SSE/WebSocket streaming
- LoRA
- quantization
- ONNX/TensorRT export

## Future extensions

- LoRA finetuning
- quantization
- ONNX/TensorRT export
- batched serving
- SSE token streaming
- retrieval augmentation
- conversation persistence
- function/tool calling experiments
