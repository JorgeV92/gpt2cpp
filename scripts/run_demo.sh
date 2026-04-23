#!/usr/bin/env bash
set -euo pipefail
./build/gpt2cpp tokenizer train   --input examples/tiny_shakespeare/sample.txt   --output artifacts/tokenizer   --vocab-size 512   --special "<|system|>,<|user|>,<|assistant|>,<|endoftext|>"
./build/gpt2cpp data prepare   --tokenizer artifacts/tokenizer   --input examples/tiny_shakespeare   --output artifacts/dataset   --val-ratio 0.1
echo "Tokenizer and dataset are ready."
echo "Next: ./build/gpt2cpp train --config configs/tiny.toml"
