# Architecture

`gpt2cpp` is divided into clean layers:

- `core/`: config, logging, file helpers, device selection
- `tokenizer/`: byte-level BPE training + serialization
- `data/`: corpus ingestion, token packing, binary dataset format
- `model/`: GPT-2-style transformer stack using LibTorch
- `training/`: optimizer, scheduler, validation, checkpoints
- `inference/`: bundle loading and generation
- `app/`: terminal chat REPL
- `serving/`: local HTTP server + static frontend

The key design goal is to make each artifact explicit and easy to explain in an interview.
