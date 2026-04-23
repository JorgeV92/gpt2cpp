# Tokenizer

The tokenizer is a byte-level BPE-style tokenizer implemented in C++.

Base vocabulary:
- bytes `0..255`

Learned vocabulary:
- merge tokens created from frequent adjacent pairs

Serialization:
- `tokenizer.meta`
- `vocab.tsv`
- `merges.tsv`
- `special_tokens.tsv`

Encoding scans for special tokens first, then BPE-encodes normal text segments.
