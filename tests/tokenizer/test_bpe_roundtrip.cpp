#include "gpt2cpp/tokenizer/bpe_tokenizer.hpp"
#include <cassert>
#include <iostream>
int main() {
    gpt2cpp::BPETokenizer tok;
    gpt2cpp::BPETrainingOptions opt;
    opt.vocab_size = 300;
    opt.special_tokens = {"<|endoftext|>"};
    tok.train_from_text("hello hello world world hello", opt);
    const std::string text = "hello<|endoftext|>world";
    const auto ids = tok.encode(text);
    assert(tok.decode(ids) == text);
    std::cout << "tokenizer roundtrip ok\n";
    return 0;
}
