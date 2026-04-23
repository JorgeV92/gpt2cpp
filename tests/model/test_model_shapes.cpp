#include "gpt2cpp/model/gpt2_model.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include <cassert>
#include <iostream>
int main() {
    gpt2cpp::GPT2Config cfg;
    cfg.vocab_size = 128; cfg.max_position_embeddings = 32; cfg.n_embd = 64; cfg.n_head = 4; cfg.n_layer = 2;
    gpt2cpp::GPT2Model model(cfg);
    auto x = torch::randint(0, cfg.vocab_size, {2,16}, torch::TensorOptions().dtype(torch::kInt64));
    auto y = torch::randint(0, cfg.vocab_size, {2,16}, torch::TensorOptions().dtype(torch::kInt64));
    auto out = model->forward(x, y);
    assert(out.logits.size(0) == 2 && out.logits.size(1) == 16 && out.logits.size(2) == cfg.vocab_size);
    assert(out.has_loss);
    std::cout << "model shapes ok\n";
    return 0;
}
#else
int main() { return 0; }
#endif
