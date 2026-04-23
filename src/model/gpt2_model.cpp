#include "gpt2cpp/model/gpt2_model.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/core/error.hpp"
#include <cmath>
namespace gpt2cpp {

AttentionImpl::AttentionImpl(const GPT2Config& c)
    : cfg(c),
      head_dim(c.n_embd / c.n_head),
      qkv(register_module("qkv", torch::nn::Linear(c.n_embd, 3 * c.n_embd))),
      proj(register_module("proj", torch::nn::Linear(c.n_embd, c.n_embd))),
      attn_drop(register_module("attn_drop", torch::nn::Dropout(c.dropout))),
      resid_drop(register_module("resid_drop", torch::nn::Dropout(c.dropout))) {
    require(c.n_embd % c.n_head == 0, "n_embd must be divisible by n_head.");
}

torch::Tensor AttentionImpl::forward(const torch::Tensor& x) {
    const auto B = x.size(0), T = x.size(1), C = x.size(2);
    auto parts = qkv->forward(x).split(C, -1);
    auto q = parts[0].view({B, T, cfg.n_head, head_dim}).transpose(1, 2);
    auto k = parts[1].view({B, T, cfg.n_head, head_dim}).transpose(1, 2);
    auto v = parts[2].view({B, T, cfg.n_head, head_dim}).transpose(1, 2);
    auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim));
    auto mask = torch::tril(torch::ones({T, T}, x.options())).view({1, 1, T, T});
    att = att.masked_fill(mask == 0, -1e10);
    att = torch::softmax(att, -1);
    att = attn_drop->forward(att);
    auto y = torch::matmul(att, v).transpose(1, 2).contiguous().view({B, T, C});
    return resid_drop->forward(proj->forward(y));
}

MLPImpl::MLPImpl(const GPT2Config& c)
    : fc(register_module("fc", torch::nn::Linear(c.n_embd, 4 * c.n_embd))),
      proj(register_module("proj", torch::nn::Linear(4 * c.n_embd, c.n_embd))),
      drop(register_module("drop", torch::nn::Dropout(c.dropout))) {}

torch::Tensor MLPImpl::forward(const torch::Tensor& x) {
    return drop->forward(proj->forward(torch::gelu(fc->forward(x))));
}

BlockImpl::BlockImpl(const GPT2Config& c)
    : ln1(register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.n_embd})))),
      ln2(register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.n_embd})))),
      attn(register_module("attn", Attention(c))),
      mlp(register_module("mlp", MLP(c))) {}

torch::Tensor BlockImpl::forward(const torch::Tensor& x) {
    auto h = x + attn->forward(ln1->forward(x));
    h = h + mlp->forward(ln2->forward(h));
    return h;
}

GPT2ModelImpl::GPT2ModelImpl(const GPT2Config& c)
    : cfg(c),
      wte(register_module("wte", torch::nn::Embedding(c.vocab_size, c.n_embd))),
      wpe(register_module("wpe", torch::nn::Embedding(c.max_position_embeddings, c.n_embd))),
      drop(register_module("drop", torch::nn::Dropout(c.dropout))),
      blocks(register_module("blocks", torch::nn::ModuleList())),
      ln_f(register_module("ln_f", torch::nn::LayerNorm(torch::nn::LayerNormOptions({c.n_embd})))),
      lm_head(register_module("lm_head", torch::nn::Linear(torch::nn::LinearOptions(c.n_embd, c.vocab_size).bias(false)))) {
    for (int i = 0; i < c.n_layer; ++i) blocks->push_back(Block(c));
    if (c.tie_weights) lm_head->weight = wte->weight;
}

ModelOutput GPT2ModelImpl::forward(const torch::Tensor& input_ids, const c10::optional<torch::Tensor>& targets) {
    const auto B = input_ids.size(0), T = input_ids.size(1);
    require(T <= cfg.max_position_embeddings, "Input exceeds max_position_embeddings.");
    auto pos = torch::arange(0, T, input_ids.options()).unsqueeze(0);
    auto x = drop->forward(wte->forward(input_ids) + wpe->forward(pos));
    for (const auto& block : *blocks) x = block->as<Block>()->forward(x);
    x = ln_f->forward(x);
    auto logits = lm_head->forward(x);
    ModelOutput out; out.logits = logits;
    if (targets.has_value()) {
        out.loss = torch::nn::functional::cross_entropy(logits.view({B * T, cfg.vocab_size}), targets.value().view({B * T}));
        out.has_loss = true;
    }
    return out;
}

}  // namespace gpt2cpp
#endif
