#ifndef GPT2CPP_MODEL_GPT2_MODEL_HPP
#define GPT2CPP_MODEL_GPT2_MODEL_HPP
#include "gpt2cpp/core/config.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include <torch/torch.h>
namespace gpt2cpp {
using GPT2Config = ModelSection;
struct ModelOutput { torch::Tensor logits; torch::Tensor loss; bool has_loss{false}; };
struct AttentionImpl : torch::nn::Module {
    explicit AttentionImpl(const GPT2Config& c);
    torch::Tensor forward(const torch::Tensor& x);
    GPT2Config cfg;
    int head_dim{0};
    torch::nn::Linear qkv{nullptr}, proj{nullptr};
    torch::nn::Dropout attn_drop{nullptr}, resid_drop{nullptr};
};
TORCH_MODULE(Attention);

struct MLPImpl : torch::nn::Module {
    explicit MLPImpl(const GPT2Config& c);
    torch::Tensor forward(const torch::Tensor& x);
    torch::nn::Linear fc{nullptr}, proj{nullptr};
    torch::nn::Dropout drop{nullptr};
};
TORCH_MODULE(MLP);

struct BlockImpl : torch::nn::Module {
    explicit BlockImpl(const GPT2Config& c);
    torch::Tensor forward(const torch::Tensor& x);
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    Attention attn{nullptr};
    MLP mlp{nullptr};
};
TORCH_MODULE(Block);

struct GPT2ModelImpl : torch::nn::Module {
    explicit GPT2ModelImpl(const GPT2Config& c);
    ModelOutput forward(const torch::Tensor& input_ids, const c10::optional<torch::Tensor>& targets = c10::nullopt);
    GPT2Config cfg;
    torch::nn::Embedding wte{nullptr}, wpe{nullptr};
    torch::nn::Dropout drop{nullptr};
    torch::nn::ModuleList blocks;
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};
};
TORCH_MODULE(GPT2Model);
}  // namespace gpt2cpp
#endif
#endif
