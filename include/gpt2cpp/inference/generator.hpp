#ifndef GPT2CPP_INFERENCE_GENERATOR_HPP
#define GPT2CPP_INFERENCE_GENERATOR_HPP
#include "gpt2cpp/core/config.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/model/gpt2_model.hpp"
#include "gpt2cpp/tokenizer/bpe_tokenizer.hpp"
#include <functional>
namespace gpt2cpp {
struct GenerationOptions {
    double temperature{0.8};
    int top_k{40};
    double top_p{0.95};
    int max_new_tokens{128};
    double repetition_penalty{1.0};
    std::vector<std::string> stop_sequences{};
};
class Generator {
public:
    explicit Generator(const std::filesystem::path& model_bundle);
    std::string generate(const std::string& prompt, const GenerationOptions& options);
    std::string generate_streaming(const std::string& prompt, const GenerationOptions& options, const std::function<void(const std::string&)>& on_chunk);
    const AppConfig& config() const;
private:
    torch::Tensor sample_next(torch::Tensor logits, std::vector<int32_t>& generated, const GenerationOptions& options) const;
    bool should_stop(const std::string& text, const GenerationOptions& options) const;
    AppConfig config_;
    BPETokenizer tokenizer_;
    GPT2Model model_{nullptr};
    torch::Device device_;
};
}  // namespace gpt2cpp
#endif
#endif
