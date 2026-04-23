#include "gpt2cpp/inference/generator.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/core/device.hpp"
namespace gpt2cpp {

Generator::Generator(const std::filesystem::path& model_bundle)
    : config_(load_config(model_bundle / "config.snapshot.toml")),
      tokenizer_(BPETokenizer::load(model_bundle / "tokenizer")),
      device_(to_torch_device(parse_device_kind(config_.run.device))) {
    config_.model.vocab_size = tokenizer_.vocab_size();
    model_ = GPT2Model(config_.model);
    torch::load(model_, (model_bundle / "model.pt").string());
    model_->to(device_);
    model_->eval();
}

torch::Tensor Generator::sample_next(torch::Tensor logits, std::vector<int32_t>& generated, const GenerationOptions& options) const {
    logits = logits / std::max(1e-6, options.temperature);
    if (options.repetition_penalty > 1.0) {
        for (int32_t id : generated) logits[0][id] = logits[0][id] / options.repetition_penalty;
    }
    if (options.top_k > 0 && options.top_k < logits.size(1)) {
        auto topk = std::get<0>(torch::topk(logits, options.top_k, -1));
        const auto kth = topk[0][-1].item<double>();
        logits = torch::where(logits < kth, torch::full_like(logits, -1e10), logits);
    }
    auto probs = torch::softmax(logits, -1);
    if (options.temperature <= 1e-6) return std::get<1>(torch::max(probs, -1, true));
    return torch::multinomial(probs, 1);
}

bool Generator::should_stop(const std::string& text, const GenerationOptions& options) const {
    for (const auto& stop : options.stop_sequences)
        if (!stop.empty() && text.size() >= stop.size() && text.compare(text.size() - stop.size(), stop.size(), stop) == 0) return true;
    return false;
}

std::string Generator::generate(const std::string& prompt, const GenerationOptions& options) {
    return generate_streaming(prompt, options, [](const std::string&) {});
}

std::string Generator::generate_streaming(const std::string& prompt, const GenerationOptions& options, const std::function<void(const std::string&)>& on_chunk) {
    torch::NoGradGuard ng;
    std::vector<int32_t> generated = tokenizer_.encode(prompt);
    std::string emitted;
    for (int i = 0; i < options.max_new_tokens; ++i) {
        auto input = torch::tensor(generated, torch::TensorOptions().dtype(torch::kInt64)).unsqueeze(0).to(device_);
        auto out = model_->forward(input, c10::nullopt);
        auto logits = out.logits.index({0, -1}).unsqueeze(0).to(torch::kCPU);
        const int32_t next_id = static_cast<int32_t>(sample_next(logits, generated, options).item<int64_t>());
        generated.push_back(next_id);
        const auto full = tokenizer_.decode(generated);
        const auto delta = full.size() >= prompt.size() + emitted.size() ? full.substr(prompt.size() + emitted.size()) : std::string{};
        if (!delta.empty()) { emitted += delta; on_chunk(delta); if (should_stop(emitted, options)) break; }
    }
    return emitted;
}

const AppConfig& Generator::config() const { return config_; }

}  // namespace gpt2cpp
#endif
