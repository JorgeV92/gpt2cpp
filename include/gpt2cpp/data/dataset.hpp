#ifndef GPT2CPP_DATA_DATASET_HPP
#define GPT2CPP_DATA_DATASET_HPP
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#ifdef GPT2CPP_HAVE_TORCH
#include <torch/torch.h>
#endif
namespace gpt2cpp {
class BPETokenizer;
struct PackedDatasetInfo { std::size_t token_count{0}; int32_t min_id{0}; int32_t max_id{0}; std::filesystem::path path; };
class PackedDataset {
public:
    PackedDataset() = default;
    explicit PackedDataset(std::vector<int32_t> tokens);
    void save(const std::filesystem::path& path) const;
    static PackedDataset load(const std::filesystem::path& path);
    static PackedDatasetInfo inspect(const std::filesystem::path& path);
    std::size_t size() const;
    const std::vector<int32_t>& tokens() const;
#ifdef GPT2CPP_HAVE_TORCH
    std::pair<torch::Tensor, torch::Tensor> sample_batch(int batch_size, int sequence_length, std::uint64_t seed, const torch::Device& device) const;
#endif
private:
    std::vector<int32_t> tokens_;
};
struct PrepareDataOptions { std::filesystem::path input; std::filesystem::path output; double val_ratio{0.1}; std::string jsonl_text_field{"text"}; };
std::string load_corpus_text(const std::filesystem::path& input, const std::string& jsonl_text_field);
void prepare_packed_dataset(const BPETokenizer& tokenizer, const PrepareDataOptions& options);
}  // namespace gpt2cpp
#endif
