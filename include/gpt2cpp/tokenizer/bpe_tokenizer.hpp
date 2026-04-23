#ifndef GPT2CPP_TOKENIZER_BPE_TOKENIZER_HPP
#define GPT2CPP_TOKENIZER_BPE_TOKENIZER_HPP
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace gpt2cpp {
struct BPETrainingOptions { int vocab_size{1024}; std::vector<std::string> special_tokens{}; };
class BPETokenizer {
public:
    BPETokenizer();
    void train_from_text(const std::string& text, const BPETrainingOptions& options);
    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    void save(const std::filesystem::path& directory) const;
    static BPETokenizer load(const std::filesystem::path& directory);
    int vocab_size() const;
private:
    struct PairHash { std::size_t operator()(const std::pair<int32_t, int32_t>& v) const noexcept; };
    void reset();
    void init_base_vocab();
    std::vector<int32_t> encode_segment(const std::string& text) const;
    std::pair<bool, std::pair<int32_t, std::size_t>> match_special(const std::string& text, std::size_t pos) const;
    std::vector<std::vector<uint8_t>> id_to_bytes_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::pair<int32_t, int32_t>> merges_;
    std::unordered_map<std::pair<int32_t, int32_t>, int32_t, PairHash> merge_ranks_;
    std::unordered_map<std::string, int32_t> special_to_id_;
    std::vector<std::string> special_tokens_;
};
}  // namespace gpt2cpp
#endif
