#include "gpt2cpp/data/dataset.hpp"
#include "gpt2cpp/core/error.hpp"
#include "gpt2cpp/core/files.hpp"
#include "gpt2cpp/tokenizer/bpe_tokenizer.hpp"
#include <algorithm>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>

namespace gpt2cpp {
namespace {
constexpr std::uint32_t kMagic = 0x47505432U;
constexpr std::uint32_t kVersion = 1U;

std::string extract_jsonl_text(const std::string& line, const std::string& field) {
    const std::regex pat("\"" + field + "\"\\s*:\\s*\"((?:\\\\.|[^\"])*)\"");
    std::smatch match;
    if (!std::regex_search(line, match, pat)) return {};
    std::string value = match[1].str(); std::string out;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 1 < value.size()) { out.push_back(value[i + 1] == 'n' ? '\n' : value[i + 1]); ++i; }
        else out.push_back(value[i]);
    }
    return out;
}
}  // namespace

PackedDataset::PackedDataset(std::vector<int32_t> tokens) : tokens_(std::move(tokens)) {}

void PackedDataset::save(const std::filesystem::path& path) const {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "Failed to open dataset for write: " + path.string());
    const std::uint64_t count = static_cast<std::uint64_t>(tokens_.size());
    out.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    out.write(reinterpret_cast<const char*>(&kVersion), sizeof(kVersion));
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    out.write(reinterpret_cast<const char*>(tokens_.data()), static_cast<std::streamsize>(tokens_.size() * sizeof(int32_t)));
}

PackedDataset PackedDataset::load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "Failed to open dataset for read: " + path.string());
    std::uint32_t magic = 0, version = 0; std::uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&version), sizeof(version));
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    require(magic == kMagic && version == kVersion, "Invalid packed dataset header.");
    std::vector<int32_t> ids(static_cast<std::size_t>(count));
    in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(int32_t)));
    return PackedDataset(std::move(ids));
}

PackedDatasetInfo PackedDataset::inspect(const std::filesystem::path& path) {
    PackedDataset ds = PackedDataset::load(path);
    PackedDatasetInfo info; info.path = path; info.token_count = ds.tokens_.size();
    if (!ds.tokens_.empty()) {
        auto [mn, mx] = std::minmax_element(ds.tokens_.begin(), ds.tokens_.end());
        info.min_id = *mn; info.max_id = *mx;
    }
    return info;
}

std::size_t PackedDataset::size() const { return tokens_.size(); }
const std::vector<int32_t>& PackedDataset::tokens() const { return tokens_; }

#ifdef GPT2CPP_HAVE_TORCH
std::pair<torch::Tensor, torch::Tensor> PackedDataset::sample_batch(int batch_size, int sequence_length, std::uint64_t seed, const torch::Device& device) const {
    require(tokens_.size() > static_cast<std::size_t>(sequence_length + 1), "Dataset too small for requested sequence length.");
    std::mt19937_64 rng(seed);
    const std::size_t max_start = tokens_.size() - static_cast<std::size_t>(sequence_length) - 1;
    std::uniform_int_distribution<std::size_t> dist(0, max_start);

    std::vector<int64_t> x(static_cast<std::size_t>(batch_size * sequence_length));
    std::vector<int64_t> y(static_cast<std::size_t>(batch_size * sequence_length));
    for (int b = 0; b < batch_size; ++b) {
        const auto start = dist(rng);
        for (int t = 0; t < sequence_length; ++t) {
            x[static_cast<std::size_t>(b * sequence_length + t)] = tokens_[start + static_cast<std::size_t>(t)];
            y[static_cast<std::size_t>(b * sequence_length + t)] = tokens_[start + static_cast<std::size_t>(t) + 1];
        }
    }
    auto xt = torch::from_blob(x.data(), {batch_size, sequence_length}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(device);
    auto yt = torch::from_blob(y.data(), {batch_size, sequence_length}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(device);
    return {xt, yt};
}
#endif

std::string load_corpus_text(const std::filesystem::path& input, const std::string& jsonl_text_field) {
    std::ostringstream ss;
    for (const auto& path : list_files_recursive(input)) {
        const auto ext = to_lower(path.extension().string());
        if (ext == ".txt") ss << read_text_file(path) << '\n';
        else if (ext == ".jsonl") {
            std::istringstream in(read_text_file(path));
            for (std::string line; std::getline(in, line); ) {
                const auto text = extract_jsonl_text(line, jsonl_text_field);
                if (!text.empty()) ss << text << '\n';
            }
        }
    }
    return ss.str();
}

void prepare_packed_dataset(const BPETokenizer& tokenizer, const PrepareDataOptions& options) {
    ensure_directory(options.output);
    const auto corpus = load_corpus_text(options.input, options.jsonl_text_field);
    require(!corpus.empty(), "No text found for dataset preparation.");
    const auto ids = tokenizer.encode(corpus);
    require(ids.size() > 1, "Tokenized corpus too small.");

    const auto split_idx = static_cast<std::size_t>((1.0 - options.val_ratio) * static_cast<double>(ids.size()));
    const auto cut = std::min(ids.size() - 1, std::max<std::size_t>(1, split_idx));
    PackedDataset(std::vector<int32_t>(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(cut))).save(options.output / "train.bin");
    PackedDataset(std::vector<int32_t>(ids.begin() + static_cast<std::ptrdiff_t>(cut), ids.end())).save(options.output / "val.bin");

    std::ostringstream meta;
    meta << "total_tokens=" << ids.size() << "\ntrain_tokens=" << cut << "\nval_tokens=" << (ids.size() - cut) << "\n";
    write_text_file(options.output / "dataset.meta", meta.str());
}

}  // namespace gpt2cpp
