#include "gpt2cpp/tokenizer/bpe_tokenizer.hpp"
#include "gpt2cpp/core/error.hpp"
#include "gpt2cpp/core/files.hpp"
#include <algorithm>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace gpt2cpp {

std::size_t BPETokenizer::PairHash::operator()(const std::pair<int32_t, int32_t>& v) const noexcept {
    return (static_cast<std::size_t>(v.first) << 32U) ^ static_cast<std::size_t>(v.second);
}

BPETokenizer::BPETokenizer() { reset(); }

void BPETokenizer::reset() {
    id_to_bytes_.clear();
    token_to_id_.clear();
    merges_.clear();
    merge_ranks_.clear();
    special_to_id_.clear();
    special_tokens_.clear();
    init_base_vocab();
}

void BPETokenizer::init_base_vocab() {
    for (int i = 0; i < 256; ++i) {
        std::vector<uint8_t> bytes{static_cast<uint8_t>(i)};
        id_to_bytes_.push_back(bytes);
        token_to_id_[std::string(bytes.begin(), bytes.end())] = i;
    }
}

void BPETokenizer::train_from_text(const std::string& text, const BPETrainingOptions& options) {
    reset();
    std::vector<std::vector<int32_t>> seqs;
    for (unsigned char c : text) seqs.push_back({static_cast<int32_t>(c)});

    while (static_cast<int>(id_to_bytes_.size() + options.special_tokens.size()) < options.vocab_size) {
        std::unordered_map<std::pair<int32_t, int32_t>, std::size_t, PairHash> counts;
        for (const auto& s : seqs) for (std::size_t i = 0; i + 1 < s.size(); ++i) ++counts[{s[i], s[i + 1]}];
        if (counts.empty()) break;
        auto best = std::max_element(counts.begin(), counts.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
        if (best == counts.end() || best->second < 2) break;

        const auto pair = best->first;
        std::vector<uint8_t> bytes = id_to_bytes_.at(static_cast<std::size_t>(pair.first));
        const auto& rhs = id_to_bytes_.at(static_cast<std::size_t>(pair.second));
        bytes.insert(bytes.end(), rhs.begin(), rhs.end());
        const std::string key(bytes.begin(), bytes.end());
        const int32_t new_id = static_cast<int32_t>(id_to_bytes_.size());
        id_to_bytes_.push_back(bytes);
        token_to_id_[key] = new_id;
        merge_ranks_[pair] = static_cast<int32_t>(merges_.size());
        merges_.push_back(pair);

        for (auto& s : seqs) {
            std::vector<int32_t> out;
            for (std::size_t i = 0; i < s.size();) {
                if (i + 1 < s.size() && s[i] == pair.first && s[i + 1] == pair.second) { out.push_back(new_id); i += 2; }
                else { out.push_back(s[i]); ++i; }
            }
            s.swap(out);
        }
    }

    for (const auto& tok : options.special_tokens) {
        const int32_t id = static_cast<int32_t>(id_to_bytes_.size());
        id_to_bytes_.push_back(std::vector<uint8_t>(tok.begin(), tok.end()));
        special_tokens_.push_back(tok);
        special_to_id_[tok] = id;
        token_to_id_[tok] = id;
    }
}

std::pair<bool, std::pair<int32_t, std::size_t>> BPETokenizer::match_special(const std::string& text, std::size_t pos) const {
    std::size_t best_len = 0; int32_t best_id = -1;
    for (const auto& [tok, id] : special_to_id_) {
        if (pos + tok.size() <= text.size() && text.compare(pos, tok.size(), tok) == 0 && tok.size() > best_len) {
            best_len = tok.size(); best_id = id;
        }
    }
    if (best_id >= 0) return {true, {best_id, best_len}};
    return {false, {-1, 0}};
}

std::vector<int32_t> BPETokenizer::encode_segment(const std::string& text) const {
    std::vector<int32_t> ids; for (unsigned char c : text) ids.push_back(static_cast<int32_t>(c));
    bool changed = true;
    while (changed && ids.size() >= 2) {
        changed = false;
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        std::size_t best_pos = ids.size();
        for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
            auto it = merge_ranks_.find({ids[i], ids[i + 1]});
            if (it != merge_ranks_.end() && it->second < best_rank) { best_rank = it->second; best_pos = i; }
        }
        if (best_pos < ids.size()) {
            const auto pair = std::make_pair(ids[best_pos], ids[best_pos + 1]);
            std::vector<uint8_t> bytes = id_to_bytes_.at(static_cast<std::size_t>(pair.first));
            const auto& rhs = id_to_bytes_.at(static_cast<std::size_t>(pair.second));
            bytes.insert(bytes.end(), rhs.begin(), rhs.end());
            const int32_t merged_id = token_to_id_.at(std::string(bytes.begin(), bytes.end()));
            std::vector<int32_t> out;
            for (std::size_t i = 0; i < ids.size();) {
                if (i == best_pos) { out.push_back(merged_id); i += 2; }
                else { out.push_back(ids[i]); ++i; }
            }
            ids.swap(out); changed = true;
        }
    }
    return ids;
}

std::vector<int32_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids; std::string cur;
    auto flush = [&](){ if (!cur.empty()) { auto part = encode_segment(cur); ids.insert(ids.end(), part.begin(), part.end()); cur.clear(); } };
    for (std::size_t i = 0; i < text.size();) {
        auto [hit, payload] = match_special(text, i);
        if (hit) { flush(); ids.push_back(payload.first); i += payload.second; }
        else { cur.push_back(text[i]); ++i; }
    }
    flush(); return ids;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        require(id >= 0 && static_cast<std::size_t>(id) < id_to_bytes_.size(), "Decode token id out of range.");
        const auto& bytes = id_to_bytes_.at(static_cast<std::size_t>(id));
        out.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    return out;
}

void BPETokenizer::save(const std::filesystem::path& dir) const {
    ensure_directory(dir);
    std::ostringstream meta; meta << "version\t1\n" << "vocab_size\t" << id_to_bytes_.size() << "\n" << "merge_count\t" << merges_.size() << "\n";
    write_text_file(dir / "tokenizer.meta", meta.str());

    std::ostringstream vocab;
    for (std::size_t i = 0; i < id_to_bytes_.size(); ++i) {
        vocab << i << '\t';
        for (std::size_t j = 0; j < id_to_bytes_[i].size(); ++j) { if (j) vocab << ' '; vocab << static_cast<int>(id_to_bytes_[i][j]); }
        vocab << '\n';
    }
    write_text_file(dir / "vocab.tsv", vocab.str());

    std::ostringstream merges_out;
    for (std::size_t i = 0; i < merges_.size(); ++i) merges_out << i << '\t' << merges_[i].first << '\t' << merges_[i].second << '\n';
    write_text_file(dir / "merges.tsv", merges_out.str());

    std::ostringstream specials;
    for (const auto& s : special_tokens_) specials << s << '\t' << special_to_id_.at(s) << '\n';
    write_text_file(dir / "special_tokens.tsv", specials.str());
}

BPETokenizer BPETokenizer::load(const std::filesystem::path& dir) {
    BPETokenizer tok;
    tok.id_to_bytes_.clear(); tok.token_to_id_.clear(); tok.merges_.clear(); tok.merge_ranks_.clear(); tok.special_to_id_.clear(); tok.special_tokens_.clear();

    std::istringstream vocab_in(read_text_file(dir / "vocab.tsv"));
    for (std::string line; std::getline(vocab_in, line); ) {
        if (trim(line).empty()) continue;
        auto parts = split(line, '\t'); require(parts.size() == 2, "Invalid vocab.tsv line");
        std::vector<uint8_t> bytes;
        for (const auto& x : split(trim(parts[1]), ' ')) if (!x.empty()) bytes.push_back(static_cast<uint8_t>(std::stoi(x)));
        tok.id_to_bytes_.push_back(bytes);
        tok.token_to_id_[std::string(bytes.begin(), bytes.end())] = static_cast<int32_t>(tok.id_to_bytes_.size() - 1);
    }

    std::istringstream merges_in(read_text_file(dir / "merges.tsv"));
    for (std::string line; std::getline(merges_in, line); ) {
        if (trim(line).empty()) continue;
        auto parts = split(line, '\t'); require(parts.size() == 3, "Invalid merges.tsv line");
        const int32_t a = static_cast<int32_t>(std::stoi(parts[1])), b = static_cast<int32_t>(std::stoi(parts[2]));
        tok.merge_ranks_[{a, b}] = static_cast<int32_t>(tok.merges_.size());
        tok.merges_.push_back({a, b});
    }

    const auto sp = dir / "special_tokens.tsv";
    if (std::filesystem::exists(sp)) {
        std::istringstream sp_in(read_text_file(sp));
        for (std::string line; std::getline(sp_in, line); ) {
            if (trim(line).empty()) continue;
            auto parts = split(line, '\t'); require(parts.size() == 2, "Invalid special_tokens.tsv line");
            tok.special_tokens_.push_back(parts[0]);
            tok.special_to_id_[parts[0]] = static_cast<int32_t>(std::stoi(parts[1]));
            tok.token_to_id_[parts[0]] = static_cast<int32_t>(std::stoi(parts[1]));
        }
    }
    return tok;
}

int BPETokenizer::vocab_size() const { return static_cast<int>(id_to_bytes_.size()); }

}  // namespace gpt2cpp
