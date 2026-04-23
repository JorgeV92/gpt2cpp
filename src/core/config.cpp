#include "gpt2cpp/core/config.hpp"
#include "gpt2cpp/core/error.hpp"
#include "gpt2cpp/core/files.hpp"
#include <fstream>
#include <sstream>

namespace gpt2cpp {
namespace {
std::string unquote(std::string v) {
    v = trim(v);
    if (v.size() >= 2 && v.front() == '"' && v.back() == '"') v = v.substr(1, v.size() - 2);
    return v;
}
bool parse_bool(const std::string& v) { const auto t = to_lower(trim(v)); if (t == "true") return true; if (t == "false") return false; throw Error("Invalid bool: " + v); }
int parse_int(const std::string& v) { return std::stoi(trim(v)); }
double parse_double(const std::string& v) { return std::stod(trim(v)); }
std::vector<std::string> parse_csv(const std::string& v) { auto raw = unquote(v); if (trim(raw).empty()) return {}; auto xs = split(raw, ','); for (auto& x : xs) x = trim(x); return xs; }

void set_kv(AppConfig& c, const std::string& s, const std::string& k, const std::string& v) {
    if (s == "run") {
        if (k == "name") c.run.name = unquote(v);
        else if (k == "output_root") c.run.output_root = unquote(v);
        else if (k == "device") c.run.device = unquote(v);
        else if (k == "seed") c.run.seed = parse_int(v);
        else if (k == "mixed_precision") c.run.mixed_precision = parse_bool(v);
        else throw Error("Unknown run key: " + k);
    } else if (s == "tokenizer") {
        if (k == "dir") c.tokenizer.dir = unquote(v); else throw Error("Unknown tokenizer key: " + k);
    } else if (s == "data") {
        if (k == "train_bin") c.data.train_bin = unquote(v);
        else if (k == "val_bin") c.data.val_bin = unquote(v);
        else if (k == "batch_size") c.data.batch_size = parse_int(v);
        else if (k == "sequence_length") c.data.sequence_length = parse_int(v);
        else if (k == "jsonl_text_field") c.data.jsonl_text_field = unquote(v);
        else throw Error("Unknown data key: " + k);
    } else if (s == "model") {
        if (k == "vocab_size") c.model.vocab_size = parse_int(v);
        else if (k == "max_position_embeddings") c.model.max_position_embeddings = parse_int(v);
        else if (k == "n_embd") c.model.n_embd = parse_int(v);
        else if (k == "n_head") c.model.n_head = parse_int(v);
        else if (k == "n_layer") c.model.n_layer = parse_int(v);
        else if (k == "dropout") c.model.dropout = parse_double(v);
        else if (k == "tie_weights") c.model.tie_weights = parse_bool(v);
        else throw Error("Unknown model key: " + k);
    } else if (s == "training") {
        if (k == "max_steps") c.training.max_steps = parse_int(v);
        else if (k == "learning_rate") c.training.learning_rate = parse_double(v);
        else if (k == "weight_decay") c.training.weight_decay = parse_double(v);
        else if (k == "warmup_steps") c.training.warmup_steps = parse_int(v);
        else if (k == "grad_clip") c.training.grad_clip = parse_double(v);
        else if (k == "eval_interval") c.training.eval_interval = parse_int(v);
        else if (k == "save_interval") c.training.save_interval = parse_int(v);
        else if (k == "log_interval") c.training.log_interval = parse_int(v);
        else if (k == "eval_batches") c.training.eval_batches = parse_int(v);
        else if (k == "resume_from") c.training.resume_from = unquote(v);
        else throw Error("Unknown training key: " + k);
    } else if (s == "inference") {
        if (k == "temperature") c.inference.temperature = parse_double(v);
        else if (k == "top_k") c.inference.top_k = parse_int(v);
        else if (k == "top_p") c.inference.top_p = parse_double(v);
        else if (k == "max_new_tokens") c.inference.max_new_tokens = parse_int(v);
        else if (k == "repetition_penalty") c.inference.repetition_penalty = parse_double(v);
        else if (k == "stop_sequences") c.inference.stop_sequences = parse_csv(v);
        else throw Error("Unknown inference key: " + k);
    } else if (s == "serving") {
        if (k == "host") c.serving.host = unquote(v);
        else if (k == "port") c.serving.port = parse_int(v);
        else if (k == "web_root") c.serving.web_root = unquote(v);
        else throw Error("Unknown serving key: " + k);
    } else {
        throw Error("Unknown section: " + s);
    }
}
}  // namespace

AppConfig load_config(const std::filesystem::path& path) {
    std::ifstream in(path);
    require(in.good(), "Failed to open config: " + path.string());
    AppConfig c;
    std::string section;
    for (std::string line; std::getline(in, line); ) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        if (line.front() == '[' && line.back() == ']') { section = trim(line.substr(1, line.size() - 2)); continue; }
        const auto pos = line.find('=');
        if (pos == std::string::npos) throw Error("Invalid config line: " + line);
        set_kv(c, section, trim(line.substr(0, pos)), trim(line.substr(pos + 1)));
    }
    return c;
}

void save_config(const AppConfig& c, const std::filesystem::path& path) {
    std::ostringstream ss;
    ss << "[run]\nname = \"" << c.run.name << "\"\noutput_root = \"" << c.run.output_root << "\"\nseed = " << c.run.seed << "\ndevice = \"" << c.run.device << "\"\nmixed_precision = " << (c.run.mixed_precision ? "true" : "false") << "\n\n";
    ss << "[tokenizer]\ndir = \"" << c.tokenizer.dir << "\"\n\n";
    ss << "[data]\ntrain_bin = \"" << c.data.train_bin << "\"\nval_bin = \"" << c.data.val_bin << "\"\nbatch_size = " << c.data.batch_size << "\nsequence_length = " << c.data.sequence_length << "\njsonl_text_field = \"" << c.data.jsonl_text_field << "\"\n\n";
    ss << "[model]\nvocab_size = " << c.model.vocab_size << "\nmax_position_embeddings = " << c.model.max_position_embeddings << "\nn_embd = " << c.model.n_embd << "\nn_head = " << c.model.n_head << "\nn_layer = " << c.model.n_layer << "\ndropout = " << c.model.dropout << "\ntie_weights = " << (c.model.tie_weights ? "true" : "false") << "\n\n";
    ss << "[training]\nmax_steps = " << c.training.max_steps << "\nlearning_rate = " << c.training.learning_rate << "\nweight_decay = " << c.training.weight_decay << "\nwarmup_steps = " << c.training.warmup_steps << "\ngrad_clip = " << c.training.grad_clip << "\neval_interval = " << c.training.eval_interval << "\nsave_interval = " << c.training.save_interval << "\nlog_interval = " << c.training.log_interval << "\neval_batches = " << c.training.eval_batches << "\nresume_from = \"" << c.training.resume_from << "\"\n\n";
    ss << "[inference]\ntemperature = " << c.inference.temperature << "\ntop_k = " << c.inference.top_k << "\ntop_p = " << c.inference.top_p << "\nmax_new_tokens = " << c.inference.max_new_tokens << "\nrepetition_penalty = " << c.inference.repetition_penalty << "\nstop_sequences = \"" << join(c.inference.stop_sequences, ",") << "\"\n\n";
    ss << "[serving]\nhost = \"" << c.serving.host << "\"\nport = " << c.serving.port << "\nweb_root = \"" << c.serving.web_root << "\"\n";
    write_text_file(path, ss.str());
}

}  // namespace gpt2cpp
