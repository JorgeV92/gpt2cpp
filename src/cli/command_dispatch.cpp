#include "gpt2cpp/cli/command_dispatch.hpp"
#include "gpt2cpp/core/config.hpp"
#include "gpt2cpp/core/error.hpp"
#include "gpt2cpp/core/files.hpp"
#include "gpt2cpp/core/logging.hpp"
#include "gpt2cpp/data/dataset.hpp"
#include "gpt2cpp/tokenizer/bpe_tokenizer.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/app/chat_repl.hpp"
#include "gpt2cpp/inference/generator.hpp"
#include "gpt2cpp/serving/http_server.hpp"
#include "gpt2cpp/training/trainer.hpp"
#endif
#include <algorithm>
#include <iostream>

namespace gpt2cpp {
namespace {
class Args {
public:
    explicit Args(std::vector<std::string> a) : args_(std::move(a)) {}
    std::string get(const std::string& name, const std::string& def = "") const {
        for (std::size_t i = 0; i + 1 < args_.size(); ++i) if (args_[i] == name) return args_[i + 1];
        return def;
    }
private:
    std::vector<std::string> args_;
};

std::vector<std::string> csv(const std::string& s) {
    if (trim(s).empty()) return {};
    auto xs = split(s, ','); for (auto& x : xs) x = trim(x); return xs;
}

int tokenizer_cmd(const std::vector<std::string>& args) {
    require(args.size() >= 2, "tokenizer requires a subcommand");
    const Args a(args);
    if (args[1] == "train") {
        BPETrainingOptions opt;
        opt.vocab_size = std::stoi(a.get("--vocab-size", "1024"));
        opt.special_tokens = csv(a.get("--special"));
        BPETokenizer tok; tok.train_from_text(load_corpus_text(a.get("--input"), "text"), opt); tok.save(a.get("--output"));
        std::cout << "Tokenizer trained and saved to " << a.get("--output") << "\n"; return 0;
    }
    if (args[1] == "encode") {
        auto tok = BPETokenizer::load(a.get("--tokenizer"));
        auto ids = tok.encode(a.get("--text"));
        for (std::size_t i = 0; i < ids.size(); ++i) { if (i) std::cout << ','; std::cout << ids[i]; }
        std::cout << "\n"; return 0;
    }
    if (args[1] == "decode") {
        auto tok = BPETokenizer::load(a.get("--tokenizer"));
        std::vector<int32_t> ids; for (const auto& x : split(a.get("--ids"), ',')) if (!trim(x).empty()) ids.push_back(static_cast<int32_t>(std::stoi(trim(x))));
        std::cout << tok.decode(ids) << "\n"; return 0;
    }
    throw Error("Unknown tokenizer subcommand");
}

int data_cmd(const std::vector<std::string>& args) {
    require(args.size() >= 2, "data requires a subcommand");
    const Args a(args);
    if (args[1] == "prepare") {
        auto tok = BPETokenizer::load(a.get("--tokenizer"));
        PrepareDataOptions opt; opt.input = a.get("--input"); opt.output = a.get("--output"); opt.val_ratio = std::stod(a.get("--val-ratio", "0.1")); opt.jsonl_text_field = a.get("--jsonl-text-field", "text");
        prepare_packed_dataset(tok, opt);
        std::cout << "Packed dataset written to " << opt.output << "\n"; return 0;
    }
    if (args[1] == "inspect" || args[1] == "stats") {
        auto info = PackedDataset::inspect(a.get("--dataset"));
        std::cout << "path=" << info.path.string() << "\n"
                  << "token_count=" << info.token_count << "\n"
                  << "min_id=" << info.min_id << "\n"
                  << "max_id=" << info.max_id << "\n";
        return 0;
    }
    throw Error("Unknown data subcommand");
}
}  // namespace

int dispatch_command(const std::vector<std::string>& args) {
    try {
        if (args.empty()) {
            std::cout << "gpt2cpp commands: tokenizer, data, train, eval, generate, chat, serve\n";
            return 0;
        }
        if (args[0] == "tokenizer") return tokenizer_cmd(args);
        if (args[0] == "data") return data_cmd(args);

#ifdef GPT2CPP_HAVE_TORCH
        const Args a(args);
        if (args[0] == "train") { Trainer(load_config(a.get("--config"))).run(); return 0; }
        if (args[0] == "eval") { Trainer t(load_config(a.get("--config"))); t.evaluate_checkpoint(a.get("--checkpoint")); return 0; }
        if (args[0] == "generate") {
            Generator g(a.get("--model"));
            GenerationOptions opt; opt.temperature = std::stod(a.get("--temperature", std::to_string(g.config().inference.temperature)));
            opt.top_k = std::stoi(a.get("--top-k", std::to_string(g.config().inference.top_k)));
            opt.top_p = std::stod(a.get("--top-p", std::to_string(g.config().inference.top_p)));
            opt.max_new_tokens = std::stoi(a.get("--max-new-tokens", std::to_string(g.config().inference.max_new_tokens)));
            opt.repetition_penalty = std::stod(a.get("--repetition-penalty", std::to_string(g.config().inference.repetition_penalty)));
            opt.stop_sequences = g.config().inference.stop_sequences;
            std::cout << g.generate(a.get("--prompt"), opt) << "\n"; return 0;
        }
        if (args[0] == "chat") { ChatRepl(a.get("--model"), a.get("--system", "You are a concise helpful assistant.")).run(); return 0; }
        if (args[0] == "serve") {
            auto generator = std::make_shared<Generator>(a.get("--model"));
            HttpServer(generator, generator->config().serving.host, std::stoi(a.get("--port", std::to_string(generator->config().serving.port))), a.get("--web-root", generator->config().serving.web_root)).serve_forever();
            return 0;
        }
#else
        if (args[0] == "train" || args[0] == "eval" || args[0] == "generate" || args[0] == "chat" || args[0] == "serve")
            throw Error("This build was compiled without LibTorch. Reconfigure with GPT2CPP_ENABLE_TORCH=ON and CMAKE_PREFIX_PATH pointing at LibTorch.");
#endif
        throw Error("Unknown command: " + args[0]);
    } catch (const std::exception& ex) {
        GPT2CPP_LOG_ERROR(ex.what());
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}

int dispatch_alias_command(const std::string& alias, const std::vector<std::string>& args) {
    if (alias == "train_tokenizer") { auto a = std::vector<std::string>{"tokenizer", "train"}; a.insert(a.end(), args.begin(), args.end()); return dispatch_command(a); }
    if (alias == "prepare_data")   { auto a = std::vector<std::string>{"data", "prepare"}; a.insert(a.end(), args.begin(), args.end()); return dispatch_command(a); }
    if (alias == "train_model")    { auto a = std::vector<std::string>{"train"}; a.insert(a.end(), args.begin(), args.end()); return dispatch_command(a); }
    if (alias == "eval_model")     { auto a = std::vector<std::string>{"eval"}; a.insert(a.end(), args.begin(), args.end()); return dispatch_command(a); }
    if (alias == "generate")       { auto a = std::vector<std::string>{"generate"}; a.insert(a.end(), args.begin(), args.end()); return dispatch_command(a); }
    if (alias == "chat_cli")       { auto a = std::vector<std::string>{"chat"}; a.insert(a.end(), args.begin(), args.end()); return dispatch_command(a); }
    if (alias == "serve_web")      { auto a = std::vector<std::string>{"serve"}; a.insert(a.end(), args.begin(), args.end()); return dispatch_command(a); }
    return dispatch_command(args);
}

}  // namespace gpt2cpp
