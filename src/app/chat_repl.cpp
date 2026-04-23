#include "gpt2cpp/app/chat_repl.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/core/files.hpp"
#include <iostream>
#include <sstream>
namespace gpt2cpp {
namespace {
std::string color(const std::string& s, const std::string& code) { return "\033[" + code + "m" + s + "\033[0m"; }
}
ChatRepl::ChatRepl(const std::filesystem::path& model_bundle, std::string system_prompt)
    : generator_(model_bundle), system_prompt_(std::move(system_prompt)) {}

std::string ChatRepl::render_prompt() const {
    std::ostringstream ss;
    ss << "<|system|>\n" << system_prompt_ << "\n";
    for (const auto& [role, content] : messages_) ss << "<|" << role << "|>\n" << content << "\n";
    ss << "<|assistant|>\n";
    return ss.str();
}

void ChatRepl::run() {
    std::cout << "gpt2cpp chat\nType /help for commands.\n\n";
    while (true) {
        std::cout << color("user> ", "36");
        std::string input; if (!std::getline(std::cin, input)) { std::cout << '\n'; break; }
        if (input == "/help") {
            std::cout << "/help /reset /save /params\n"; continue;
        }
        if (input == "/reset") { messages_.clear(); std::cout << color("system> conversation reset\n", "33"); continue; }
        if (input == "/save") {
            std::ostringstream ss; ss << "system: " << system_prompt_ << "\n";
            for (const auto& [role, content] : messages_) ss << role << ": " << content << "\n";
            write_text_file("chat_transcript.txt", ss.str());
            std::cout << color("system> saved to chat_transcript.txt\n", "33"); continue;
        }
        if (input == "/params") {
            const auto& inf = generator_.config().inference;
            std::cout << "temperature=" << inf.temperature << " top_k=" << inf.top_k << " top_p=" << inf.top_p << " max_new_tokens=" << inf.max_new_tokens << "\n"; continue;
        }

        messages_.push_back({"user", input});
        GenerationOptions opt;
        opt.temperature = generator_.config().inference.temperature;
        opt.top_k = generator_.config().inference.top_k;
        opt.top_p = generator_.config().inference.top_p;
        opt.max_new_tokens = generator_.config().inference.max_new_tokens;
        opt.repetition_penalty = generator_.config().inference.repetition_penalty;
        opt.stop_sequences = generator_.config().inference.stop_sequences;

        std::cout << color("assistant> ", "32");
        const auto response = generator_.generate_streaming(render_prompt(), opt, [](const std::string& chunk){ std::cout << chunk << std::flush; });
        std::cout << "\n\n";
        messages_.push_back({"assistant", response});
    }
}
}  // namespace gpt2cpp
#endif
