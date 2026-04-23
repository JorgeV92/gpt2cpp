#ifndef GPT2CPP_APP_CHAT_REPL_HPP
#define GPT2CPP_APP_CHAT_REPL_HPP
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/inference/generator.hpp"
namespace gpt2cpp {
class ChatRepl {
public:
    ChatRepl(const std::filesystem::path& model_bundle, std::string system_prompt);
    void run();
private:
    std::string render_prompt() const;
    Generator generator_;
    std::string system_prompt_;
    std::vector<std::pair<std::string, std::string>> messages_;
};
}  // namespace gpt2cpp
#endif
#endif
