#ifndef GPT2CPP_CLI_COMMAND_DISPATCH_HPP
#define GPT2CPP_CLI_COMMAND_DISPATCH_HPP
#include <string>
#include <vector>
namespace gpt2cpp {
int dispatch_command(const std::vector<std::string>& args);
int dispatch_alias_command(const std::string& alias, const std::vector<std::string>& args);
}  // namespace gpt2cpp
#endif
