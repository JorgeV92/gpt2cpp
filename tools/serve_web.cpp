#include "gpt2cpp/cli/command_dispatch.hpp"
#include <string>
#include <vector>
int main(int argc, char** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.emplace_back(argv[i]);
    return gpt2cpp::dispatch_alias_command("serve_web", args);
}
