#ifndef GPT2CPP_CORE_ERROR_HPP
#define GPT2CPP_CORE_ERROR_HPP
#include <stdexcept>
#include <string>
namespace gpt2cpp {
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};
void require(bool condition, const std::string& message);
}  // namespace gpt2cpp
#endif
