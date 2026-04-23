#ifndef GPT2CPP_CORE_LOGGING_HPP
#define GPT2CPP_CORE_LOGGING_HPP
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
namespace gpt2cpp {
enum class LogLevel { Debug, Info, Warn, Error };
class Logger {
public:
    static Logger& instance();
    void set_log_file(const std::filesystem::path& path);
    void log(LogLevel level, const std::string& message);
private:
    Logger() = default;
    std::mutex mutex_;
    std::ofstream file_;
};
std::string level_to_string(LogLevel level);
#define GPT2CPP_LOG_INFO(msg)  ::gpt2cpp::Logger::instance().log(::gpt2cpp::LogLevel::Info,  (msg))
#define GPT2CPP_LOG_WARN(msg)  ::gpt2cpp::Logger::instance().log(::gpt2cpp::LogLevel::Warn,  (msg))
#define GPT2CPP_LOG_ERROR(msg) ::gpt2cpp::Logger::instance().log(::gpt2cpp::LogLevel::Error, (msg))
}  // namespace gpt2cpp
#endif
