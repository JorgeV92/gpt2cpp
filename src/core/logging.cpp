#include "gpt2cpp/core/logging.hpp"
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace gpt2cpp {
namespace {
std::string now_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream ss; ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S"); return ss.str();
}
}  // namespace

Logger& Logger::instance() { static Logger logger; return logger; }
void Logger::set_log_file(const std::filesystem::path& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    file_.open(path, std::ios::out | std::ios::app);
}
std::string level_to_string(LogLevel level) {
    switch (level) { case LogLevel::Debug: return "DEBUG"; case LogLevel::Info: return "INFO"; case LogLevel::Warn: return "WARN"; case LogLevel::Error: return "ERROR"; }
    return "INFO";
}
void Logger::log(LogLevel level, const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string line = "[" + now_string() + "] [" + level_to_string(level) + "] " + message;
    std::cerr << line << '\n';
    if (file_.is_open()) { file_ << line << '\n'; file_.flush(); }
}
}  // namespace gpt2cpp
