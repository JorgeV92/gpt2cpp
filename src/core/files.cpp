#include "gpt2cpp/core/files.hpp"
#include "gpt2cpp/core/error.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace gpt2cpp {

void require(bool condition, const std::string& message) {
    if (!condition) throw Error(message);
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "Failed to open file for reading: " + path.string());
    std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}

void write_text_file(const std::filesystem::path& path, const std::string& content) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "Failed to open file for writing: " + path.string());
    out << content;
}

std::vector<std::filesystem::path> list_files_recursive(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> out;
    if (std::filesystem::is_regular_file(root)) { out.push_back(root); return out; }
    require(std::filesystem::exists(root), "Path not found: " + root.string());
    for (const auto& e : std::filesystem::recursive_directory_iterator(root))
        if (e.is_regular_file()) out.push_back(e.path());
    std::sort(out.begin(), out.end());
    return out;
}

std::string trim(const std::string& value) {
    const auto b = std::find_if_not(value.begin(), value.end(), [](unsigned char c){ return std::isspace(c) != 0; });
    const auto e = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c){ return std::isspace(c) != 0; }).base();
    return (b >= e) ? std::string{} : std::string(b, e);
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> out; std::string cur;
    for (char c : text) {
        if (c == delimiter) { out.push_back(cur); cur.clear(); } else { cur.push_back(c); }
    }
    out.push_back(cur); return out;
}

std::string join(const std::vector<std::string>& items, const std::string& delimiter) {
    std::ostringstream ss;
    for (std::size_t i = 0; i < items.size(); ++i) { if (i) ss << delimiter; ss << items[i]; }
    return ss.str();
}

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string timestamp_string() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    std::ostringstream ss; ss << std::put_time(&tm, "%Y%m%d-%H%M%S"); return ss.str();
}

void ensure_directory(const std::filesystem::path& path) { std::filesystem::create_directories(path); }

}  // namespace gpt2cpp
