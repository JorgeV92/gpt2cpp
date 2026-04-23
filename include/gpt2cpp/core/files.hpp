#ifndef GPT2CPP_CORE_FILES_HPP
#define GPT2CPP_CORE_FILES_HPP
#include <filesystem>
#include <string>
#include <vector>
namespace gpt2cpp {
std::string read_text_file(const std::filesystem::path& path);
void write_text_file(const std::filesystem::path& path, const std::string& content);
std::vector<std::filesystem::path> list_files_recursive(const std::filesystem::path& root);
std::string trim(const std::string& value);
std::vector<std::string> split(const std::string& text, char delimiter);
std::string join(const std::vector<std::string>& items, const std::string& delimiter);
std::string to_lower(std::string value);
std::string timestamp_string();
void ensure_directory(const std::filesystem::path& path);
}  // namespace gpt2cpp
#endif
