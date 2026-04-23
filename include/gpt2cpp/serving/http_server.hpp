#ifndef GPT2CPP_SERVING_HTTP_SERVER_HPP
#define GPT2CPP_SERVING_HTTP_SERVER_HPP
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/inference/generator.hpp"
#include <memory>
namespace gpt2cpp {
class HttpServer {
public:
    HttpServer(std::shared_ptr<Generator> generator, std::string host, int port, std::filesystem::path web_root);
    void serve_forever();
private:
    std::string handle_request(const std::string& request) const;
    std::string response(int code, const std::string& type, const std::string& body) const;
    std::shared_ptr<Generator> generator_;
    std::string host_;
    int port_;
    std::filesystem::path web_root_;
};
}  // namespace gpt2cpp
#endif
#endif
