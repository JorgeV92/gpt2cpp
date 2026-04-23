#include "gpt2cpp/serving/http_server.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/core/error.hpp"
#include "gpt2cpp/core/files.hpp"
#include "gpt2cpp/core/logging.hpp"
#include <array>
#include <sstream>

#if defined(_WIN32)
#error "This demo server expects POSIX sockets."
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace gpt2cpp {

HttpServer::HttpServer(std::shared_ptr<Generator> generator, std::string host, int port, std::filesystem::path web_root)
    : generator_(std::move(generator)), host_(std::move(host)), port_(port), web_root_(std::move(web_root)) {}

std::string HttpServer::response(int code, const std::string& type, const std::string& body) const {
    std::ostringstream ss;
    ss << "HTTP/1.1 " << code << (code == 200 ? " OK" : " Not Found") << "\r\n";
    ss << "Content-Type: " << type << "\r\n";
    ss << "Content-Length: " << body.size() << "\r\n";
    ss << "Connection: close\r\n\r\n";
    ss << body;
    return ss.str();
}

std::string HttpServer::handle_request(const std::string& request) const {
    const auto eol = request.find("\r\n");
    if (eol == std::string::npos) return response(404, "text/plain", "Malformed request");
    const auto line = request.substr(0, eol);

    if (line.rfind("GET / ", 0) == 0) return response(200, "text/html", read_text_file(web_root_ / "index.html"));
    if (line.rfind("GET /styles.css ", 0) == 0) return response(200, "text/css", read_text_file(web_root_ / "styles.css"));
    if (line.rfind("GET /app.js ", 0) == 0) return response(200, "application/javascript", read_text_file(web_root_ / "app.js"));
    if (line.rfind("POST /chat ", 0) == 0) {
        const auto body_pos = request.find("\r\n\r\n");
        const auto body = body_pos == std::string::npos ? std::string{} : request.substr(body_pos + 4);
        GenerationOptions opt;
        opt.temperature = generator_->config().inference.temperature;
        opt.top_k = generator_->config().inference.top_k;
        opt.top_p = generator_->config().inference.top_p;
        opt.max_new_tokens = generator_->config().inference.max_new_tokens;
        opt.repetition_penalty = generator_->config().inference.repetition_penalty;
        opt.stop_sequences = generator_->config().inference.stop_sequences;
        return response(200, "text/plain", generator_->generate(body, opt));
    }
    return response(404, "text/plain", "Not found");
}

void HttpServer::serve_forever() {
    const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) throw Error("Failed to create socket.");

    int opt = 1;
    ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    addr.sin_addr.s_addr = inet_addr(host_.c_str());

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(server_fd);
        throw Error("Failed to bind socket.");
    }
    if (::listen(server_fd, 16) < 0) {
        ::close(server_fd);
        throw Error("Failed to listen on socket.");
    }

    GPT2CPP_LOG_INFO("Serving web UI at http://" + host_ + ":" + std::to_string(port_) + "/");

    while (true) {
        const int client_fd = ::accept(server_fd, nullptr, nullptr);
        if (client_fd < 0) continue;
        std::array<char, 1 << 16> buf{};
        const ssize_t n = ::read(client_fd, buf.data(), static_cast<int>(buf.size()));
        if (n > 0) {
            const std::string req(buf.data(), static_cast<std::size_t>(n));
            const auto res = handle_request(req);
            ::write(client_fd, res.data(), static_cast<int>(res.size()));
        }
        ::close(client_fd);
    }
}

}  // namespace gpt2cpp
#endif
