#include "gpt2cpp/core/device.hpp"
#include "gpt2cpp/core/error.hpp"
#include "gpt2cpp/core/files.hpp"
namespace gpt2cpp {
DeviceKind parse_device_kind(const std::string& text) {
    const std::string t = to_lower(text);
    if (t == "cpu") return DeviceKind::Cpu;
    if (t == "cuda" || t == "gpu") return DeviceKind::Cuda;
    throw Error("Unknown device kind: " + text);
}
std::string to_string(DeviceKind kind) { return kind == DeviceKind::Cpu ? "cpu" : "cuda"; }
#ifdef GPT2CPP_HAVE_TORCH
torch::Device to_torch_device(DeviceKind kind) {
    if (kind == DeviceKind::Cuda) {
        if (!torch::cuda::is_available()) throw Error("CUDA requested but torch::cuda::is_available() is false.");
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}
#endif
}  // namespace gpt2cpp
