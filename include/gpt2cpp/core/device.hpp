#ifndef GPT2CPP_CORE_DEVICE_HPP
#define GPT2CPP_CORE_DEVICE_HPP
#include <string>
#ifdef GPT2CPP_HAVE_TORCH
#include <torch/torch.h>
#endif
namespace gpt2cpp {
enum class DeviceKind { Cpu, Cuda };
DeviceKind parse_device_kind(const std::string& text);
std::string to_string(DeviceKind kind);
#ifdef GPT2CPP_HAVE_TORCH
torch::Device to_torch_device(DeviceKind kind);
#endif
}  // namespace gpt2cpp
#endif
