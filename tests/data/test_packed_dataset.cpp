#include "gpt2cpp/data/dataset.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
int main() {
    const std::filesystem::path p = "test_dataset.bin";
    gpt2cpp::PackedDataset ds({1,2,3,4,5,6});
    ds.save(p);
    auto loaded = gpt2cpp::PackedDataset::load(p);
    assert(loaded.size() == 6);
    auto info = gpt2cpp::PackedDataset::inspect(p);
    assert(info.token_count == 6);
    std::filesystem::remove(p);
    std::cout << "packed dataset ok\n";
    return 0;
}
