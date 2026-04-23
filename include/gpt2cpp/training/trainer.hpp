#ifndef GPT2CPP_TRAINING_TRAINER_HPP
#define GPT2CPP_TRAINING_TRAINER_HPP
#include "gpt2cpp/core/config.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/data/dataset.hpp"
#include "gpt2cpp/model/gpt2_model.hpp"
namespace gpt2cpp {
class Trainer {
public:
    explicit Trainer(AppConfig config);
    void run();
    void evaluate_checkpoint(const std::filesystem::path& checkpoint_path);
private:
    double compute_learning_rate(int step) const;
    double validate(GPT2Model& model, const PackedDataset& ds, const torch::Device& device);
    std::filesystem::path create_run_dir() const;
    AppConfig config_;
};
}  // namespace gpt2cpp
#endif
#endif
