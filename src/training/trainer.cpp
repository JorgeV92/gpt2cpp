#include "gpt2cpp/training/trainer.hpp"
#ifdef GPT2CPP_HAVE_TORCH
#include "gpt2cpp/core/device.hpp"
#include "gpt2cpp/core/files.hpp"
#include "gpt2cpp/core/logging.hpp"
#include "gpt2cpp/inference/generator.hpp"
#include "gpt2cpp/tokenizer/bpe_tokenizer.hpp"
#include <chrono>
#include <cmath>
#include <fstream>

namespace gpt2cpp {

Trainer::Trainer(AppConfig config) : config_(std::move(config)) {}

std::filesystem::path Trainer::create_run_dir() const {
    const auto dir = std::filesystem::path(config_.run.output_root) / (timestamp_string() + "-" + config_.run.name);
    ensure_directory(dir / "logs"); ensure_directory(dir / "checkpoints"); ensure_directory(dir / "samples"); ensure_directory(dir / "model_bundle" / "tokenizer");
    return dir;
}

double Trainer::compute_learning_rate(int step) const {
    if (config_.training.warmup_steps > 0 && step < config_.training.warmup_steps)
        return config_.training.learning_rate * (static_cast<double>(step + 1) / static_cast<double>(config_.training.warmup_steps));
    const auto decay_steps = std::max(1, config_.training.max_steps - config_.training.warmup_steps);
    const auto progress = static_cast<double>(std::max(0, step - config_.training.warmup_steps)) / static_cast<double>(decay_steps);
    return config_.training.learning_rate * (0.5 * (1.0 + std::cos(3.14159265358979323846 * std::min(1.0, progress))));
}

double Trainer::validate(GPT2Model& model, const PackedDataset& ds, const torch::Device& device) {
    model->eval(); torch::NoGradGuard ng;
    double loss_sum = 0.0;
    for (int i = 0; i < std::max(1, config_.training.eval_batches); ++i) {
        auto [x, y] = ds.sample_batch(config_.data.batch_size, config_.data.sequence_length, static_cast<std::uint64_t>(config_.run.seed + 1000 + i), device);
        loss_sum += model->forward(x, y).loss.item<double>();
    }
    model->train();
    return loss_sum / static_cast<double>(std::max(1, config_.training.eval_batches));
}

void Trainer::run() {
    const auto run_dir = create_run_dir();
    save_config(config_, run_dir / "config.snapshot.toml");
    Logger::instance().set_log_file(run_dir / "logs" / "train.log");

    BPETokenizer tokenizer = BPETokenizer::load(config_.tokenizer.dir);
    AppConfig cfg = config_; cfg.model.vocab_size = tokenizer.vocab_size();
    const auto device = to_torch_device(parse_device_kind(cfg.run.device));

    PackedDataset train_ds = PackedDataset::load(cfg.data.train_bin);
    PackedDataset val_ds = PackedDataset::load(cfg.data.val_bin);
    GPT2Model model(cfg.model); model->to(device);

    torch::optim::AdamW optim(model->parameters(), torch::optim::AdamWOptions(cfg.training.learning_rate).weight_decay(cfg.training.weight_decay));
    std::ofstream metrics(run_dir / "metrics.csv"); metrics << "step,train_loss,val_loss,lr,tokens_per_sec,step_time_ms\n";

    for (int step = 0; step < cfg.training.max_steps; ++step) {
        const auto t0 = std::chrono::steady_clock::now();
        optim.zero_grad();

        auto [x, y] = train_ds.sample_batch(cfg.data.batch_size, cfg.data.sequence_length, static_cast<std::uint64_t>(cfg.run.seed + step), device);
        auto out = model->forward(x, y);
        out.loss.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.training.grad_clip);

        const auto lr = compute_learning_rate(step);
        for (auto& group : optim.param_groups()) static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
        optim.step();

        const auto t1 = std::chrono::steady_clock::now();
        const auto ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
        const auto tokens_per_sec = ms > 0.0 ? (1000.0 * cfg.data.batch_size * cfg.data.sequence_length / ms) : 0.0;

        double val_loss = -1.0;
        if ((step + 1) % cfg.training.eval_interval == 0 || step + 1 == cfg.training.max_steps) val_loss = validate(model, val_ds, device);
        if ((step + 1) % cfg.training.log_interval == 0 || step == 0) GPT2CPP_LOG_INFO("step=" + std::to_string(step + 1) + " train_loss=" + std::to_string(out.loss.item<double>()));

        metrics << (step + 1) << ',' << out.loss.item<double>() << ',' << val_loss << ',' << lr << ',' << tokens_per_sec << ',' << ms << '\n';

        if ((step + 1) % cfg.training.save_interval == 0 || step + 1 == cfg.training.max_steps) {
            const auto ckpt = run_dir / "checkpoints" / ("step_" + std::to_string(step + 1) + ".pt");
            torch::save(model, ckpt.string());
            write_text_file(run_dir / "latest_checkpoint.txt", ckpt.string());

            const auto bundle = run_dir / "model_bundle";
            torch::save(model, (bundle / "model.pt").string());
            tokenizer.save(bundle / "tokenizer");
            save_config(cfg, bundle / "config.snapshot.toml");
            write_text_file(bundle / "manifest.txt", "name=" + cfg.run.name + "\nvocab_size=" + std::to_string(cfg.model.vocab_size) + "\n");
        }
    }
}

void Trainer::evaluate_checkpoint(const std::filesystem::path& checkpoint_path) {
    AppConfig cfg = config_;
    const auto device = to_torch_device(parse_device_kind(cfg.run.device));
    PackedDataset val_ds = PackedDataset::load(cfg.data.val_bin);
    GPT2Model model(cfg.model); torch::load(model, checkpoint_path.string()); model->to(device);
    const auto loss = validate(model, val_ds, device);
    std::cout << "checkpoint=" << checkpoint_path.string() << "\nval_loss=" << loss << "\nperplexity=" << std::exp(loss) << "\n";
}

}  // namespace gpt2cpp
#endif
