#ifndef GPT2CPP_CORE_CONFIG_HPP
#define GPT2CPP_CORE_CONFIG_HPP
#include <filesystem>
#include <string>
#include <vector>
namespace gpt2cpp {

struct RunSection { std::string name{"run"}; std::string output_root{"runs"}; std::string device{"cpu"}; int seed{1337}; bool mixed_precision{false}; };
struct TokenizerSection { std::string dir{"artifacts/tokenizer"}; };
struct DataSection { std::string train_bin{"artifacts/dataset/train.bin"}; std::string val_bin{"artifacts/dataset/val.bin"}; int batch_size{8}; int sequence_length{128}; std::string jsonl_text_field{"text"}; };
struct ModelSection { int vocab_size{512}; int max_position_embeddings{256}; int n_embd{128}; int n_head{4}; int n_layer{4}; double dropout{0.1}; bool tie_weights{true}; };
struct TrainingSection { int max_steps{1000}; double learning_rate{3e-4}; double weight_decay{0.01}; int warmup_steps{100}; double grad_clip{1.0}; int eval_interval{100}; int save_interval{100}; int log_interval{10}; int eval_batches{10}; std::string resume_from{}; };
struct InferenceSection { double temperature{0.8}; int top_k{40}; double top_p{0.95}; int max_new_tokens{128}; double repetition_penalty{1.0}; std::vector<std::string> stop_sequences{}; };
struct ServingSection { std::string host{"127.0.0.1"}; int port{8080}; std::string web_root{"web"}; };

struct AppConfig {
    RunSection run;
    TokenizerSection tokenizer;
    DataSection data;
    ModelSection model;
    TrainingSection training;
    InferenceSection inference;
    ServingSection serving;
};

AppConfig load_config(const std::filesystem::path& path);
void save_config(const AppConfig& config, const std::filesystem::path& path);
}  // namespace gpt2cpp
#endif
