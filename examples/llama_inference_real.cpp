#include "mlx/nn/llama.h"
#include "mlx/core/ops.h"
#include "mlx/core/tokenizer.hpp"
#include <iostream>
#include <chrono>
#include <fstream>
#include <map>
#include <stdint.h>

using namespace mlx::core;
using namespace mlx::nn;

// Minimal SafeTensors Loader to avoid include issues
struct SafeTensorHeader {
    std::string name;
    size_t start;
    size_t end;
    std::vector<int> shape;
};

bool load_smollm_weights(const std::string& path, std::shared_ptr<Model> model) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    
    std::string header_json(header_size, '\0');
    f.read(&header_json[0], header_size);
    
    auto params = model->parameters();
    int loaded = 0;

    // Very simple parser for Safetensors JSON
    for (auto& [name, array_ptr] : params) {
        size_t pos = header_json.find("\"" + name + "\"");
        if (pos != std::string::npos) {
            size_t data_off_pos = header_json.find("\"data_offsets\"", pos);
            size_t start_pos = header_json.find("[", data_off_pos) + 1;
            size_t end_pos = header_json.find(",", start_pos);
            size_t start_off = std::stoull(header_json.substr(start_pos, end_pos - start_pos));
            
            f.seekg(8 + header_size + start_off);
            f.read(reinterpret_cast<char*>(array_ptr->data()), array_ptr->size() * sizeof(float));
            loaded++;
        }
    }

    // Tie embeddings if needed
    if (params.count("lm_head.weight")) {
        auto embed = params.at("model.embed_tokens.weight");
        auto head = params.at("lm_head.weight");
        std::memcpy(head->data(), embed->data(), embed->size() * sizeof(float));
    }

    printf("✅ Loaded %d parameters from %s\n", loaded, path.c_str());
    return loaded > 0;
}

int main(int argc, char** argv) {
    std::string model_dir = "SmolLM-135M";
    std::string model_path = model_dir + "/model.safetensors";
    
    LlamaConfig config;
    config.vocab_size = 49152;
    config.hidden_size = 576;
    config.intermediate_size = 1536;
    config.num_hidden_layers = 30;
    config.num_attention_heads = 9;
    config.num_key_value_heads = 3;
    config.rms_norm_eps = 1e-5;
    
    auto model = std::make_shared<Model>(config);
    if (!load_smollm_weights(model_path, model)) {
        printf("❌ Failed to load weights!\n");
        return 1;
    }
    
    mlx::tokenizer::LlamaTokenizer tokenizer;
    tokenizer.load(model_dir + "/vocab.txt");
    std::vector<int> tokens = tokenizer.encode("The capital of France is");
    
    printf("\n📝 Prompt: 'The capital of France is' (%zu tokens)\n", tokens.size());
    printf("\nGenerating with KV-Cache...\n");

    int max_tokens = 30;
    int current_pos = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 1. Process Prompt
    std::vector<float> input_data;
    for(int t : tokens) input_data.push_back(static_cast<float>(t));
    auto logits = (*model)(Array(input_data, {1, (int)tokens.size()}), 0);
    current_pos = tokens.size();

    // 2. Generate
    for (int i = 0; i < max_tokens; i++) {
        float* logits_data = logits.data();
        int last_token_idx = (logits.shape()[logits.ndim()-2] - 1) * config.vocab_size;
        
        int next_token = 0;
        float max_val = -1e9f;
        for (int v = 0; v < config.vocab_size; v++) {
            if (logits_data[last_token_idx + v] > max_val) {
                max_val = logits_data[last_token_idx + v];
                next_token = v;
            }
        }

        std::string word = tokenizer.decode({next_token});
        printf("%s", word.c_str());
        fflush(stdout);
        
        if (next_token == 0 || next_token == 2) break;

        // KV-Cache Step: Feed only one token
        logits = (*model)(Array({(float)next_token}, {1, 1}), current_pos);
        current_pos++;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();
    printf("\n\n⚡ Speed: %.2f tokens/sec\n", max_tokens / duration);
    
    return 0;
}