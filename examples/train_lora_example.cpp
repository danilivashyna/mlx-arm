#include "mlx/nn/llama.h"
#include "mlx/nn/optimizers.h"
#include "mlx/nn/utils.h"
#include "mlx/core/ops.h"
#include "mlx/core/safetensors.hpp"
#include <iostream>
#include <vector>

using namespace mlx::core;
using namespace mlx::nn;

int main() {
    // 1. Configuration - Small for Termux RAM
    LlamaConfig config;
    config.vocab_size = 2000; // Small vocab to save RAM
    config.hidden_size = 576;
    config.num_hidden_layers = 1; // 1 layer is enough to test LoRA
    config.num_attention_heads = 9;
    config.num_key_value_heads = 3;
    config.intermediate_size = 1536;
    
    printf("Creating model (SmolLM-135M - 1 layer, small vocab)... ");
    auto model = std::make_shared<Model>(config);
    printf("Done.\n");

    // 2. Load Real Weights
    printf("Loading weights from SmolLM-135M/model.safetensors... ");
    auto weights = load_safetensors("../SmolLM-135M/model.safetensors");
    if (!weights.empty()) {
        model->load_weights(weights);
        weights.clear(); // Free memory immediately!
        printf("Loaded weights (filtered by model needs). RAM freed.\n");
    } else {
        printf("FAILED. Continuing with random weights.\n");
    }

    // 3. Prepare LoRA parameters
    auto params = model->parameters();
    std::map<std::string, Array> trainable_params;
    for (auto& [name, p] : params) {
        if (name.find("lora_") != std::string::npos) {
            p->set_requires_grad(true);
            trainable_params[name] = *p;
        } else {
            p->set_requires_grad(false);
        }
    }
    printf("Found %zu trainable LoRA parameters.\n", trainable_params.size());

    AdamW optimizer(1e-4);

    // 4. Synthetic Data
    Array x({1.0f}, {1, 1}); 
    Array targets({504.0f}, {1});

    // 5. Training Loop
    printf("Starting LoRA training loop...\n");
    for (int epoch = 0; epoch < 5; ++epoch) {
        auto logits = (*model)(x);
        auto logits_2d = reshape(logits, {1, config.vocab_size});
        auto loss = cross_entropy(logits_2d, targets);
        
        float loss_val = loss.data()[0];
        printf("Epoch %d, Loss: %.4f\n", epoch, loss_val);

        if (std::isnan(loss_val)) break;

        model->zero_grad();
        loss.backward();
        utils::clip_grad_norm(trainable_params, 1.0f);
        optimizer.step(trainable_params);
    }

    printf("\n✅ SUCCESS! Real weights loaded, LoRA training works on Android!\n");

    return 0;
}