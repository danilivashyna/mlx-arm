// Test for Object-Oriented LLaMA implementation
#include "mlx/nn/llama.h"
#include "mlx/core/array.h"
#include <iostream>
#include <vector>

using namespace mlx::core;
using namespace mlx::nn;

int main() {
    std::cout << "🏗️  Initializing LLaMA Model (OO-style)..." << std::endl;

    // 1. Setup Config (TinyLlama parameters)
    LlamaConfig config;
    config.vocab_size = 1000;      // Reduced for test
    config.hidden_size = 64;       // Reduced for test
    config.intermediate_size = 128;// Reduced for test
    config.num_hidden_layers = 2;  // Reduced for test
    config.num_attention_heads = 4;
    config.num_key_value_heads = 4;

    // 2. Instantiate Model
    Model model(config);
    std::cout << "✅ Model created successfully." << std::endl;

    // 3. Check parameters
    auto params = model.parameters();
    std::cout << "📊 Parameter count check:" << std::endl;
    for (const auto& [name, arr] : params) {
        // Just print a few to verify registration works
        if (name.find("layers.0.self_attn.q_proj.weight") != std::string::npos) {
            std::cout << "   Found: " << name << " Shape: [";
            for (int d : arr.shape()) std::cout << d << " ";
            std::cout << "]" << std::endl;
        }
    }

    // 4. Dummy Input (Batch=1, Seq=10)
    std::cout << "🔄 Running dummy forward pass..." << std::endl;
    std::vector<int> input_data(10, 1); // 10 tokens with ID 1
    // Shape needs to be explicitly handled in Array creation for now if passing vector directly
    // Our current Array(vector) constructor is 1D. Let's force it.
    Array x(std::vector<float>(input_data.begin(), input_data.end()), {1, 10});

    // 5. Forward Pass
    try {
        Array logits = model(x);
        std::cout << "✅ Forward pass complete!" << std::endl;
        std::cout << "   Output Shape: [";
        for (int d : logits.shape()) std::cout << d << " ";
        std::cout << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Forward pass failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
