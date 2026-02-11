// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/nn/module.h"
#include "mlx/core/ops.h"
#include <cmath>
#include <cstring>
#include <cstdlib>

namespace mlx::nn {

using namespace mlx::core;

/**
 * Linear Layer (y = xW + b)
 */
class Linear : public Module {
public:
    Linear(int input_dims, int output_dims, bool bias = true) {
        Array weight({output_dims, input_dims});
        float scale = std::sqrt(1.0f / input_dims);
        float* w_ptr = weight.data();
        for (size_t i = 0; i < weight.size(); ++i) {
            w_ptr[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f) * scale;
        }
        register_parameter("weight", weight); 
        
        if (bias) {
            Array b({output_dims});
            register_parameter("bias", b);
        }
    }

    Array operator()(const Array& x) override {
        auto out = matmul(x, transpose(*(parameters_["weight"]), {1, 0}));
        if (parameters_.count("bias")) {
            out = add(out, *(parameters_["bias"]));
        }
        return out;
    }
};

/**
 * LoRA Linear Layer
 */
class LoRALinear : public Linear {
public:
    LoRALinear(int input_dims, int output_dims, int r = 8, float lora_alpha = 16.0f, bool bias = true)
        : Linear(input_dims, output_dims, bias), r_(r), alpha_(lora_alpha) {
        
        float scale = lora_alpha / r;
        scale_ = scale;

        // LoRA matrices A and B
        // A is [r, input_dims], B is [output_dims, r]
        Array lora_a({r, input_dims});
        Array lora_b({output_dims, r});
        
        // Initialize A and B with small values
        float* a_ptr = lora_a.data();
        float* b_ptr = lora_b.data();
        for (size_t i = 0; i < lora_a.size(); ++i) a_ptr[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f) * 0.01f;
        for (size_t i = 0; i < lora_b.size(); ++i) b_ptr[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f) * 0.01f;

        register_parameter("lora_a", lora_a);
        register_parameter("lora_b", lora_b);
    }

    Array operator()(const Array& x) override {
        // Standard linear output
        auto out = Linear::operator()(x);

        if (fused_) return out;

        // LoRA path: (x @ A.T) @ B.T * scale
        auto lora_a = *(parameters_["lora_a"]);
        auto lora_b = *(parameters_["lora_b"]);

        auto x_a = matmul(x, transpose(lora_a, {1, 0}));
        auto x_ab = matmul(x_a, transpose(lora_b, {1, 0}));
        
        auto res = add(out, multiply(x_ab, scale_));
        return res;
    }

    void fuse() {
        auto lora_a = *(parameters_["lora_a"]);
        auto lora_b = *(parameters_["lora_b"]);
        auto weight = *(parameters_["weight"]);

        // W_new = W + (B @ A) * scale
        // Wait, our LoRA path was: (x @ A.T) @ B.T * scale
        // Which is x @ (A.T @ B.T) * scale
        // So the weight update is delta_W = (A.T @ B.T).T * scale = (B @ A) * scale
        // A is [r, in], B is [out, r]. B @ A is [out, in].
        
        auto delta_w = multiply(matmul(lora_b, lora_a), scale_);
        auto fused_w = add(weight, delta_w);
        
        // Update base weight and clear LoRA
        register_parameter("weight", fused_w);
        parameters_.erase("lora_a");
        parameters_.erase("lora_b");
        
        fused_ = true;
    }

    bool is_fused() const { return fused_; }

private:
    int r_;
    float alpha_;
    float scale_;
    bool fused_ = false;
};

/**
 * RMSNorm Layer
 */
class RMSNorm : public Module {
public:
    RMSNorm(int dims, float eps = 1e-5) : eps_(eps) {
        Array weight({dims});
        float* w_ptr = weight.data();
        for (size_t i = 0; i < weight.size(); ++i) w_ptr[i] = 1.0f;
        register_parameter("weight", weight);
    }

    Array operator()(const Array& x) override {
        return rms_norm(x, *(parameters_["weight"]), eps_);
    }

private:
    float eps_;
};

/**
 * Embedding Layer
 */
class Embedding : public Module {
public:
    Embedding(int num_embeddings, int dims) {
        Array weight({num_embeddings, dims});
        float* w_ptr = weight.data();
        for (size_t i = 0; i < weight.size(); ++i) {
            w_ptr[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f) * 0.02f;
        }
        register_parameter("weight", weight);
    }

    Array operator()(const Array& x) override {
        const float* weight_ptr = parameters_["weight"]->data();
        int dims = parameters_["weight"]->shape()[1];
        
        // Assuming x contains int32 indices
        std::vector<float> res_data(x.size() * dims);
        const float* indices = x.data();
        
        for (size_t i = 0; i < x.size(); ++i) {
            int idx = static_cast<int>(indices[i]);
            if (idx < 0) idx = 0;
            if (idx >= parameters_["weight"]->shape()[0]) idx = parameters_["weight"]->shape()[0] - 1;
            std::memcpy(res_data.data() + i * dims, weight_ptr + idx * dims, dims * sizeof(float));
        }
        
        std::vector<int> out_shape = x.shape();
        out_shape.push_back(dims);
        return Array(res_data, out_shape);
    }
};

/**
 * SiLU Activation
 */
class SiLU : public Module {
public:
    Array operator()(const Array& x) override {
        return silu(x);
    }
};

} // namespace mlx::nn