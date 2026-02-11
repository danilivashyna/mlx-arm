// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/nn/module.h"
#include "mlx/core/ops.h"
#include <cmath>
#include <cstring>

namespace mlx::nn {

using namespace mlx::core;

/**
 * Linear Layer (y = xW + b)
 */
class Linear : public Module {
public:
    Linear(int input_dims, int output_dims, bool bias = true) {
        register_parameter("weight", Array({output_dims, input_dims})); 
        if (bias) {
            register_parameter("bias", Array({output_dims}));
        }
    }

    Array operator()(const Array& x) override {
        // x: [..., in_dims], weight: [out_dims, in_dims]
        // matmul expects [M, K] and [K, N]. 
        // If x is [M, in_dims] and weight is [out_dims, in_dims], we need weight.T [in_dims, out_dims]
        auto out = matmul(x, transpose(*(parameters_["weight"]), {1, 0}));
        if (parameters_.count("bias")) {
            out = add(out, *(parameters_["bias"]));
        }
        return out;
    }
};

/**
 * RMSNorm Layer
 */
class RMSNorm : public Module {
public:
    RMSNorm(int dims, float eps = 1e-5) : eps_(eps) {
        register_parameter("weight", Array({dims}));
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
        register_parameter("weight", Array({num_embeddings, dims}));
    }

    Array operator()(const Array& x) override {
        const float* weight_ptr = parameters_["weight"]->data<float>();
        int dims = parameters_["weight"]->shape()[1];
        
        // Assuming x contains int32 indices
        std::vector<float> res_data(x.size() * dims);
        // We need to handle different input types, but for now assume float cast to int
        const float* indices = x.data<float>();
        
        for (size_t i = 0; i < x.size(); ++i) {
            int idx = static_cast<int>(indices[i]);
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