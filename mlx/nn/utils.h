// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/core/array.h"
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

namespace mlx::nn {

namespace utils {

using namespace mlx::core;

inline void clip_grad_norm(std::map<std::string, Array>& parameters, float max_norm) {
    float total_norm = 0.0f;
    
    // 1. Calculate total norm
    for (auto& [name, p] : parameters) {
        auto grad = p.grad();
        if (!grad) continue;
        
        const float* g_data = grad->data();
        for (size_t i = 0; i < grad->size(); ++i) {
            float g = g_data[i];
            if (!std::isnan(g) && !std::isinf(g)) {
                total_norm += g * g;
            }
        }
    }
    total_norm = std::sqrt(total_norm);
    
    // 2. Clip if necessary
    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        // printf("DEBUG: Clipping gradients. Total norm: %f, Scale: %f\n", total_norm, scale);
        
        for (auto& [name, p] : parameters) {
            auto grad = p.grad();
            if (!grad) continue;
            
            float* g_data = grad->data();
            for (size_t i = 0; i < grad->size(); ++i) {
                g_data[i] *= scale;
            }
        }
    }
}

// Helper to count parameters
inline size_t count_parameters(std::map<std::string, Array>& parameters) {
    size_t count = 0;
    for (auto& [name, p] : parameters) {
        count += p.size();
    }
    return count;
}

} // namespace utils
} // namespace mlx::nn
