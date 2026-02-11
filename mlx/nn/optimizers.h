// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/core/array.h"
#include <unordered_map>
#include <cmath>

namespace mlx::nn {

using namespace mlx::core;

class AdamW {
public:
    AdamW(float lr = 1e-3, float beta1 = 0.9, float beta2 = 0.999, float eps = 1e-8, float weight_decay = 0.01)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), wd_(weight_decay), t_(0) {}

    void step(std::map<std::string, std::shared_ptr<Array>>& params) {
        t_++;
        for (auto& [name, p] : params) {
            if (!p->grad) continue;

            // Initialize moments if needed
            if (m_.find(name) == m_.end()) {
                m_[name] = std::vector<float>(p->size(), 0.0f);
                v_[name] = std::vector<float>(p->size(), 0.0f);
            }

            float* data = p->data<float>();
            float* grad = p->grad->data<float>();
            auto& m = m_[name];
            auto& v = v_[name];

            for (size_t i = 0; i < p->size(); ++i) {
                // Weight decay
                data[i] -= lr_ * wd_ * data[i];

                // Adam update
                m[i] = beta1_ * m[i] + (1 - beta1_) * grad[i];
                v[i] = beta2_ * v[i] + (1 - beta2_) * grad[i] * grad[i];

                float m_hat = m[i] / (1 - std::pow(beta1_, t_));
                float v_hat = v[i] / (1 - std::pow(beta2_, t_));

                data[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }

private:
    float lr_, beta1_, beta2_, eps_, wd_;
    int t_;
    std::unordered_map<std::string, std::vector<float>> m_, v_;
};

} // namespace mlx::nn
