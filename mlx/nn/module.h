// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include <map>
#include <memory>
#include "mlx/core/array.h"

namespace mlx::nn {

using namespace mlx::core;

/**
 * Base class for all neural network modules.
 * Mimics PyTorch/MLX nn.Module structure.
 */
class Module {
public:
    virtual ~Module() = default;

    // Forward pass (must be implemented by subclasses if they have a simple input/output)
    virtual Array operator()(const Array& x) {
        throw std::runtime_error("Not implemented");
    }

    // Parameter management
    void register_parameter(const std::string& name, const Array& param) {
        parameters_[name] = new Array(param);
    }
    
    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        submodules_[name] = module;
    }

    // Recursive parameter traversal
    std::map<std::string, Array*> parameters() {
        std::map<std::string, Array*> all_params = parameters_;
        for (auto& [name, module] : submodules_) {
            auto sub_params = module->parameters();
            for (auto& [sub_name, param] : sub_params) {
                all_params[name + "." + sub_name] = param;
            }
        }
        return all_params;
    }
    
    // Load weights from a flat map (e.g. from safetensors)
    // Keys in weights map are like "model.layers.0.self_attn.q_proj.weight"
    // Our prefix might be empty or specific path
    virtual void update(const std::map<std::string, Array>& weights, const std::string& prefix = "") {
        // 1. Update own parameters
        for (auto& [name, param] : parameters_) {
            std::string key = prefix.empty() ? name : prefix + "." + name;
            // Common suffixes for weights
            if (weights.count(key + ".weight")) {
                 *param = weights.at(key + ".weight");
            } else if (weights.count(key)) {
                 *param = weights.at(key);
            }
        }
        
        // 2. Delegate to submodules
        for (auto& [name, module] : submodules_) {
            std::string new_prefix = prefix.empty() ? name : prefix + "." + name;
            module->update(weights, new_prefix);
        }
    }

protected:
    std::map<std::string, Array*> parameters_;
    std::map<std::string, std::shared_ptr<Module>> submodules_;
};

} // namespace mlx::nn
