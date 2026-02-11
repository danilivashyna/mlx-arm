// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/core/array.h"
#include <map>
#include <string>
#include <memory>
#include <vector>

namespace mlx::nn {

using namespace mlx::core;

class Module {
public:
    virtual ~Module() = default;
    virtual Array operator()(const Array& x) { return x; }

    void register_parameter(const std::string& name, const Array& array) {
        parameters_[name] = std::make_shared<Array>(array);
    }

    void register_module(const std::string& name, std::shared_ptr<Module> module) {
        submodules_[name] = module;
    }

    virtual std::map<std::string, Array*> parameters() {
        std::map<std::string, Array*> all_params;
        for (auto& [name, p] : parameters_) {
            all_params[name] = p.get();
        }
        for (auto& [name, m] : submodules_) {
            auto sub_params = m->parameters();
            for (auto& [sub_name, p] : sub_params) {
                all_params[name + "." + sub_name] = p;
            }
        }
        return all_params;
    }

    void zero_grad() {
        auto params = parameters();
        for (auto& [name, p] : params) {
            p->zero_grad();
        }
    }

    void load_weights(const std::map<std::string, Array>& weights) {
        auto params = parameters();
        for (auto& [name, p] : params) {
            if (weights.count(name)) {
                auto& w = weights.at(name);
                if (p->size() == w.size()) {
                    std::memcpy(p->data(), w.data(), p->size() * sizeof(float));
                } else {
                    printf("WARNING: Size mismatch for %s: model=%zu, file=%zu\n", name.c_str(), p->size(), w.size());
                }
            }
        }
    }

protected:
    std::map<std::string, std::shared_ptr<Array>> parameters_;
    std::map<std::string, std::shared_ptr<Module>> submodules_;
};

} // namespace mlx::nn