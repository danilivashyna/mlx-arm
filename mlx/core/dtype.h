// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace mlx::core {

enum class Dtype {
    Float32,
    Float16,
    Int32,
    Int8,
    Bool
};

inline size_t size_of(Dtype t) {
    switch (t) {
        case Dtype::Float32: return 4;
        case Dtype::Float16: return 2;
        case Dtype::Int32:   return 4;
        case Dtype::Int8:    return 1;
        case Dtype::Bool:    return 1;
        default: return 1;
    }
}

inline std::string dtype_to_string(Dtype t) {
    switch (t) {
        case Dtype::Float32: return "float32";
        case Dtype::Float16: return "float16";
        case Dtype::Int32:   return "int32";
        case Dtype::Int8:    return "int8";
        case Dtype::Bool:    return "bool";
        default: return "unknown";
    }
}

} // namespace mlx::core
