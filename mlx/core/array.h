// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <vector>

namespace mlx::core {

/**
 * Minimal array stub for compilation
 * Full implementation coming in next iteration
 */
class Array {
public:
    Array() = default;
    explicit Array(std::vector<float> data) : data_(std::move(data)) {}
    
    size_t size() const { return data_.size(); }
    const float* data() const { return data_.data(); }
    float* data() { return data_.data(); }
    
private:
    std::vector<float> data_;
};

/**
 * Stream stub for async execution
 */
class Stream {
public:
    Stream() = default;
};

}  // namespace mlx::core
