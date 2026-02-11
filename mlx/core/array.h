// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <functional>
#include <cstring>

#include "mlx/core/dtype.h"

namespace mlx::core {

class ArrayImpl;

class Array {
public:
    Array();
    Array(const std::vector<float>& data, const std::vector<int>& shape);
    Array(const std::vector<int>& shape, Dtype dtype = Dtype::Float32);
    Array(std::shared_ptr<std::vector<float>> data, const std::vector<int>& shape, Dtype dtype);
    Array(std::shared_ptr<ArrayImpl> impl) : impl_(impl) {}

    const std::vector<int>& shape() const;
    int ndim() const;
    size_t size() const;
    Dtype dtype() const;
    
    float* data();
    const float* data() const;
    std::shared_ptr<std::vector<float>> data_shared() const;

    std::shared_ptr<Array> grad() const;
    void set_grad(Array g);
    
    bool requires_grad() const;
    void set_requires_grad(bool r);
    
    void backward();
    void zero_grad();

    std::shared_ptr<ArrayImpl> impl() const { return impl_; }
    bool is_null() const { return !impl_; }

private:
    std::shared_ptr<ArrayImpl> impl_;
};

class ArrayImpl : public std::enable_shared_from_this<ArrayImpl> {
public:
    ArrayImpl(const std::vector<float>& data, const std::vector<int>& shape);
    ArrayImpl(const std::vector<int>& shape, Dtype dtype);
    
    // For sharing data, we need a pointer. But keeping vector alive is tricky with sharing.
    // Let's stick to shared_ptr but use float* for safety.
    ArrayImpl(std::shared_ptr<std::vector<float>> data, const std::vector<int>& shape, Dtype dtype);
    
    std::vector<int> shape;
    Dtype dtype;
    std::shared_ptr<std::vector<float>> data; // Changed from void* to vector
    
    std::shared_ptr<ArrayImpl> grad;
    std::vector<Array> inputs;
    std::function<void()> backward_op;
    bool requires_grad = false;

    size_t size() const {
        size_t s = 1;
        for (int d : shape) s *= d;
        return s;
    }
};

} // namespace mlx::core
