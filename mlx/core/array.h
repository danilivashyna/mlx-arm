// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <algorithm>
#include <iostream>

#include "mlx/core/dtype.h"

namespace mlx::core {

class Array {
public:
    // Constructors
    Array() = default;
    
    // Create from existing data (copy)
    Array(const std::vector<float>& data, const std::vector<int>& shape) 
        : shape_(shape), dtype_(Dtype::Float32) {
        compute_strides();
        size_t size = data.size();
        size_t expected = this->size();
        if (size != expected) {
            std::cerr << "Shape mismatch: expected " << expected << ", got " << size << std::endl;
        }
        // Allocate and copy
        size_t nbytes = size * sizeof(float);
        data_ = std::shared_ptr<void>(::operator new(nbytes), [](void* p) { ::operator delete(p); });
        std::copy(data.begin(), data.end(), static_cast<float*>(data_.get()));
    }

    // Create empty with shape
    Array(const std::vector<int>& shape, Dtype dtype = Dtype::Float32)
        : shape_(shape), dtype_(dtype) {
        compute_strides();
        size_t nbytes = this->size() * size_of(dtype);
        data_ = std::shared_ptr<void>(::operator new(nbytes), [](void* p) { ::operator delete(p); });
    }

    // Create from shared buffer (View/Reshape)
    Array(std::shared_ptr<void> data, const std::vector<int>& shape, Dtype dtype)
        : shape_(shape), dtype_(dtype), data_(data) {
        compute_strides();
    }

    // Basic accessors
    const std::vector<int>& shape() const { return shape_; }
    int ndim() const { return shape_.size(); }
    size_t size() const {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    }
    Dtype dtype() const { return dtype_; }
    
    // Raw pointer access
    template<typename T = float>
    T* data() { return static_cast<T*>(data_.get()); }
    
    template<typename T = float>
    const T* data() const { return static_cast<const T*>(data_.get()); }

    // Shared pointer access for views/reshapes
    std::shared_ptr<void> data_shared() const { return data_; }

    // Shape manipulation
    void reshape(const std::vector<int>& new_shape) {
        // Simple reshape check
        size_t current_size = size();
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
        if (current_size != new_size) {
            throw std::runtime_error("Reshape size mismatch");
        }
        shape_ = new_shape;
        compute_strides();
    }

private:
    void compute_strides() {
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    std::vector<int> shape_;
    std::vector<size_t> strides_;
    Dtype dtype_;
    std::shared_ptr<void> data_; // Shared ownership of memory
};

} // namespace mlx::core