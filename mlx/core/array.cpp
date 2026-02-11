// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/array.h"
#include <stdexcept>

namespace mlx::core {

// --- ArrayImpl Implementation ---

ArrayImpl::ArrayImpl(const std::vector<float>& d, const std::vector<int>& s) 
    : shape(s), dtype(Dtype::Float32) {
    // size_t nbytes = size() * sizeof(float);
    data = std::make_shared<std::vector<float>>(d);
}

ArrayImpl::ArrayImpl(const std::vector<int>& s, Dtype dt)
    : shape(s), dtype(dt) {
    size_t sz = size();
    data = std::make_shared<std::vector<float>>(sz, 0.0f);
}

ArrayImpl::ArrayImpl(std::shared_ptr<std::vector<float>> d, const std::vector<int>& s, Dtype dt)
    : shape(s), dtype(dt), data(d) {}

// --- Array Implementation ---

Array::Array() : impl_(nullptr) {}

Array::Array(const std::vector<float>& data, const std::vector<int>& shape)
    : impl_(std::make_shared<ArrayImpl>(data, shape)) {}

Array::Array(const std::vector<int>& shape, Dtype dtype)
    : impl_(std::make_shared<ArrayImpl>(shape, dtype)) {}

Array::Array(std::shared_ptr<std::vector<float>> data, const std::vector<int>& shape, Dtype dtype)
    : impl_(std::make_shared<ArrayImpl>(data, shape, dtype)) {}

const std::vector<int>& Array::shape() const { return impl_->shape; }
int Array::ndim() const { return impl_->shape.size(); }
size_t Array::size() const { return impl_->size(); }
Dtype Array::dtype() const { return impl_->dtype; }

float* Array::data() { return impl_->data->data(); }
const float* Array::data() const { return impl_->data->data(); }
std::shared_ptr<std::vector<float>> Array::data_shared() const { return impl_->data; }

bool Array::requires_grad() const { return impl_->requires_grad; }
void Array::set_requires_grad(bool r) { impl_->requires_grad = r; }

std::shared_ptr<Array> Array::grad() const {
    if (!impl_->grad) return nullptr;
    return std::make_shared<Array>(impl_->grad);
}

void Array::set_grad(Array g) {
    impl_->grad = g.impl();
}

void Array::zero_grad() {
    if (!impl_) return;
    if (impl_->grad) {
        std::fill(impl_->grad->data->begin(), impl_->grad->data->end(), 0.0f);
    }
    for (auto& in : impl_->inputs) {
        in.zero_grad();
    }
}

void Array::backward() {
    if (!impl_) return;
    if (!impl_->grad) {
        Array g(shape(), dtype());
        std::fill(g.data(), g.data() + g.size(), 1.0f);
        impl_->grad = g.impl();
    }

    std::vector<std::shared_ptr<ArrayImpl>> topo;
    std::vector<std::shared_ptr<ArrayImpl>> visited;
    
    std::function<void(std::shared_ptr<ArrayImpl>)> build_topo = [&](std::shared_ptr<ArrayImpl> a) {
        bool already_visited = false;
        for(auto& v : visited) if(v == a) { already_visited = true; break; }
        if (!already_visited) {
            visited.push_back(a);
            for (auto& input : a->inputs) if (!input.is_null()) build_topo(input.impl());
            topo.push_back(a);
        }
    };

    build_topo(impl_);
    printf("DEBUG: Topo size: %zu\n", topo.size());
    int node_idx = 0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_op) {
            float g0 = 0;
            if ((*it)->grad) g0 = (*it)->grad->data->at(0);
            printf("DEBUG: Node %d: shape=[", node_idx++);
            for(int s : (*it)->shape) printf("%d,", s);
            printf("], grad[0]=%f, req_grad=%d\n", g0, (*it)->requires_grad);
            (*it)->backward_op();
        }
    }
}

} // namespace mlx::core