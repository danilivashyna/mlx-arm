// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/ops.h"
#include "mlx/core/array.h"
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <arm_neon.h>

namespace mlx::core {

void check_same_shape(const Array& a, const Array& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shape mismatch");
    }
}

Array add(const Array& a, const Array& b) {
    check_same_shape(a, b);
    std::vector<float> res_data(a.size());
    const float* a_ptr = a.data<float>();
    const float* b_ptr = b.data<float>();
    for (size_t i = 0; i < a.size(); ++i) res_data[i] = a_ptr[i] + b_ptr[i];
    return Array(res_data, a.shape());
}

Array multiply(const Array& a, const Array& b) {
    check_same_shape(a, b);
    std::vector<float> res_data(a.size());
    const float* a_ptr = a.data<float>();
    const float* b_ptr = b.data<float>();
    for (size_t i = 0; i < a.size(); ++i) res_data[i] = a_ptr[i] * b_ptr[i];
    return Array(res_data, a.shape());
}

Array multiply(const Array& a, float b) {
    std::vector<float> res_data(a.size());
    const float* a_ptr = a.data<float>();
    for (size_t i = 0; i < a.size(); ++i) res_data[i] = a_ptr[i] * b;
    return Array(res_data, a.shape());
}

Array silu(const Array& a) {
    std::vector<float> res_data(a.size());
    const float* a_ptr = a.data<float>();
    for (size_t i = 0; i < a.size(); ++i) {
        float x = a_ptr[i];
        res_data[i] = x / (1.0f + std::exp(-x));
    }
    return Array(res_data, a.shape());
}

Array rms_norm(const Array& x, const Array& weight, float eps) {
    int D = weight.size();
    int batch_size = x.size() / D;
    std::vector<float> res_data(x.size());
    const float* x_ptr = x.data<float>();
    const float* w_ptr = weight.data<float>();
    for (int b = 0; b < batch_size; ++b) {
        float sum_sq = 0;
        for (int i = 0; i < D; ++i) {
            float val = x_ptr[b * D + i];
            sum_sq += val * val;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / D + eps);
        for (int i = 0; i < D; ++i) res_data[b * D + i] = (x_ptr[b * D + i] * inv_rms) * w_ptr[i];
    }
    return Array(res_data, x.shape());
}

Array rope(const Array& x, int dims, int offset, float theta, float scale) {
    // x can be [L, D] or [batch, L, D]
    // dims is head_dim
    int ndim = x.ndim();
    int L = x.shape()[ndim - 2];
    int D = x.shape()[ndim - 1];
    int num_heads = D / dims;
    int batch_size = x.size() / (L * D);

    std::vector<float> res_data(x.size());
    const float* x_ptr = x.data<float>();

    for (int b = 0; b < batch_size; ++b) {
        for (int l = 0; l < L; ++l) {
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < dims / 2; ++d) {
                    float freq = 1.0f / std::pow(theta, 2.0f * d / dims);
                    float val = (l + offset) * scale * freq;
                    float cos_v = std::cos(val);
                    float sin_v = std::sin(val);

                    int idx0 = b * (L * D) + l * D + h * dims + d;
                    int idx1 = b * (L * D) + l * D + h * dims + d + dims / 2;
                    
                    float x0 = x_ptr[idx0];
                    float x1 = x_ptr[idx1];
                    res_data[idx0] = x0 * cos_v - x1 * sin_v;
                    res_data[idx1] = x0 * sin_v + x1 * cos_v;
                }
            }
        }
    }
    return Array(res_data, x.shape());
}

Array matmul(const Array& a, const Array& b) {
    const auto& a_shape = a.shape();
    const auto& b_shape = b.shape();
    int K = b_shape[0];
    int N = b_shape[1];
    int M = a_shape[a_shape.size()-2];
    
    std::vector<int> res_shape = a_shape;
    res_shape.back() = N;
    
    std::vector<float> res_data(M * N, 0.0f);
    const float* a_ptr = a.data<float>();
    const float* b_ptr = b.data<float>();

    // Optimized cache-friendly matmul: Rows of A x Rows of B
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_val = a_ptr[i * K + k];
            for (int j = 0; j < N; ++j) {
                res_data[i * N + j] += a_val * b_ptr[k * N + j];
            }
        }
    }
    return Array(res_data, res_shape);
}

Array softmax(const Array& a, int axis) {
    int D = a.shape().back();
    int batch_size = a.size() / D;
    std::vector<float> res_data(a.size());
    const float* a_ptr = a.data<float>();
    for (int b = 0; b < batch_size; ++b) {
        const float* row = a_ptr + b * D;
        float max_val = *std::max_element(row, row + D);
        float sum = 0;
        for (int i = 0; i < D; ++i) {
            res_data[b * D + i] = std::exp(row[i] - max_val);
            sum += res_data[b * D + i];
        }
        for (int i = 0; i < D; ++i) res_data[b * D + i] /= sum;
    }
    return Array(res_data, a.shape());
}

Array reshape(const Array& a, const std::vector<int>& shape) { return Array(a.data_shared(), shape, a.dtype()); }

Array transpose(const Array& a, const std::vector<int>& axes) {
    if (a.ndim() == 2 && axes[0] == 1 && axes[1] == 0) {
        int M = a.shape()[0], N = a.shape()[1];
        std::vector<float> res(a.size());
        const float* p = a.data<float>();
        for(int i=0; i<M; ++i) for(int j=0; j<N; ++j) res[j*M+i] = p[i*N+j];
        return Array(res, {N, M});
    }
    return a;
}

} // namespace mlx::core