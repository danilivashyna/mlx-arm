// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/ops.h"
#include "mlx/core/array.h"
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cstring>

namespace mlx::core {

void ensure_grad(Array a) {
    if (!a.impl()->grad) {
        a.impl()->grad = std::make_shared<ArrayImpl>(a.shape(), a.dtype());
    }
}

Array add(const Array& a, const Array& b) {
    Array res(a.shape(), a.dtype());
    float *rp = res.data();
    const float *ap = a.data(), *bp = b.data();
    for(size_t i=0; i<a.size(); ++i) rp[i] = ap[i] + bp[i];
    res.impl()->requires_grad = a.requires_grad() || b.requires_grad();
    res.impl()->inputs = {a, b};
    res.impl()->backward_op = [a, b, res]() {
        if (!res.impl()->grad) return;
        float* g = res.impl()->grad->data->data();
        if (a.requires_grad()) {
            ensure_grad(a); float* ag = a.impl()->grad->data->data();
            for (size_t i = 0; i < a.size(); ++i) ag[i] += g[i];
        }
        if (b.requires_grad()) {
            ensure_grad(b); float* bg = b.impl()->grad->data->data();
            for (size_t i = 0; i < b.size(); ++i) bg[i] += g[i];
        }
    };
    return res;
}

Array multiply(const Array& a, const Array& b) {
    Array res(a.shape(), a.dtype());
    float *rp = res.data();
    const float *ap = a.data(), *bp = b.data();
    for(size_t i=0; i<a.size(); ++i) rp[i] = ap[i] * bp[i];
    res.impl()->requires_grad = a.requires_grad() || b.requires_grad();
    res.impl()->inputs = {a, b};
    res.impl()->backward_op = [a, b, res]() {
        if (!res.impl()->grad) return;
        float* g = res.impl()->grad->data->data();
        if (a.requires_grad()) {
            ensure_grad(a); float* ag = a.impl()->grad->data->data();
            const float* bd = b.data();
            for (size_t i = 0; i < a.size(); ++i) ag[i] += g[i] * bd[i];
        }
        if (b.requires_grad()) {
            ensure_grad(b); float* bg = b.impl()->grad->data->data();
            const float* ad = a.data();
            for (size_t i = 0; i < b.size(); ++i) bg[i] += g[i] * ad[i];
        }
    };
    return res;
}

Array multiply(const Array& a, float b) {
    Array res(a.shape(), a.dtype());
    float *rp = res.data(); const float* ap = a.data();
    for(size_t i=0; i<a.size(); ++i) rp[i] = ap[i] * b;
    res.impl()->requires_grad = a.requires_grad();
    res.impl()->inputs = {a};
    res.impl()->backward_op = [a, b, res]() {
        if (!res.impl()->grad || !a.requires_grad()) return;
        ensure_grad(a); float* ag = a.impl()->grad->data->data();
        float* g = res.impl()->grad->data->data();
        for (size_t i = 0; i < a.size(); ++i) ag[i] += g[i] * b;
    };
    return res;
}

Array matmul(const Array& a, const Array& b) {
    int M = a.shape()[a.ndim()-2], K = a.shape().back(), N = b.shape().back();
    auto rs = a.shape(); rs.back() = N;
    Array res(rs, a.dtype());
    float *rp = res.data(); const float *ap = a.data(), *bp = b.data();
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            float av = ap[i * K + k];
            for (int j = 0; j < N; ++j) rp[i * N + j] += av * bp[k * N + j];
        }
    res.impl()->requires_grad = a.requires_grad() || b.requires_grad();
    res.impl()->inputs = {a, b};
    res.impl()->backward_op = [a, b, res]() {
        if (!res.impl()->grad) return;
        float* gy = res.impl()->grad->data->data();
        int M = a.shape()[a.ndim()-2], K = a.shape().back(), N = b.shape().back();
        if (a.requires_grad()) {
            ensure_grad(a); float* ga = a.impl()->grad->data->data();
            const float* bd = b.data();
            for(int i=0; i<M; ++i) for(int j=0; j<N; ++j) {
                float g = gy[i*N+j];
                for(int k=0; k<K; ++k) ga[i*K+k] += g * bd[k*N+j];
            }
        }
        if (b.requires_grad()) {
            ensure_grad(b); float* gb = b.impl()->grad->data->data();
            const float* ad = a.data();
            for(int k=0; k<K; ++k) for(int i=0; i<M; ++i) {
                float av = ad[i*K+k];
                for(int j=0; j<N; ++j) gb[k*N+j] += av * gy[i*N+j];
            }
        }
    };
    return res;
}

Array transpose(const Array& a, const std::vector<int>& axes) {
    int M = a.shape()[0], N = a.shape()[1];
    Array res({N, M}, a.dtype());
    float *rd = res.data(); const float* p = a.data();
    for(int i=0; i<M; ++i) for(int j=0; j<N; ++j) rd[j*M+i] = p[i*N+j];
    res.impl()->requires_grad = a.requires_grad();
    res.impl()->inputs = {a};
    res.impl()->backward_op = [a, res]() {
        if (!res.impl()->grad || !a.requires_grad()) return;
        ensure_grad(a); float* ag = a.impl()->grad->data->data();
        float* g = res.impl()->grad->data->data();
        int MN = res.shape()[0], NN = res.shape()[1];
        for(int i=0; i<MN; ++i) for(int j=0; j<NN; ++j) ag[j*MN+i] += g[i*NN+j];
    };
    return res;
}

Array reshape(const Array& a, const std::vector<int>& shape) {
    Array res(a.data_shared(), shape, a.dtype());
    res.impl()->requires_grad = a.requires_grad();
    res.impl()->inputs = {a};
    res.impl()->backward_op = [a, res]() {
        if (!res.impl()->grad || !a.requires_grad()) return;
        ensure_grad(a); float* ag = a.impl()->grad->data->data();
        float* g = res.impl()->grad->data->data();
        for (size_t i = 0; i < a.size(); ++i) ag[i] += g[i];
    };
    return res;
}

Array softmax(const Array& a, int axis) {
    int D = a.shape().back(), B = a.size() / D;
    Array res(a.shape(), a.dtype());
    float* rp = res.data(); const float* ap = a.data();
    for (int b = 0; b < B; ++b) {
        const float* row = ap + b * D;
        float mv = *std::max_element(row, row + D);
        float s = 0;
        for (int i = 0; i < D; ++i) { rp[b*D+i] = std::exp(row[i]-mv); s += rp[b*D+i]; }
        for (int i = 0; i < D; ++i) rp[b*D+i] /= s;
    }
    res.impl()->requires_grad = a.requires_grad();
    res.impl()->inputs = {a};
    res.impl()->backward_op = [a, res]() {
        if (!res.impl()->grad || !a.requires_grad()) return;
        ensure_grad(a); float* ag = a.impl()->grad->data->data();
        float* gy = res.impl()->grad->data->data();
        const float* y = res.data();
        int D = a.shape().back(), B = a.size() / D;
        for (int b = 0; b < B; ++b) {
            float dot = 0;
            for (int i = 0; i < D; ++i) dot += gy[b*D+i] * y[b*D+i];
            for (int i = 0; i < D; ++i) ag[b*D+i] += y[b*D+i] * (gy[b*D+i] - dot);
        }
    };
    return res;
}

Array silu(const Array& a) {
    Array res(a.shape(), a.dtype());
    float* rp = res.data(); const float* ap = a.data();
    for (size_t i = 0; i < a.size(); ++i) {
        float sig = 1.0f / (1.0f + std::exp(-ap[i]));
        rp[i] = ap[i] * sig;
    }
    res.impl()->requires_grad = a.requires_grad();
    res.impl()->inputs = {a};
    res.impl()->backward_op = [a, res]() {
        if (!res.impl()->grad || !a.requires_grad()) return;
        ensure_grad(a); float* ag = a.impl()->grad->data->data();
        float* gy = res.impl()->grad->data->data();
        const float* ad = a.data();
        for (size_t i = 0; i < a.size(); ++i) {
            float sig = 1.0f / (1.0f + std::exp(-ad[i]));
            ag[i] += gy[i] * sig * (1.0f + ad[i] * (1.0f - sig));
        }
    };
    return res;
}

Array rms_norm(const Array& x, const Array& weight, float eps) {
    int D = weight.size(), B = x.size() / D;
    Array res(x.shape(), x.dtype());
    float* rd = res.data(); const float *xp = x.data(), *wp = weight.data();
    for (int b = 0; b < B; ++b) {
        float ss = 0; for (int i = 0; i < D; ++i) { float v = xp[b*D+i]; ss += v*v; }
        float ir = 1.0f / std::sqrt(ss/D + eps);
        for (int i = 0; i < D; ++i) rd[b*D+i] = (xp[b*D+i]*ir)*wp[i];
    }
    res.impl()->requires_grad = x.requires_grad() || weight.requires_grad();
    res.impl()->inputs = {x, weight};
    res.impl()->backward_op = [x, weight, eps, res]() {
        if (!res.impl()->grad) return;
        float* gy = res.impl()->grad->data->data();
        const float* xd = x.data();
        const float* wd = weight.data();
        int D = weight.size(), B = x.size() / D;

        if (x.requires_grad()) {
            ensure_grad(x); float* gx = x.impl()->grad->data->data();
            for (int b = 0; b < B; ++b) {
                float ss = 0; for (int i = 0; i < D; ++i) { float v = xd[b*D+i]; ss += v*v; }
                float rms2 = ss/D + eps;
                float irms = 1.0f / std::sqrt(rms2);
                float irms3 = irms / rms2;
                float dot = 0;
                for (int i = 0; i < D; ++i) dot += gy[b*D+i] * wd[i] * xd[b*D+i];
                for (int i = 0; i < D; ++i) {
                    gx[b*D+i] += (gy[b*D+i] * wd[i] * irms) - (xd[b*D+i] * dot * irms3 / D);
                }
            }
        }
        if (weight.requires_grad()) {
            ensure_grad(weight); float* gw = weight.impl()->grad->data->data();
            for (int b = 0; b < B; ++b) {
                float ss = 0; for (int i = 0; i < D; ++i) { float v = xd[b*D+i]; ss += v*v; }
                float irms = 1.0f / std::sqrt(ss/D + eps);
                for (int i = 0; i < D; ++i) gw[i] += gy[b*D+i] * (xd[b*D+i] * irms);
            }
        }
    };
    return res;
}

Array rope(const Array& x, int dims, int offset, float theta, float scale) {
    int L = x.shape()[x.ndim()-2], D = x.shape()[x.ndim()-1], heads = D / dims, B = x.size() / (L * D);
    Array res(x.shape(), x.dtype());
    float* rd = res.data(); const float* xp = x.data();
    for (int b = 0; b < B; ++b)
    for (int l = 0; l < L; ++l)
    for (int h = 0; h < heads; ++h)
    for (int d = 0; d < dims / 2; ++d) {
        float v = (l + offset) * scale * (1.0f / std::pow(theta, 2.0f * d / dims));
        float cv = std::cos(v), sv = std::sin(v);
        int i0 = b*L*D + l*D + h*dims + d, i1 = i0 + dims/2;
        rd[i0] = xp[i0]*cv - xp[i1]*sv; rd[i1] = xp[i0]*sv + xp[i1]*cv;
    }
    res.impl()->requires_grad = x.requires_grad();
    res.impl()->inputs = {x};
    res.impl()->backward_op = [x, dims, offset, theta, scale, res]() {
        if (!res.impl()->grad || !x.requires_grad()) return;
        ensure_grad(x); float* gx = x.impl()->grad->data->data();
        float* gy = res.impl()->grad->data->data();
        int L = x.shape()[x.ndim()-2], D = x.shape()[x.ndim()-1], heads = D / dims, B = x.size() / (L * D);
        for (int b = 0; b < B; ++b)
        for (int l = 0; l < L; ++l)
        for (int h = 0; h < heads; ++h)
        for (int d = 0; d < dims / 2; ++d) {
            float v = (l + offset) * scale * (1.0f / std::pow(theta, 2.0f * d / dims));
            float cv = std::cos(v), sv = std::sin(v);
            int i0 = b*L*D + l*D + h*dims + d, i1 = i0 + dims/2;
            gx[i0] += gy[i0] * cv + gy[i1] * sv;
            gx[i1] += -gy[i0] * sv + gy[i1] * cv;
        }
    };
    return res;
}

Array cross_entropy(const Array& logits, const Array& targets) {
    int B = logits.shape()[0], V = logits.shape()[1];
    Array res({1}, logits.dtype());
    const float *lp = logits.data(), *tp = targets.data();
    std::vector<float> probs(B * V);
    float tl = 0;
    for (int b = 0; b < B; ++b) {
        const float* row = lp + b * V;
        float mv = *std::max_element(row, row + V);
        float s = 0;
        for (int i = 0; i < V; ++i) { probs[b*V+i] = std::exp(row[i]-mv); s += probs[b*V+i]; }
        for (int i = 0; i < V; ++i) probs[b*V+i] /= s;
        int tidx = static_cast<int>(tp[b]);
        tl -= std::log(std::max(probs[b*V+tidx], 1e-9f));
    }
    res.data()[0] = tl / B;
    res.impl()->requires_grad = logits.requires_grad();
    res.impl()->inputs = {logits};
    res.impl()->backward_op = [logits, probs, tp_v = std::vector<float>(tp, tp+B), B, V, res]() {
        if (!logits.requires_grad()) return;
        ensure_grad(logits); float* lg = logits.impl()->grad->data->data();
        float* loss_grad = res.impl()->grad->data->data();
        for (int b = 0; b < B; ++b) {
            int tidx = static_cast<int>(tp_v[b]);
            for (int v = 0; v < V; ++v) lg[b*V+v] += loss_grad[0] * (probs[b*V+v] - (v == tidx ? 1.0f : 0.0f)) / B;
        }
    };
    return res;
}

Array concat(const std::vector<Array>& arrays, int axis) {
    if (arrays.empty()) return Array();
    auto shape = arrays[0].shape();
    size_t total_dim = 0;
    for (auto& a : arrays) total_dim += a.shape()[axis];
    auto out_shape = shape; out_shape[axis] = total_dim;
    
    Array res(out_shape, arrays[0].dtype());
    float* rd = res.data();
    size_t offset = 0;
    int inner_size = 1;
    for (size_t i = (size_t)axis + 1; i < shape.size(); ++i) inner_size *= shape[i];
    int outer_size = 1;
    for (int i = 0; i < axis; ++i) outer_size *= shape[i];

    for (int o = 0; o < outer_size; ++o) {
        for (auto& a : arrays) {
            int dim = a.shape()[axis];
            std::memcpy(rd + (o * total_dim + offset) * inner_size, 
                        a.data() + o * dim * inner_size, 
                        dim * inner_size * sizeof(float));
            offset += dim;
        }
        offset = 0;
    }

    res.impl()->inputs = arrays;
    bool req = false; for(auto& a : arrays) req |= a.requires_grad();
    res.impl()->requires_grad = req;
    res.impl()->backward_op = [arrays, axis, res]() {
        if (!res.impl()->grad) return;
        float* gy = res.impl()->grad->data->data();
        auto out_shape = res.shape();
        int total_dim = out_shape[axis];
        int inner_size = 1;
        for (size_t i = (size_t)axis + 1; i < out_shape.size(); ++i) inner_size *= out_shape[i];
        int outer_size = 1;
        for (int i = 0; i < axis; ++i) outer_size *= out_shape[i];

        size_t offset = 0;
        for (int o = 0; o < outer_size; ++o) {
            for (auto& a : arrays) {
                int dim = a.shape()[axis];
                if (a.requires_grad()) {
                    ensure_grad(a); float* ag = a.impl()->grad->data->data();
                    for (int i = 0; i < dim * inner_size; ++i)
                        ag[o * dim * inner_size + i] += gy[(o * total_dim + offset) * inner_size + i];
                }
                offset += dim;
            }
            offset = 0;
        }
    };
    return res;
}

Array slice(const Array& a, const std::vector<int>& start, const std::vector<int>& end) {
    std::vector<int> out_shape;
    for (size_t i = 0; i < start.size(); ++i) out_shape.push_back(end[i] - start[i]);
    
    Array res(out_shape, a.dtype());
    float* rd = res.data(); const float* ad = a.data();
    if (a.ndim() == 2) {
        for (int i = 0; i < out_shape[0]; ++i)
            std::memcpy(rd + i * out_shape[1], ad + (start[0] + i) * a.shape()[1] + start[1], out_shape[1] * sizeof(float));
    } else if (a.ndim() == 3) {
         for (int i = 0; i < out_shape[0]; ++i)
            for (int j = 0; j < out_shape[1]; ++j)
                std::memcpy(rd + (i * out_shape[1] + j) * out_shape[2], 
                            ad + ((start[0] + i) * a.shape()[1] + (start[1] + j)) * a.shape()[2] + start[2], 
                            out_shape[2] * sizeof(float));
    }

    res.impl()->inputs = {a};
    res.impl()->requires_grad = a.requires_grad();
    res.impl()->backward_op = [a, start, end, res]() {
        if (!res.impl()->grad || !a.requires_grad()) return;
        ensure_grad(a); float* ag = a.impl()->grad->data->data();
        float* gy = res.impl()->grad->data->data();
        auto out_shape = res.shape();
        if (a.ndim() == 2) {
            for (int i = 0; i < out_shape[0]; ++i)
                for (int j = 0; j < out_shape[1]; ++j)
                    ag[(start[0] + i) * a.shape()[1] + (start[1] + j)] += gy[i * out_shape[1] + j];
        } else if (a.ndim() == 3) {
            for (int i = 0; i < out_shape[0]; ++i)
                for (int j = 0; j < out_shape[1]; ++j)
                    for (int k = 0; k < out_shape[2]; ++k)
                        ag[((start[0] + i) * a.shape()[1] + (start[1] + j)) * a.shape()[2] + (start[2] + k)] += gy[(i * out_shape[1] + j) * out_shape[2] + k];
        }
    };
    return res;
}

Array embedding(const Array& weight, const Array& indices) {
    int num_embeddings = weight.shape()[0];
    int dims = weight.shape()[1];
    std::vector<int> out_shape = indices.shape();
    out_shape.push_back(dims);
    
    Array res(out_shape, weight.dtype());
    const float* w_ptr = weight.data();
    const float* i_ptr = indices.data();
    float* r_ptr = res.data();
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = static_cast<int>(i_ptr[i]);
        if (idx < 0) idx = 0;
        if (idx >= num_embeddings) idx = num_embeddings - 1;
        std::memcpy(r_ptr + i * dims, w_ptr + idx * dims, dims * sizeof(float));
    }

    res.impl()->inputs = {weight, indices};
    res.impl()->requires_grad = weight.requires_grad();
    res.impl()->backward_op = [weight, indices, res]() {
        if (!weight.requires_grad() || !res.impl()->grad) return;
        ensure_grad(weight);
        float* wg = weight.impl()->grad->data->data();
        float* gy = res.impl()->grad->data->data();
        const float* i_ptr = indices.data();
        int num_embeddings = weight.shape()[0];
        int dims = weight.shape()[1];
        
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = static_cast<int>(i_ptr[i]);
            if (idx < 0) idx = 0;
            if (idx >= num_embeddings) idx = num_embeddings - 1;
            for (int d = 0; d < dims; ++d) {
                wg[idx * dims + d] += gy[i * dims + d];
            }
        }
    };
    return res;
}

} // namespace mlx::core