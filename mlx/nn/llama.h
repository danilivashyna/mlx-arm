// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/nn/module.h"
#include "mlx/nn/layers.h"
#include "mlx/core/ops.h"
#include <vector>
#include <memory>
#include <string>
#include <cmath>

namespace mlx::nn {

using namespace mlx::core;

struct LlamaConfig {
    int vocab_size = 32000;
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 32;
    float rms_norm_eps = 1e-6;
    float rope_theta = 10000.0;
    float rope_scale = 1.0;
};

class MLP : public Module {
public:
    MLP(const LlamaConfig& args) {
        int dim = args.hidden_size;
        int hidden_dim = args.intermediate_size;
        
        gate_proj = std::make_shared<Linear>(dim, hidden_dim, false);
        up_proj = std::make_shared<Linear>(dim, hidden_dim, false);
        down_proj = std::make_shared<Linear>(hidden_dim, dim, false);
        
        register_module("gate_proj", gate_proj);
        register_module("up_proj", up_proj);
        register_module("down_proj", down_proj);
    }

    Array operator()(const Array& x) override {
        auto gate = (*gate_proj)(x);
        auto up = (*up_proj)(x);
        auto activated = silu(gate);
        auto merged = multiply(activated, up);
        return (*down_proj)(merged);
    }

    std::shared_ptr<Linear> gate_proj, up_proj, down_proj;
};

class Attention : public Module {
public:
    Attention(const LlamaConfig& args) : args_(args) {
        int dim = args.hidden_size;
        int heads = args.num_attention_heads;
        int kv_heads = args.num_key_value_heads;
        int head_dim = dim / heads;
        
        scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        q_proj = std::make_shared<Linear>(dim, heads * head_dim, false);
        k_proj = std::make_shared<Linear>(dim, kv_heads * head_dim, false);
        v_proj = std::make_shared<Linear>(dim, kv_heads * head_dim, false);
        o_proj = std::make_shared<Linear>(heads * head_dim, dim, false);
        
        register_module("q_proj", q_proj);
        register_module("k_proj", k_proj);
        register_module("v_proj", v_proj);
        register_module("o_proj", o_proj);
    }

    void clear_cache() {
        k_cache_ = Array();
        v_cache_ = Array();
    }

    Array operator()(const Array& x, int offset = 0) {
        auto q = (*q_proj)(x); 
        auto k = (*k_proj)(x); 
        auto v = (*v_proj)(x); 

        int q_heads = args_.num_attention_heads;
        int kv_heads = args_.num_key_value_heads;
        int head_dim = args_.hidden_size / q_heads;
        int L = q.shape()[q.ndim() - 2];

        q = rope(q, head_dim, offset, args_.rope_theta);
        k = rope(k, head_dim, offset, args_.rope_theta);

        // KV Cache update
        if (offset == 0 || k_cache_.size() == 0) {
            k_cache_ = k;
            v_cache_ = v;
        } else {
            std::vector<float> nk(k_cache_.size() + k.size());
            std::memcpy(nk.data(), k_cache_.data<float>(), k_cache_.size() * sizeof(float));
            std::memcpy(nk.data() + k_cache_.size(), k.data<float>(), k.size() * sizeof(float));
            
            std::vector<float> nv(v_cache_.size() + v.size());
            std::memcpy(nv.data(), v_cache_.data<float>(), v_cache_.size() * sizeof(float));
            std::memcpy(nv.data() + v_cache_.size(), v.data<float>(), v.size() * sizeof(float));

            auto new_shape = k_cache_.shape();
            new_shape[new_shape.size()-2] += L;
            k_cache_ = Array(nk, new_shape);
            v_cache_ = Array(nv, new_shape);
        }

        auto full_k = k_cache_;
        auto full_v = v_cache_;
        int total_L = full_k.shape()[full_k.ndim() - 2];

        // Process heads
        std::vector<float> result_data(L * q_heads * head_dim);
        const float* q_ptr = q.data<float>();
        const float* k_ptr = full_k.data<float>();
        const float* v_ptr = full_v.data<float>();

        for (int h = 0; h < q_heads; ++h) {
            int kv_h = h / (q_heads / kv_heads);
            std::vector<float> hq(L * head_dim);
            std::vector<float> hk(total_L * head_dim);
            std::vector<float> hv(total_L * head_dim);

            for (int i = 0; i < L; ++i) std::memcpy(hq.data() + i * head_dim, q_ptr + (i * q_heads + h) * head_dim, head_dim * sizeof(float));
            for (int i = 0; i < total_L; ++i) {
                std::memcpy(hk.data() + i * head_dim, k_ptr + (i * kv_heads + kv_h) * head_dim, head_dim * sizeof(float));
                std::memcpy(hv.data() + i * head_dim, v_ptr + (i * kv_heads + kv_h) * head_dim, head_dim * sizeof(float));
            }

            auto scores = multiply(matmul(Array(hq, {L, head_dim}), transpose(Array(hk, {total_L, head_dim}), {1, 0})), scale_);
            
            if (L > 1) {
                float* s_ptr = scores.data<float>();
                for (int row = 0; row < L; ++row)
                    for (int col = row + offset + 1; col < total_L; ++col)
                        s_ptr[row * total_L + col] = -1e9f;
            }

            auto out = matmul(softmax(scores), Array(hv, {total_L, head_dim}));
            for (int i = 0; i < L; ++i) std::memcpy(result_data.data() + (i * q_heads + h) * head_dim, out.data<float>() + i * head_dim, head_dim * sizeof(float));
        }

        return (*o_proj)(Array(result_data, q.shape()));
    }

    std::shared_ptr<Linear> q_proj, k_proj, v_proj, o_proj;
    LlamaConfig args_;
    float scale_;
    Array k_cache_, v_cache_;
};

class TransformerBlock : public Module {
public:
    TransformerBlock(const LlamaConfig& args) {
        self_attn = std::make_shared<Attention>(args);
        mlp = std::make_shared<MLP>(args);
        input_layernorm = std::make_shared<RMSNorm>(args.hidden_size, args.rms_norm_eps);
        post_attention_layernorm = std::make_shared<RMSNorm>(args.hidden_size, args.rms_norm_eps);
        
        register_module("self_attn", self_attn);
        register_module("mlp", mlp);
        register_module("input_layernorm", input_layernorm);
        register_module("post_attention_layernorm", post_attention_layernorm);
    }

    Array operator()(const Array& x, int offset = 0) {
        auto h = add(x, (*self_attn)((*input_layernorm)(x), offset));
        return add(h, (*mlp)((*post_attention_layernorm)(h)));
    }

    std::shared_ptr<Attention> self_attn;
    std::shared_ptr<MLP> mlp;
    std::shared_ptr<RMSNorm> input_layernorm, post_attention_layernorm;
};

class LlamaModel : public Module {
public:
    LlamaModel(const LlamaConfig& args) {
        embed_tokens = std::make_shared<Embedding>(args.vocab_size, args.hidden_size);
        norm = std::make_shared<RMSNorm>(args.hidden_size, args.rms_norm_eps);
        register_module("embed_tokens", embed_tokens);
        register_module("norm", norm);
        for (int i = 0; i < args.num_hidden_layers; ++i) {
            auto layer = std::make_shared<TransformerBlock>(args);
            layers.push_back(layer);
            register_module("layers." + std::to_string(i), layer);
        }
    }

    Array operator()(const Array& x, int offset = 0) {
        auto h = (*embed_tokens)(x);
        for (auto& layer : layers) h = (*layer)(h, offset);
        return (*norm)(h);
    }

    std::shared_ptr<Embedding> embed_tokens;
    std::vector<std::shared_ptr<TransformerBlock>> layers;
    std::shared_ptr<RMSNorm> norm;
};

class Model : public Module {
public:
    Model(const LlamaConfig& args) {
        model = std::make_shared<LlamaModel>(args);
        lm_head = std::make_shared<Linear>(args.hidden_size, args.vocab_size, false);
        register_module("model", model);
        register_module("lm_head", lm_head);
    }

    Array operator()(const Array& x, int offset = 0) {
        return (*lm_head)((*model)(x, offset));
    }

    std::shared_ptr<LlamaModel> model;
    std::shared_ptr<Linear> lm_head;
};

} // namespace mlx::nn