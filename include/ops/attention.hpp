#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace ops {

/**
 * Simplified attention for initial testing
 * TODO: Implement proper GPU-accelerated attention
 * 
 * For now: Use mean-pooling over sequence as a placeholder
 * This will at least produce reasonable outputs while we optimize
 */
inline std::vector<float> attention(
    const std::vector<float>& Q,
    const std::vector<float>& K_full,
    const std::vector<float>& V_full,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t head_dim
) {
    size_t hidden_dim = num_heads * head_dim;
    std::vector<float> output(hidden_dim, 0.0f);
    
    // Simple strategy: Average V vectors weighted toward recent tokens
    // This avoids O(seq_len^2) CPU computation
    
    if (seq_len == 0 || V_full.size() != seq_len * hidden_dim) {
        // Return Q as fallback
        return Q;
    }
    
    // Weighted average: recent tokens get more weight
    // Weight decays exponentially: w_t = exp(-λ * (seq_len - t - 1))
    float lambda = 0.1f;  // Decay rate
    float total_weight = 0.0f;
    
    for (uint32_t t = 0; t < seq_len; t++) {
        // Recent tokens (higher t) get higher weight
        float distance = (float)(seq_len - t - 1);
        float weight = expf(-lambda * distance);
        total_weight += weight;
        
        size_t v_offset = t * hidden_dim;
        for (uint32_t i = 0; i < hidden_dim; i++) {
            output[i] += weight * V_full[v_offset + i];
        }
    }
    
    // Normalize
    if (total_weight > 0.0f) {
        for (uint32_t i = 0; i < hidden_dim; i++) {
            output[i] /= total_weight;
        }
    }
    
    // Mix with Q (25% Q, 75% attended V)
    for (uint32_t i = 0; i < hidden_dim; i++) {
        output[i] = 0.25f * Q[i] + 0.75f * output[i];
    }
    
    return output;
}

} // namespace ops
