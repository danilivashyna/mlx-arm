// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

/**
 * Q4_0 Quantization for LLM Inference
 * 
 * Reduces weight memory by 4x: FP32 → 4-bit integers
 * Each block: 32 weights + 1 FP16 scale = 18 bytes (vs 128 bytes FP32)
 */

#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// Q4_0 block structure (18 bytes for 32 weights)
struct BlockQ4_0 {
    uint16_t scale;      // FP16 scale factor (stored as uint16)
    uint8_t weights[16]; // 32 weights packed (2 per byte, 4 bits each)
};

// FP32 → FP16 conversion (software fallback)
inline uint16_t float_to_fp16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(f));
    
    uint32_t sign = (u >> 31) & 0x1;
    uint32_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    
    // Handle special cases
    if (exp == 255) { // Inf or NaN
        return (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);
    }
    
    int32_t new_exp = exp - 127 + 15;
    if (new_exp <= 0) { // Underflow → zero
        return sign << 15;
    }
    if (new_exp >= 31) { // Overflow → inf
        return (sign << 15) | 0x7C00;
    }
    
    return (sign << 15) | (new_exp << 10) | (mant >> 13);
}

// FP16 → FP32 conversion
inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    uint32_t f;
    if (exp == 0) { // Zero or denormal
        f = (sign << 31);
    } else if (exp == 31) { // Inf or NaN
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    
    float result;
    std::memcpy(&result, &f, sizeof(f));
    return result;
}

/**
 * Quantize FP32 weights to Q4_0 format with correct layout for GPU
 * 
 * CRITICAL: B matrix is [K×N] in row-major, but GPU expects [N×K_blocks]!
 * We quantize column-by-column for optimal GPU access pattern.
 * 
 * Algorithm:
 * 1. For each column in B (N columns total)
 * 2. Split column into blocks of 32 elements
 * 3. Quantize each block independently
 * 4. Store as [N × K_blocks] layout
 * 
 * @param B Input FP32 weights matrix [K×N] row-major
 * @param K Number of rows (must be multiple of 32)
 * @param N Number of columns
 * @return Vector of Q4_0 blocks in [N × K_blocks] layout
 */
inline std::vector<BlockQ4_0> quantize_q4_0_matrix(const float* B, size_t K, size_t N) {
    if (K % 32 != 0) {
        throw std::runtime_error("K must be multiple of 32 for Q4_0");
    }
    
    size_t K_blocks = K / 32;
    std::vector<BlockQ4_0> blocks(N * K_blocks);
    
    // Iterate by COLUMNS first (correct layout for GPU)
    for (size_t col = 0; col < N; col++) {
        for (size_t block_idx = 0; block_idx < K_blocks; block_idx++) {
            size_t out_idx = col * K_blocks + block_idx;
            
            // Extract 32 weights from this column
            float block_weights[32];
            for (int i = 0; i < 32; i++) {
                size_t k = block_idx * 32 + i;
                // B is [K×N] row-major: B[k][col] = B[k * N + col]
                block_weights[i] = B[k * N + col];
            }
            
            // 1. Find max absolute value for this block
            float max_abs = 0.0f;
            for (int i = 0; i < 32; i++) {
                max_abs = std::max(max_abs, std::abs(block_weights[i]));
            }
            
            // 2. Compute scale (use 7.5 for better range utilization)
            float scale = max_abs / 7.5f;
            if (scale == 0.0f) scale = 1.0f; // Avoid division by zero
            blocks[out_idx].scale = float_to_fp16(scale);
            
            // 3. Quantize 32 weights → 16 bytes
            for (int i = 0; i < 16; i++) {
                float w0 = block_weights[i * 2];
                float w1 = block_weights[i * 2 + 1];
                
                // Quantize to -8..7 range (4-bit signed)
                int q0 = static_cast<int>(std::round(w0 / scale));
                int q1 = static_cast<int>(std::round(w1 / scale));
                
                // Clamp to -8..7
                q0 = std::clamp(q0, -8, 7);
                q1 = std::clamp(q1, -8, 7);
                
                // Convert to unsigned 0-15 for storage
                uint8_t u0 = static_cast<uint8_t>(q0 + 8);
                uint8_t u1 = static_cast<uint8_t>(q1 + 8);
                
                // Pack: low 4 bits = q0, high 4 bits = q1
                blocks[out_idx].weights[i] = (u1 << 4) | u0;
            }
        }
    }
    
    return blocks;
}

// Legacy function for backward compatibility
inline std::vector<BlockQ4_0> quantize_q4_0(const float* weights, size_t size) {
    if (size % 32 != 0) {
        throw std::runtime_error("Size must be multiple of 32 for Q4_0");
    }
    
    size_t num_blocks = size / 32;
    std::vector<BlockQ4_0> blocks(num_blocks);
    
    for (size_t b = 0; b < num_blocks; b++) {
        const float* block_weights = &weights[b * 32];
        
        // 1. Find max absolute value for this block
        float max_abs = 0.0f;
        for (int i = 0; i < 32; i++) {
            max_abs = std::max(max_abs, std::abs(block_weights[i]));
        }
        
        // 2. Compute scale (use 7.5 for better utilization)
        float scale = max_abs / 7.5f;
        if (scale == 0.0f) scale = 1.0f;
        blocks[b].scale = float_to_fp16(scale);
        
        // 3. Quantize 32 weights → 16 bytes
        for (int i = 0; i < 16; i++) {
            float w0 = block_weights[i * 2];
            float w1 = block_weights[i * 2 + 1];
            
            int q0 = static_cast<int>(std::round(w0 / scale));
            int q1 = static_cast<int>(std::round(w1 / scale));
            
            q0 = std::clamp(q0, -8, 7);
            q1 = std::clamp(q1, -8, 7);
            
            uint8_t u0 = static_cast<uint8_t>(q0 + 8);
            uint8_t u1 = static_cast<uint8_t>(q1 + 8);
            
            blocks[b].weights[i] = (u1 << 4) | u0;
        }
    }
    
    return blocks;
}

/**
 * Dequantize Q4_0 block back to FP32 (for verification)
 */
inline void dequantize_q4_0_block(const BlockQ4_0& block, float* output) {
    float scale = fp16_to_float(block.scale);
    
    for (int i = 0; i < 16; i++) {
        uint8_t packed = block.weights[i];
        
        // Extract low 4 bits
        int q0 = packed & 0x0F;
        int q1 = (packed >> 4) & 0x0F;
        
        // Dequantize: shift back to -8..7, then scale
        output[i * 2] = scale * (q0 - 8);
        output[i * 2 + 1] = scale * (q1 - 8);
    }
}

/**
 * Dequantize entire Q4_0 matrix
 */
inline std::vector<float> dequantize_q4_0(const std::vector<BlockQ4_0>& blocks) {
    size_t size = blocks.size() * 32;
    std::vector<float> weights(size);
    
    for (size_t b = 0; b < blocks.size(); b++) {
        dequantize_q4_0_block(blocks[b], &weights[b * 32]);
    }
    
    return weights;
}
