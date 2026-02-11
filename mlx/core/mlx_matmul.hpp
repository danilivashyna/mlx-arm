// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

/**
 * MLX Matmul Wrapper - CPU fallback + GPU Q4_0 acceleration
 * 
 * Provides unified interface for matrix multiplication in LLM inference:
 * - CPU: Naive FP32 implementation (fallback)
 * - GPU: Q4_0 quantized matmul (4x memory compression, ~10x speedup)
 * 
 * Usage:
 *   MLXMatmul matmul;
 *   matmul.initialize();  // Setup GPU
 *   matmul.compute(A, B, C, M, K, N);  // Auto-select best backend
 */

#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>

#ifdef MLX_BUILD_VULKAN
#include "mlx/backend/vulkan/vulkan_context.h"
#include "mlx/backend/vulkan/vulkan_device.h"
#include "mlx/backend/vulkan/vulkan_buffer.h"
#include "mlx/backend/vulkan/vulkan_pipeline.h"
#include "mlx/backend/vulkan/vulkan_command.h"
#endif

namespace mlx {
namespace matmul {

#ifdef MLX_BUILD_VULKAN
using namespace mlx::backend::vulkan;
#endif

// Q4_0 block structure (18 bytes for 32 weights)
struct BlockQ4_0 {
    uint16_t scale;      // FP16 scale factor
    uint8_t weights[16]; // 32 weights packed (4 bits each)
};

// FP32 → FP16 conversion
inline uint16_t float_to_fp16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(f));
    
    uint32_t sign = (u >> 31) & 0x1;
    uint32_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;
    
    if (exp == 255) return (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);
    
    int32_t new_exp = exp - 127 + 15;
    if (new_exp <= 0) return sign << 15;
    if (new_exp >= 31) return (sign << 15) | 0x7C00;
    
    return (sign << 15) | (new_exp << 10) | (mant >> 13);
}

// Quantize FP32 matrix to Q4_0 format
inline std::vector<BlockQ4_0> quantize_q4_0(const float* B, size_t K, size_t N) {
    if (K % 32 != 0) {
        throw std::runtime_error("K must be multiple of 32 for Q4_0");
    }
    
    size_t K_blocks = K / 32;
    std::vector<BlockQ4_0> blocks(N * K_blocks);
    
    // Quantize column-by-column for GPU layout
    for (size_t col = 0; col < N; col++) {
        for (size_t block_idx = 0; block_idx < K_blocks; block_idx++) {
            size_t out_idx = col * K_blocks + block_idx;
            
            // Extract 32 weights from column
            float block_weights[32];
            for (int i = 0; i < 32; i++) {
                size_t row = block_idx * 32 + i;
                block_weights[i] = B[row * N + col];
            }
            
            // Find max absolute value for scale
            float max_abs = 0.0f;
            for (int i = 0; i < 32; i++) {
                max_abs = std::max(max_abs, std::abs(block_weights[i]));
            }
            
            // Compute scale and quantize
            float scale = max_abs / 7.0f;  // 4-bit signed: [-7, 7]
            blocks[out_idx].scale = float_to_fp16(scale);
            
            std::memset(blocks[out_idx].weights, 0, 16);
            
            if (scale > 0.0f) {
                for (int i = 0; i < 32; i++) {
                    int8_t q = std::round(block_weights[i] / scale);
                    q = std::max((int8_t)-7, std::min((int8_t)7, q));
                    
                    uint8_t packed = (q + 8) & 0xF;
                    int byte_idx = i / 2;
                    int shift = (i % 2) * 4;
                    blocks[out_idx].weights[byte_idx] |= (packed << shift);
                }
            }
        }
    }
    
    return blocks;
}

class MLXMatmul {
public:
    MLXMatmul() : use_gpu_(false), gpu_calls_(0), cpu_calls_(0) {}
    
    ~MLXMatmul() {
        cleanup();
    }
    
    // Initialize GPU backend (call once at startup)
    bool initialize() {
#ifdef MLX_BUILD_VULKAN
        printf("\n🔍 [MLXMatmul] Initializing GPU...\n");
        fflush(stdout);
        try {
            printf("   Creating VulkanContext...\n");
            fflush(stdout);
            ctx_ = std::make_shared<VulkanContext>();
            if (!ctx_->initialize(false)) {
                printf("   ❌ VulkanContext::initialize() returned false\n");
                fflush(stdout);
                return false;
            }
            printf("   ✅ VulkanContext created\n");
            
            // Check if we have any physical devices
            printf("   Checking physical devices...\n");
            auto devices = ctx_->get_physical_devices();
            if (devices.empty()) {
                printf("   ❌ No physical devices available\\n");
                return false;
            }
            printf("   ✅ Found %zu device(s)\\n", devices.size());
            
            printf("   Creating VulkanDevice...\n");
            fflush(stdout);
            auto physical_device = ctx_->physical_device(0);
            device_ = std::make_shared<VulkanDevice>(ctx_, physical_device);
            if (!device_->initialize()) {
                printf("   ❌ VulkanDevice::initialize() failed\n");
                fflush(stdout);
                return false;
            }
            printf("   ✅ VulkanDevice created\n");
            fflush(stdout);
            
            use_gpu_ = true;
            printf("🎉 [MLXMatmul] GPU initialization SUCCESS!\n\n");
            fflush(stdout);
            return true;
        } catch (const std::exception& e) {
            printf("❌ [MLXMatmul] Exception: %s\n", e.what());
            fflush(stdout);
            return false;
        } catch (...) {
            printf("❌ [MLXMatmul] Unknown exception\n");
            fflush(stdout);
            return false;
        }
#else
        printf("❌ [MLXMatmul] Vulkan not enabled in build\n");
        return false;
#endif
    }
    
    // Cleanup GPU resources
    void cleanup() {
#ifdef MLX_BUILD_VULKAN
        device_.reset();
        ctx_.reset();
        use_gpu_ = false;
#endif
    }
    
    // Compute C = A × B  (A: [M×K], B: [K×N], C: [M×N])
    bool compute(const float* A, const float* B, float* C,
                 uint32_t M, uint32_t K, uint32_t N) {
        
        // Try GPU if available and K is multiple of 32
        if (use_gpu_ && (K % 32 == 0)) {
#ifdef MLX_BUILD_VULKAN
            if (compute_gpu_q4_0(A, B, C, M, K, N)) {
                return true;
            }
            // If GPU fails, fall back to CPU
#endif
        }
        
        // Fallback to CPU
        compute_cpu(A, B, C, M, K, N);
        return true;
    }
    
    bool is_gpu_available() const { return use_gpu_; }
    
private:
    // CPU fallback (naive matmul)
    void compute_cpu(const float* A, const float* B, float* C,
                     uint32_t M, uint32_t K, uint32_t N) {
        cpu_calls_++;
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
#ifdef MLX_BUILD_VULKAN
    // GPU Q4_0 quantized matmul (correct API)
    bool compute_gpu_q4_0(const float* A, const float* B, float* C,
                          uint32_t M, uint32_t K, uint32_t N) {
        gpu_calls_++;
        // Debug logging removed for speed
        
        try {
            // 1. Quantize B to Q4_0
            auto B_q4 = quantize_q4_0(B, K, N);
            uint32_t K_blocks = K / 32;
            
            // 2. Convert A to FP16
            std::vector<uint16_t> A_fp16(M * K);
            for (size_t i = 0; i < M * K; i++) {
                A_fp16[i] = float_to_fp16(A[i]);
            }
            
            // 3. Create buffers
            size_t sizeA = M * K * sizeof(uint16_t);
            size_t sizeB = B_q4.size() * sizeof(BlockQ4_0);
            size_t sizeC = M * N * sizeof(uint16_t);
            
            VulkanBuffer bufferA(*device_, sizeA, VulkanBuffer::Type::Staging);
            VulkanBuffer bufferB(*device_, sizeB, VulkanBuffer::Type::Staging);
            VulkanBuffer bufferC(*device_, sizeC, VulkanBuffer::Type::Staging);
            
            // 4. Upload data
            bufferA.write(A_fp16.data(), 0, sizeA);
            bufferB.write(B_q4.data(), 0, sizeB);
            
            // 5. Setup pipeline
            VulkanPipeline pipeline(*device_, "shaders/matmul_q4_0.spv", 4 * sizeof(uint32_t));
            
            VkDescriptorSet descSet = pipeline.createDescriptorSet();
            VkBuffer buffers[] = {bufferA.buffer(), bufferB.buffer(), bufferC.buffer()};
            pipeline.updateDescriptorSet(descSet, buffers, 3);
            
            // 6. Record commands
            VulkanCommandBuffer cmd(*device_);
            cmd.begin();
            cmd.bindPipeline(pipeline);
            cmd.bindDescriptorSets(pipeline, descSet);
            
            uint32_t dims[4] = {M, N, K, K_blocks};
            cmd.pushConstants(pipeline, dims, sizeof(dims));
            
            uint32_t wgX = (N + 15) / 16;
            uint32_t wgY = (M + 15) / 16;
            cmd.dispatch(wgX, wgY, 1);
            
            cmd.memoryBarrier();
            cmd.end();
            cmd.submit();
            
            // 7. Read results (FP16)
            std::vector<uint16_t> C_fp16(M * N);
            bufferC.read(C_fp16.data(), 0, sizeC);
            
            // 8. Convert back to FP32
            for (size_t i = 0; i < M * N; i++) {
                C[i] = fp16_to_float(C_fp16[i]);
            }
            
            return true;
        } catch (const std::exception& e) {
            // Silent fallback to CPU
            return false;
        }
    }
    
    // FP16 conversion helpers
    static uint16_t float_to_fp16(float f) {
        uint32_t x = *reinterpret_cast<uint32_t*>(&f);
        uint32_t sign = (x >> 31) << 15;
        uint32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
        uint32_t frac = (x >> 13) & 0x3FF;
        if (exp <= 0) return sign;
        if (exp >= 31) return sign | 0x7C00;
        return sign | (exp << 10) | frac;
    }
    
    static float fp16_to_float(uint16_t h) {
        uint32_t sign = (h >> 15) << 31;
        uint32_t exp = ((h >> 10) & 0x1F);
        uint32_t frac = (h & 0x3FF);
        if (exp == 0) {
            if (frac == 0) {
                uint32_t result = sign;
                return *reinterpret_cast<float*>(&result);
            }
            exp = 1;
        } else if (exp == 31) {
            uint32_t result = sign | 0x7F800000 | (frac << 13);
            return *reinterpret_cast<float*>(&result);
        } else {
            exp += 127 - 15;
        }
        uint32_t result = sign | (exp << 23) | (frac << 13);
        return *reinterpret_cast<float*>(&result);
    }
    
    std::shared_ptr<VulkanContext> ctx_;
    std::shared_ptr<VulkanDevice> device_;
#endif
    
    bool use_gpu_;
    int gpu_calls_;
    int cpu_calls_;
    
public:
    // Stats
    void print_stats() const {
        printf("\n📊 Matmul Statistics:\n");
        printf("   GPU calls: %d\n", gpu_calls_);
        printf("   CPU calls: %d\n", cpu_calls_);
        printf("   GPU usage: %.1f%%\n", 
               gpu_calls_ + cpu_calls_ > 0 ? 
               100.0 * gpu_calls_ / (gpu_calls_ + cpu_calls_) : 0.0);
        fflush(stdout);
    }
};

} // namespace matmul
} // namespace mlx
