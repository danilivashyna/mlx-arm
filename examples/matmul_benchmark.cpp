// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

/**
 * Matrix Multiplication Benchmark - GPU vs CPU
 * 
 * Tests REAL GPU performance with compute-heavy workload.
 * Vector addition is memory-bound, GEMM is compute-bound.
 */

#ifdef MLX_BUILD_VULKAN
#include "mlx/backend/vulkan/vulkan_context.h"
#include "mlx/backend/vulkan/vulkan_device.h"
#include "mlx/backend/vulkan/vulkan_buffer.h"
#include "mlx/backend/vulkan/vulkan_pipeline.h"
#include "mlx/backend/vulkan/vulkan_command.h"
#endif
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include "quantize.h"

#ifdef MLX_BUILD_VULKAN
using namespace mlx::backend::vulkan;

// Global Vulkan state
static std::shared_ptr<VulkanContext> g_ctx;
static std::shared_ptr<VulkanDevice> g_device;
static bool g_vulkan_initialized = false;

bool init_vulkan() {
    if (g_vulkan_initialized) return true;
    
    try {
        g_ctx = std::make_shared<VulkanContext>();
        if (!g_ctx->initialize(false)) {
            return false;
        }
        
        auto physical_device = g_ctx->physical_device(0);
        g_device = std::make_shared<VulkanDevice>(g_ctx, physical_device);
        if (!g_device->initialize()) {
            return false;
        }
        
        g_vulkan_initialized = true;
        std::cout << "✅ Vulkan: " << g_ctx->device_name() << std::endl;
        std::cout << "   FP16: " << (g_ctx->supports_fp16() ? "Yes" : "No") << std::endl;
        std::cout << "   Subgroups: " << (g_ctx->supports_subgroups() ? "Yes" : "No") << std::endl;
        return true;
    } catch (...) {
        return false;
    }
}
#endif

// CPU naive matmul
void matmul_cpu(const float* A, const float* B, float* C, uint32_t M, uint32_t K, uint32_t N) {
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

// Transpose B для векторизованного доступа: B[K×N] → B_T[N×K]
void transpose_for_vectorized(const float* B, float* B_T, uint32_t K, uint32_t N) {
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t k = 0; k < K; k++) {
            B_T[n * K + k] = B[k * N + n];
        }
    }
}

#ifdef MLX_BUILD_VULKAN
bool matmul_gpu(const float* A, const float* B, float* C, 
                uint32_t M, uint32_t K, uint32_t N, 
                bool use_tiled) {
    try {
        if (!g_vulkan_initialized && !init_vulkan()) {
            return false;
        }
        
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);
        
        // Create buffers
        VulkanBuffer bufferA(*g_device, sizeA, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferB(*g_device, sizeB, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferC(*g_device, sizeC, VulkanBuffer::Type::Staging);
        
        // Upload data
        bufferA.write(A, 0, sizeA);
        bufferB.write(B, 0, sizeB);
        
        // Load pipeline
        std::string shader = use_tiled ? "shaders/matmul_tiled.spv" : "shaders/matmul_naive.spv";
        VulkanPipeline pipeline(*g_device, shader, 3 * sizeof(uint32_t));
        
        // Create descriptor set
        VkDescriptorSet descSet = pipeline.createDescriptorSet();
        VkBuffer buffers[] = {bufferA.buffer(), bufferB.buffer(), bufferC.buffer()};
        pipeline.updateDescriptorSet(descSet, buffers, 3);
        
        // Record commands
        VulkanCommandBuffer cmd(*g_device);
        cmd.begin();
        cmd.bindPipeline(pipeline);
        cmd.bindDescriptorSets(pipeline, descSet);
        
        // Push constants: M, K, N
        uint32_t dims[3] = {M, K, N};
        cmd.pushConstants(pipeline, dims, sizeof(dims));
        
        // Dispatch: (M+15)/16 × (N+15)/16 workgroups
        uint32_t wgX = (N + 15) / 16;
        uint32_t wgY = (M + 15) / 16;
        cmd.dispatch(wgX, wgY, 1);
        
        cmd.memoryBarrier();
        cmd.end();
        cmd.submit();
        
        // Read results
        bufferC.read(C, 0, sizeC);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "GPU error: " << e.what() << std::endl;
        return false;
    }
}

bool matmul_gpu_fp16(const float* A, const float* B, float* C, 
                     uint32_t M, uint32_t K, uint32_t N) {
    try {
        if (!g_vulkan_initialized && !init_vulkan()) {
            return false;
        }
        
        // Convert to FP16
        std::vector<uint16_t> A_fp16(M * K);
        std::vector<uint16_t> B_fp16(K * N);
        std::vector<uint16_t> C_fp16(M * N);
        
        for (size_t i = 0; i < M * K; i++) A_fp16[i] = float_to_fp16(A[i]);
        for (size_t i = 0; i < K * N; i++) B_fp16[i] = float_to_fp16(B[i]);
        
        size_t sizeA = M * K * sizeof(uint16_t);
        size_t sizeB = K * N * sizeof(uint16_t);
        size_t sizeC = M * N * sizeof(uint16_t);
        
        // Create buffers
        VulkanBuffer bufferA(*g_device, sizeA, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferB(*g_device, sizeB, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferC(*g_device, sizeC, VulkanBuffer::Type::Staging);
        
        // Upload data
        bufferA.write(A_fp16.data(), 0, sizeA);
        bufferB.write(B_fp16.data(), 0, sizeB);
        
        // Load FP16 pipeline
        VulkanPipeline pipeline(*g_device, "shaders/matmul_fp16.spv", 3 * sizeof(uint32_t));
        
        // Create descriptor set
        VkDescriptorSet descSet = pipeline.createDescriptorSet();
        VkBuffer buffers[] = {bufferA.buffer(), bufferB.buffer(), bufferC.buffer()};
        pipeline.updateDescriptorSet(descSet, buffers, 3);
        
        // Record commands
        VulkanCommandBuffer cmd(*g_device);
        cmd.begin();
        cmd.bindPipeline(pipeline);
        cmd.bindDescriptorSets(pipeline, descSet);
        
        // Push constants: M, K, N
        uint32_t dims[3] = {M, K, N};
        cmd.pushConstants(pipeline, dims, sizeof(dims));
        
        // Dispatch
        uint32_t wgX = (N + 15) / 16;
        uint32_t wgY = (M + 15) / 16;
        cmd.dispatch(wgX, wgY, 1);
        
        cmd.memoryBarrier();
        cmd.end();
        cmd.submit();
        
        // Read results
        bufferC.read(C_fp16.data(), 0, sizeC);
        
        // Convert back to FP32
        for (size_t i = 0; i < M * N; i++) {
            C[i] = fp16_to_float(C_fp16[i]);
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "GPU FP16 error: " << e.what() << std::endl;
        return false;
    }
}

bool matmul_gpu_vectorized(const float* A, const float* B, float* C,
                           uint32_t M, uint32_t K, uint32_t N) {
    try {
        if (!g_vulkan_initialized && !init_vulkan()) {
            return false;
        }
        
        // Transpose B для оптимального GPU access
        std::vector<float> B_T(N * K);
        transpose_for_vectorized(B, B_T.data(), K, N);
        
        // Padding до vec4 alignment
        uint32_t K_vec4 = (K + 3) / 4;
        uint32_t K_padded = K_vec4 * 4;
        
        std::vector<float> A_padded(M * K_padded, 0.0f);
        std::vector<float> B_T_padded(N * K_padded, 0.0f);
        
        // Copy с паддингом
        for (uint32_t i = 0; i < M; i++) {
            std::memcpy(&A_padded[i * K_padded], &A[i * K], K * sizeof(float));
        }
        for (uint32_t i = 0; i < N; i++) {
            std::memcpy(&B_T_padded[i * K_padded], &B_T[i * K], K * sizeof(float));
        }
        
        // Create buffers (vec4 layout)
        size_t sizeA = M * K_padded * sizeof(float);
        size_t sizeB = N * K_padded * sizeof(float);
        size_t sizeC = M * N * sizeof(float);
        
        VulkanBuffer bufferA(*g_device, sizeA, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferB(*g_device, sizeB, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferC(*g_device, sizeC, VulkanBuffer::Type::Staging);
        
        // Upload padded data
        bufferA.write(A_padded.data(), 0, sizeA);
        bufferB.write(B_T_padded.data(), 0, sizeB);
        
        // Load vectorized pipeline (4 push constants: M, N, K, K_vec4)
        VulkanPipeline pipeline(*g_device, "shaders/matmul_vectorized.spv", 4 * sizeof(uint32_t));
        
        // Create descriptor set
        VkDescriptorSet descSet = pipeline.createDescriptorSet();
        VkBuffer buffers[] = {bufferA.buffer(), bufferB.buffer(), bufferC.buffer()};
        pipeline.updateDescriptorSet(descSet, buffers, 3);
        
        // Record commands
        VulkanCommandBuffer cmd(*g_device);
        cmd.begin();
        cmd.bindPipeline(pipeline);
        cmd.bindDescriptorSets(pipeline, descSet);
        
        // Push constants: M, N, K, K_vec4
        uint32_t dims[4] = {M, N, K, K_vec4};
        cmd.pushConstants(pipeline, dims, sizeof(dims));
        
        // Dispatch
        uint32_t wgX = (N + 15) / 16;
        uint32_t wgY = (M + 15) / 16;
        cmd.dispatch(wgX, wgY, 1);
        
        cmd.memoryBarrier();
        cmd.end();
        cmd.submit();
        
        // Read results
        bufferC.read(C, 0, sizeC);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "GPU Vectorized error: " << e.what() << std::endl;
        return false;
    }
}

bool matmul_gpu_subgroup(const float* A, const float* B, float* C,
                         uint32_t M, uint32_t K, uint32_t N) {
    try {
        if (!g_vulkan_initialized && !init_vulkan()) {
            return false;
        }
        
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);
        
        // Create buffers
        VulkanBuffer bufferA(*g_device, sizeA, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferB(*g_device, sizeB, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferC(*g_device, sizeC, VulkanBuffer::Type::Staging);
        
        // Upload data
        bufferA.write(A, 0, sizeA);
        bufferB.write(B, 0, sizeB);
        
        // Load subgroup pipeline
        VulkanPipeline pipeline(*g_device, "shaders/matmul_subgroup.spv", 3 * sizeof(uint32_t));
        
        // Create descriptor set
        VkDescriptorSet descSet = pipeline.createDescriptorSet();
        VkBuffer buffers[] = {bufferA.buffer(), bufferB.buffer(), bufferC.buffer()};
        pipeline.updateDescriptorSet(descSet, buffers, 3);
        
        // Record commands
        VulkanCommandBuffer cmd(*g_device);
        cmd.begin();
        cmd.bindPipeline(pipeline);
        cmd.bindDescriptorSets(pipeline, descSet);
        
        // Push constants: M, K, N
        uint32_t dims[3] = {M, K, N};
        cmd.pushConstants(pipeline, dims, sizeof(dims));
        
        // Dispatch: M*N workgroups (each workgroup = 1 output element with 64 threads)
        uint32_t num_workgroups = M * N;
        cmd.dispatch(num_workgroups, 1, 1);
        
        cmd.memoryBarrier();
        cmd.end();
        cmd.submit();
        
        // Read results
        bufferC.read(C, 0, sizeC);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "GPU Subgroup error: " << e.what() << std::endl;
        return false;
    }
}

bool matmul_gpu_q4_0(const float* A, const float* B, float* C,
                     uint32_t M, uint32_t K, uint32_t N) {
    try {
        if (!g_vulkan_initialized && !init_vulkan()) {
            return false;
        }
        
        if (K % 32 != 0) {
            std::cerr << "Q4_0 requires K to be multiple of 32" << std::endl;
            return false;
        }
        
        // 1. Quantize B to Q4_0 format with CORRECT LAYOUT [N × K_blocks]
        std::vector<BlockQ4_0> B_q4 = quantize_q4_0_matrix(B, K, N);
        uint32_t K_blocks = K / 32;
        
        // 2. Convert A to FP16
        std::vector<uint16_t> A_fp16(M * K);
        for (size_t i = 0; i < M * K; i++) {
            A_fp16[i] = float_to_fp16(A[i]);
        }
        
        // 3. Create buffers
        size_t sizeA = M * K * sizeof(uint16_t);  // FP16
        size_t sizeB = B_q4.size() * sizeof(BlockQ4_0);  // Q4_0 blocks
        size_t sizeC = M * N * sizeof(uint16_t);  // FP16 output
        
        VulkanBuffer bufferA(*g_device, sizeA, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferB(*g_device, sizeB, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferC(*g_device, sizeC, VulkanBuffer::Type::Staging);
        
        // 4. Upload data
        bufferA.write(A_fp16.data(), 0, sizeA);
        bufferB.write(B_q4.data(), 0, sizeB);
        
        // 5. Load Q4_0 pipeline (4 push constants: M, N, K, K_blocks)
        VulkanPipeline pipeline(*g_device, "shaders/matmul_q4_0.spv", 4 * sizeof(uint32_t));
        
        // Create descriptor set
        VkDescriptorSet descSet = pipeline.createDescriptorSet();
        VkBuffer buffers[] = {bufferA.buffer(), bufferB.buffer(), bufferC.buffer()};
        pipeline.updateDescriptorSet(descSet, buffers, 3);
        
        // 6. Record commands
        VulkanCommandBuffer cmd(*g_device);
        cmd.begin();
        cmd.bindPipeline(pipeline);
        cmd.bindDescriptorSets(pipeline, descSet);
        
        // Push constants: M, N, K, K_blocks
        uint32_t dims[4] = {M, N, K, K_blocks};
        cmd.pushConstants(pipeline, dims, sizeof(dims));
        
        // Dispatch
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
        std::cerr << "GPU Q4_0 error: " << e.what() << std::endl;
        return false;
    }
}
#endif

void run_matmul_benchmark(uint32_t M, uint32_t K, uint32_t N) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "📊 Matrix Multiplication: [" << M << "×" << K << "] × [" << K << "×" << N << "]" << std::endl;
    std::cout << "   Result: [" << M << "×" << N << "] = " << (M * N * sizeof(float) / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "   FLOPs: " << (2.0 * M * N * K / 1e9) << " GFLOP" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Allocate matrices
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_cpu(M * N);
    std::vector<float> C_gpu_naive(M * N);
    std::vector<float> C_gpu_q4(M * N);
    
    // Initialize with random values
    for (size_t i = 0; i < A.size(); i++) A[i] = (rand() % 100) / 10.0f;
    for (size_t i = 0; i < B.size(); i++) B[i] = (rand() % 100) / 10.0f;
    
#ifdef MLX_BUILD_VULKAN
    // Warmup GPU
    if (K % 32 == 0) {
        matmul_gpu_q4_0(A.data(), B.data(), C_gpu_q4.data(), M, K, N);
    }
    
    // GPU Naive FP32 benchmark
    auto gpu_naive_start = std::chrono::high_resolution_clock::now();
    bool gpu_naive_ok = matmul_gpu(A.data(), B.data(), C_gpu_naive.data(), M, K, N, false);
    auto gpu_naive_end = std::chrono::high_resolution_clock::now();
    auto gpu_naive_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_naive_end - gpu_naive_start).count();
    
    // GPU Q4_0 benchmark (only if K is multiple of 32)
    bool gpu_q4_ok = false;
    auto gpu_q4_us = 0L;
    if (K % 32 == 0) {
        auto gpu_q4_start = std::chrono::high_resolution_clock::now();
        gpu_q4_ok = matmul_gpu_q4_0(A.data(), B.data(), C_gpu_q4.data(), M, K, N);
        auto gpu_q4_end = std::chrono::high_resolution_clock::now();
        gpu_q4_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_q4_end - gpu_q4_start).count();
    }
#endif
    
    // CPU benchmark
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matmul_cpu(A.data(), B.data(), C_cpu.data(), M, K, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    
    // Calculate GFLOPS
    double total_flops = 2.0 * M * N * K;
    
    std::cout << "\n⚡ Performance:" << std::endl;
    std::cout << "  CPU (NEON):           " << std::setw(8) << cpu_us << " µs  →  " 
              << std::fixed << std::setprecision(2) << (total_flops / cpu_us / 1000.0) << " GFLOPS" << std::endl;
    
#ifdef MLX_BUILD_VULKAN
    if (gpu_naive_ok) {
        std::cout << "  GPU FP32 Naive:       " << std::setw(8) << gpu_naive_us << " µs  →  " 
                  << std::fixed << std::setprecision(2) << (total_flops / gpu_naive_us / 1000.0) << " GFLOPS" << std::endl;
    }
    if (gpu_q4_ok) {
        double size_fp32 = (K * N * sizeof(float)) / 1024.0 / 1024.0;
        double size_q4 = ((K * N / 32) * sizeof(BlockQ4_0)) / 1024.0 / 1024.0;
        
        std::cout << "  GPU Q4_0 (4-bit):     " << std::setw(8) << gpu_q4_us << " µs  →  " 
                  << std::fixed << std::setprecision(2) << (total_flops / gpu_q4_us / 1000.0) << " GFLOPS 🔥" << std::endl;
        std::cout << "  Memory saved: " << std::fixed << std::setprecision(2) 
                  << size_fp32 << " MB → " << size_q4 << " MB (";
        std::cout << std::fixed << std::setprecision(1) << (size_fp32 / size_q4) << "x smaller)" << std::endl;
        
        double speedup_cpu = static_cast<double>(cpu_us) / gpu_q4_us;
        double speedup_naive = static_cast<double>(gpu_naive_us) / gpu_q4_us;
        std::cout << "\n🚀 Q4_0 Speedup vs CPU:   " << std::fixed << std::setprecision(2) << speedup_cpu << "x";
        std::cout << "\n🚀 Q4_0 Speedup vs Naive: " << std::fixed << std::setprecision(2) << speedup_naive << "x";
        if (speedup_naive > 1.2) {
            std::cout << " ✅✅✅" << std::endl;
        } else if (speedup_naive > 0.8) {
            std::cout << " ✅" << std::endl;
        } else {
            std::cout << std::endl;
        }
    }
    
    // Verify correctness
    if (gpu_q4_ok) {
        float max_error = 0.0f;
        for (size_t i = 0; i < C_cpu.size(); i++) {
            float error = std::abs(C_cpu[i] - C_gpu_q4[i]);
            max_error = std::max(max_error, error);
        }
        
        // Q4_0 has lower precision due to quantization
        float relative_error = max_error / (std::abs(C_cpu[0]) + 1e-6f);
        std::cout << "\n✅ Q4_0 Accuracy: Max error = " << std::scientific << max_error;
        std::cout << " (relative: " << std::fixed << std::setprecision(2) << (relative_error * 100.0f) << "%)";
        if (relative_error < 0.05f) { // Within 5% is acceptable for Q4_0
            std::cout << " (GOOD)" << std::endl;
        } else {
            std::cout << " (CHECK)" << std::endl;
        }
    }
#endif
}

int main() {
    std::cout << "=== MLX-ARM Matrix Multiplication Benchmark ===" << std::endl;
    std::cout << "Device: Galaxy Fold 5 (Snapdragon 8 Gen 2)" << std::endl;
    std::cout << "Testing: GEMM (General Matrix Multiply)" << std::endl;
    std::cout << "Arithmetic Intensity: O(N) — compute-heavy workload" << std::endl;
    std::cout << std::endl;
    
    try {
        // Small matrices
        run_matmul_benchmark(128, 128, 128);
        run_matmul_benchmark(256, 256, 256);
        
        // Medium matrices
        run_matmul_benchmark(512, 512, 512);
        
        // Large matrices - GPU should DOMINATE here
        run_matmul_benchmark(1024, 1024, 1024);
        run_matmul_benchmark(2048, 2048, 2048);
        
        std::cout << "\n🎉 Benchmark completed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
