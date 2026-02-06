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
    std::vector<float> C_gpu_tiled(M * N);
    
    // Initialize with random values
    for (size_t i = 0; i < A.size(); i++) A[i] = (rand() % 100) / 10.0f;
    for (size_t i = 0; i < B.size(); i++) B[i] = (rand() % 100) / 10.0f;
    
#ifdef MLX_BUILD_VULKAN
    // Warmup GPU
    matmul_gpu(A.data(), B.data(), C_gpu_tiled.data(), M, K, N, true);
    
    // GPU Naive benchmark
    auto gpu_naive_start = std::chrono::high_resolution_clock::now();
    bool gpu_naive_ok = matmul_gpu(A.data(), B.data(), C_gpu_naive.data(), M, K, N, false);
    auto gpu_naive_end = std::chrono::high_resolution_clock::now();
    auto gpu_naive_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_naive_end - gpu_naive_start).count();
    
    // GPU Tiled benchmark
    auto gpu_tiled_start = std::chrono::high_resolution_clock::now();
    bool gpu_tiled_ok = matmul_gpu(A.data(), B.data(), C_gpu_tiled.data(), M, K, N, true);
    auto gpu_tiled_end = std::chrono::high_resolution_clock::now();
    auto gpu_tiled_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_tiled_end - gpu_tiled_start).count();
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
        std::cout << "  GPU Naive (Adreno):   " << std::setw(8) << gpu_naive_us << " µs  →  " 
                  << std::fixed << std::setprecision(2) << (total_flops / gpu_naive_us / 1000.0) << " GFLOPS" << std::endl;
    }
    if (gpu_tiled_ok) {
        std::cout << "  GPU Tiled (Adreno):   " << std::setw(8) << gpu_tiled_us << " µs  →  " 
                  << std::fixed << std::setprecision(2) << (total_flops / gpu_tiled_us / 1000.0) << " GFLOPS" << std::endl;
        
        double speedup_cpu = static_cast<double>(cpu_us) / gpu_tiled_us;
        std::cout << "\n🚀 GPU Speedup over CPU: " << std::fixed << std::setprecision(2) << speedup_cpu << "x ";
        if (speedup_cpu > 1.0) {
            std::cout << "FASTER ✅✅✅" << std::endl;
        } else {
            std::cout << "(CPU still wins)" << std::endl;
        }
    }
    
    // Verify correctness
    if (gpu_tiled_ok) {
        float max_error = 0.0f;
        for (size_t i = 0; i < C_cpu.size(); i++) {
            float error = std::abs(C_cpu[i] - C_gpu_tiled[i]);
            max_error = std::max(max_error, error);
        }
        std::cout << "\n✅ Accuracy: Max error = " << std::scientific << max_error;
        if (max_error < 0.01f) {
            std::cout << " (CORRECT)" << std::endl;
        } else {
            std::cout << " (FAILED)" << std::endl;
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
