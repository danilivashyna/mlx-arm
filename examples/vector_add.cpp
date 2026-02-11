// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

/**
 * Simple Vulkan Compute Example - Vector Addition
 * 
 * This example demonstrates:
 * - Vulkan context initialization
 * - Buffer allocation
 * - Compute shader dispatch
 * - Result readback
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
#include <climits>

#ifdef MLX_BUILD_VULKAN
using namespace mlx::backend::vulkan;
#endif

// Global Vulkan context to avoid reinitialization overhead
#ifdef MLX_BUILD_VULKAN
static std::shared_ptr<VulkanContext> g_ctx;
static std::shared_ptr<VulkanDevice> g_device;
static bool g_vulkan_initialized = false;

bool init_vulkan_once() {
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
        std::cout << "✅ Vulkan initialized: " << g_ctx->device_name() << std::endl;
        return true;
    } catch (...) {
        return false;
    }
}
#endif

bool run_vector_add_gpu(const std::vector<float>& a, const std::vector<float>& b, 
                        std::vector<float>& c, bool& gpu_available) {
#ifdef MLX_BUILD_VULKAN
    try {
        if (!g_vulkan_initialized && !init_vulkan_once()) {
            gpu_available = false;
            return false;
        }
        
        const size_t N = a.size();
        const size_t buffer_size = N * sizeof(float);
        
        // Initialize Vulkan (suppress verbose output for benchmarks)
        auto ctx = g_ctx;
        auto device = g_device;
        
        // Create buffers
        VulkanBuffer bufferA(*device, buffer_size, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferB(*device, buffer_size, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferC(*device, buffer_size, VulkanBuffer::Type::Staging);
        
        // Upload data
        bufferA.write(a.data(), 0, buffer_size);
        bufferB.write(b.data(), 0, buffer_size);
        
        // Load compute pipeline
        std::string shader_path = "shaders/vector_add.spv";
        VulkanPipeline pipeline(*device, shader_path, sizeof(uint32_t));
        
        // Create descriptor set
        VkDescriptorSet descSet = pipeline.createDescriptorSet();
        VkBuffer buffers[] = {bufferA.buffer(), bufferB.buffer(), bufferC.buffer()};
        pipeline.updateDescriptorSet(descSet, buffers, 3);
        
        // Record and execute compute commands
        VulkanCommandBuffer cmd(*device);
        cmd.begin();
        
        cmd.bindPipeline(pipeline);
        cmd.bindDescriptorSets(pipeline, descSet);
        
        uint32_t size = static_cast<uint32_t>(N);
        cmd.pushConstants(pipeline, &size, sizeof(uint32_t));
        
        // Dispatch
        uint32_t workgroups = (static_cast<uint32_t>(N) + 255) / 256;
        cmd.dispatch(workgroups);
        
        cmd.memoryBarrier();
        cmd.end();
        cmd.submit();
        
        // Read back results
        bufferC.read(c.data(), 0, buffer_size);
        
        gpu_available = true;
        return true;
        
    } catch (const std::exception& e) {
        // Silently fail for benchmarks
        gpu_available = false;
        return false;
    }
#else
    gpu_available = false;
    return false;
#endif
}

void run_benchmark(size_t N) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "📊 Benchmark: " << N << " elements (" << (N * sizeof(float) / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Prepare test data
    std::vector<float> a(N), b(N), c_gpu(N), c_cpu(N);
    
    for (size_t i = 0; i < N; i++) {
        a[i] = static_cast<float>(i % 1000);
        b[i] = static_cast<float>((i * 2) % 1000);
    }
    
    // Warmup GPU (first run is slower)
    bool gpu_available = false;
    run_vector_add_gpu(a, b, c_gpu, gpu_available);
    
    // GPU benchmark (3 runs, take best)
    long long best_gpu_time = LLONG_MAX;
    bool gpu_success = false;
    for (int run = 0; run < 3; run++) {
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_success = run_vector_add_gpu(a, b, c_gpu, gpu_available);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_time = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
        if (gpu_success && gpu_time < best_gpu_time) {
            best_gpu_time = gpu_time;
        }
    }
    
    // CPU benchmark (3 runs, take best)
    long long best_cpu_time = LLONG_MAX;
    for (int run = 0; run < 3; run++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; i++) {
            c_cpu[i] = a[i] + b[i];
        }
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
        if (cpu_time < best_cpu_time) {
            best_cpu_time = cpu_time;
        }
    }
    
    // Verify
    bool correct = true;
    if (gpu_success) {
        for (size_t i = 0; i < N; i++) {
            if (std::abs(c_gpu[i] - c_cpu[i]) > 1e-5) {
                correct = false;
                std::cerr << "❌ Mismatch at " << i << ": GPU=" << c_gpu[i] << " CPU=" << c_cpu[i] << std::endl;
                break;
            }
        }
    }
    
    // Results
    std::cout << "\n⚡ Performance Results (best of 3 runs):" << std::endl;
    if (gpu_success) {
        std::cout << "  GPU (Adreno 740): " << best_gpu_time << " µs" << std::endl;
        std::cout << "  CPU (NEON):       " << best_cpu_time << " µs" << std::endl;
        
        if (best_cpu_time > 0 && best_gpu_time > 0) {
            double speedup = static_cast<double>(best_cpu_time) / best_gpu_time;
            std::cout << "\n🚀 GPU Speedup: " << std::fixed << std::setprecision(2) << speedup << "x ";
            if (speedup > 1.0) {
                std::cout << "FASTER ✅" << std::endl;
            } else {
                std::cout << "slower (overhead dominates)" << std::endl;
            }
        }
        
        // Throughput
        double gpu_gflops = (N / 1e9) / (best_gpu_time / 1e6);
        double cpu_gflops = (N / 1e9) / (best_cpu_time / 1e6);
        std::cout << "\n📈 Throughput:" << std::endl;
        std::cout << "  GPU: " << std::fixed << std::setprecision(3) << gpu_gflops << " GFLOPS" << std::endl;
        std::cout << "  CPU: " << std::fixed << std::setprecision(3) << cpu_gflops << " GFLOPS" << std::endl;
        
        std::cout << "\n✅ Accuracy: " << (correct ? "100% CORRECT" : "FAILED") << std::endl;
    } else {
        std::cout << "  CPU (NEON):       " << best_cpu_time << " µs" << std::endl;
        std::cout << "  ⚠️  GPU unavailable" << std::endl;
    }
}

bool run_vector_add_example() {
    std::cout << "=== MLX-ARM Vector Addition Benchmark ===" << std::endl;
    std::cout << "Device: Galaxy Fold 5 (Snapdragon 8 Gen 2)" << std::endl;
    std::cout << "GPU: Adreno 740, CPU: ARM Cortex-X3 + A720 + A710" << std::endl;
    
    // Run benchmarks with different sizes (bigger to show GPU advantage)
    run_benchmark(1024);          // 4 KB
    run_benchmark(10240);         // 40 KB
    run_benchmark(102400);        // 400 KB
    run_benchmark(1024000);       // 4 MB
    run_benchmark(10240000);      // 40 MB
    
    return true;
}

int main() {
    try {
        if (run_vector_add_example()) {
            std::cout << "\n🎉 Demo completed successfully!" << std::endl;
            return 0;
        } else {
            std::cout << "\n❌ Demo failed!" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
