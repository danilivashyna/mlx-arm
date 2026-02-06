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

#ifdef MLX_BUILD_VULKAN
using namespace mlx::backend::vulkan;
#endif

bool run_vector_add_gpu(const std::vector<float>& a, const std::vector<float>& b, 
                        std::vector<float>& c, bool& gpu_available) {
#ifdef MLX_BUILD_VULKAN
    std::cout << "[DEBUG] Entering GPU path..." << std::endl;
    try {
        const size_t N = a.size();
        const size_t buffer_size = N * sizeof(float);
        
        // Initialize Vulkan
        std::cout << "[DEBUG] Creating Vulkan context..." << std::endl;
        auto ctx = std::make_shared<VulkanContext>();
        
        std::cout << "[DEBUG] Initializing context..." << std::endl;
        if (!ctx->initialize(false)) {
            std::cerr << "Failed to initialize Vulkan" << std::endl;
            gpu_available = false;
            return false;
        }
        
        std::cout << "✅ Vulkan initialized" << std::endl;
        std::cout << "Device: " << ctx->device_name() << std::endl;
        
        // Get device
        auto physical_device = ctx->physical_device(0);
        auto device = std::make_shared<VulkanDevice>(ctx, physical_device);
        if (!device->initialize()) {
            std::cerr << "Failed to initialize Vulkan device" << std::endl;
            gpu_available = false;
            return false;
        }
        
        std::cout << "✅ Logical device created" << std::endl;
        
        // Create buffers
        std::cout << "Creating buffers..." << std::endl;
        VulkanBuffer bufferA(*device, buffer_size, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferB(*device, buffer_size, VulkanBuffer::Type::Staging);
        VulkanBuffer bufferC(*device, buffer_size, VulkanBuffer::Type::Staging);
        
        // Upload data
        std::cout << "Uploading data..." << std::endl;
        bufferA.write(a.data(), 0, buffer_size);
        bufferB.write(b.data(), 0, buffer_size);
        
        std::cout << "✅ Buffers created and data uploaded" << std::endl;
        
        // Load compute pipeline - try both paths
        std::cout << "Loading shader..." << std::endl;
        std::string shader_path = "shaders/vector_add.spv";
        std::cout << "Trying: " << shader_path << std::endl;
        VulkanPipeline pipeline(*device, shader_path, sizeof(uint32_t));  // push constant: size
        
        std::cout << "✅ Compute pipeline created" << std::endl;
        
        // Create descriptor set
        VkDescriptorSet descSet = pipeline.createDescriptorSet();
        VkBuffer buffers[] = {bufferA.buffer(), bufferB.buffer(), bufferC.buffer()};
        pipeline.updateDescriptorSet(descSet, buffers, 3);
        
        std::cout << "✅ Descriptor sets configured" << std::endl;
        
        // Record and execute compute commands
        VulkanCommandBuffer cmd(*device);
        cmd.begin();
        
        cmd.bindPipeline(pipeline);
        cmd.bindDescriptorSets(pipeline, descSet);
        
        uint32_t size = static_cast<uint32_t>(N);
        cmd.pushConstants(pipeline, &size, sizeof(uint32_t));
        
        // Dispatch: workgroup size is 256, so need (N+255)/256 groups
        uint32_t workgroups = (static_cast<uint32_t>(N) + 255) / 256;
        cmd.dispatch(workgroups);
        
        cmd.memoryBarrier();
        cmd.end();
        
        std::cout << "🚀 Dispatching " << workgroups << " workgroups to GPU..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        cmd.submit();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "⚡ GPU execution time: " << duration << " µs" << std::endl;
        
        // Read back results
        bufferC.read(c.data(), 0, buffer_size);
        
        std::cout << "✅ Results retrieved from GPU" << std::endl;
        gpu_available = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ GPU execution failed: " << e.what() << std::endl;
        std::cerr.flush();
        gpu_available = false;
        return false;
    }
#else
    gpu_available = false;
    return false;
#endif
}

bool run_vector_add_example() {
    std::cout << "=== MLX-ARM Vector Addition Demo ===" << std::endl;
    
    // Prepare test data
    const size_t N = 1024;
    std::vector<float> a(N), b(N), c(N);
    
    for (size_t i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    
    std::cout << "\n📊 Test data prepared: " << N << " elements" << std::endl;
    
    // Try GPU first
    bool gpu_available = false;
    bool gpu_success = run_vector_add_gpu(a, b, c, gpu_available);
    
    if (!gpu_success) {
        std::cout << "\n⚠️  GPU path unavailable - using CPU fallback" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; i++) {
            c[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "⚡ CPU execution time: " << duration << " µs" << std::endl;
    }
    
    // Verify results
    bool correct = true;
    for (size_t i = 0; i < N; i++) {
        float expected = a[i] + b[i];
        if (std::abs(c[i] - expected) > 1e-5) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": " 
                      << c[i] << " != " << expected << std::endl;
            break;
        }
    }
    
    if (correct) {
        std::cout << "✅ Results verified correctly!" << std::endl;
        std::cout << "Sample: " << a[10] << " + " << b[10] << " = " << c[10] << std::endl;
    } else {
        std::cout << "❌ Results verification failed!" << std::endl;
        return false;
    }
    
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
