// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

/**
 * Vector Addition Example using Vulkan Compute
 * 
 * This example demonstrates:
 * - Vulkan context initialization
 * - Buffer creation and management
 * - Compute shader execution
 * - Result verification
 */

#include "mlx/backend/vulkan/vulkan_context.h"
#include "mlx/backend/vulkan/vulkan_device.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace mlx::backend::vulkan;

// Simple vector addition on CPU for verification
void cpu_vector_add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    std::cout << "MLX-ARM Vector Addition Example\n";
    std::cout << "================================\n\n";
    
    // Vector size
    const size_t N = 1024;
    
    // Create input vectors
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c_cpu(N);
    std::vector<float> c_gpu(N);
    
    // Initialize input vectors
    for (size_t i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    
    std::cout << "Input vectors size: " << N << " elements\n";
    
    // Compute on CPU for reference
    std::cout << "\n[CPU] Computing vector addition...\n";
    cpu_vector_add(a.data(), b.data(), c_cpu.data(), N);
    std::cout << "[CPU] First 5 results: ";
    for (int i = 0; i < 5; i++) {
        std::cout << c_cpu[i] << " ";
    }
    std::cout << "\n";
    
    // Initialize Vulkan
    std::cout << "\n[Vulkan] Initializing...\n";
    
    auto context = std::make_shared<VulkanContext>();
    if (!context->initialize(true)) {  // Enable validation for demo
        std::cerr << "Failed to initialize Vulkan context\n";
        return 1;
    }
    
    std::cout << "[Vulkan] Context created\n";
    
    // Find compute device
    VkPhysicalDevice physical_device = context->find_best_compute_device();
    auto properties = context->get_device_properties(physical_device);
    std::cout << "[Vulkan] Using device: " << properties.deviceName << "\n";
    std::cout << "[Vulkan] Vendor ID: 0x" << std::hex << properties.vendorID << std::dec << "\n";
    std::cout << "[Vulkan] Device type: " << properties.deviceType << "\n";
    
    // Create logical device
    auto device = std::make_shared<VulkanDevice>(context, physical_device);
    if (!device->initialize()) {
        std::cerr << "Failed to initialize Vulkan device\n";
        return 1;
    }
    
    std::cout << "[Vulkan] Logical device created\n";
    std::cout << "[Vulkan] Compute queue family: " << device->compute_queue_family() << "\n";
    
    // TODO: In next iteration, we'll add:
    // - Buffer creation (VkBuffer for a, b, c)
    // - Load and compile vector_add.spv shader
    // - Create compute pipeline
    // - Record and submit command buffer
    // - Copy results back to CPU
    // - Verify results match CPU computation
    
    std::cout << "\n[Note] Full compute pipeline implementation coming in next iteration\n";
    std::cout << "[Note] For now, this demonstrates Vulkan initialization and device setup\n";
    
    // Cleanup
    device->wait_idle();
    
    std::cout << "\n[Vulkan] Cleanup complete\n";
    std::cout << "\n✓ Example completed successfully!\n";
    
    return 0;
}
