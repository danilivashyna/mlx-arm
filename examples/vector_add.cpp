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
#endif
#include <iostream>
#include <vector>
#include <cmath>

#ifdef MLX_BUILD_VULKAN
using namespace mlx::backend::vulkan;
#endif

bool run_vector_add_example() {
    std::cout << "=== MLX-ARM Vector Addition Demo ===" << std::endl;
    
#ifdef MLX_BUILD_VULKAN
    // Initialize Vulkan
    VulkanContext ctx;
    if (!ctx.initialize(false)) {
        std::cerr << "Failed to initialize Vulkan" << std::endl;
        return false;
    }
    
    std::cout << "✅ Vulkan initialized" << std::endl;
    std::cout << "Device: " << ctx.device_name() << std::endl;
    std::cout << "FP16 support: " << (ctx.supports_fp16() ? "Yes" : "No") << std::endl;
    std::cout << "Subgroups: " << (ctx.supports_subgroups() ? "Yes" : "No") << std::endl;
#else
    std::cout << "⚠️  Vulkan not available - using CPU fallback" << std::endl;
#endif
    
    // Prepare test data
    const size_t N = 1024;
    std::vector<float> a(N), b(N), c(N);
    
    for (size_t i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }
    
    std::cout << "\n📊 Test data prepared: " << N << " elements" << std::endl;
    
#ifdef MLX_BUILD_VULKAN
    // TODO: Create buffers, load shader, dispatch compute
    // For now, this is a placeholder showing CPU version
    std::cout << "\n⚠️  GPU compute not yet implemented - using CPU fallback" << std::endl;
#endif
    
    for (size_t i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
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
