// Simple Vulkan test - verify GPU compute works
// If this works, problem is in MLXMatmul, not Vulkan itself

#ifdef MLX_BUILD_VULKAN
#include "mlx/backend/vulkan/vulkan_context.h"
#include "mlx/backend/vulkan/vulkan_device.h"
#include "mlx/backend/vulkan/vulkan_buffer.h"
#include "mlx/backend/vulkan/vulkan_pipeline.h"
#include "mlx/backend/vulkan/vulkan_command.h"
#endif

#include <iostream>
#include <vector>
#include <cstdio>

#ifdef MLX_BUILD_VULKAN
using namespace mlx::backend::vulkan;
#endif

int main() {
    printf("🔬 Simple Vulkan GPU Test\n");
    printf("========================\n\n");
    
#ifdef MLX_BUILD_VULKAN
    try {
        printf("1️⃣  Creating Vulkan context...\n");
        auto ctx = std::make_shared<VulkanContext>();
        if (!ctx->initialize(false)) {
            printf("❌ Failed to initialize Vulkan context\n");
            return 1;
        }
        printf("✅ Vulkan context created\n\n");
        
        printf("2️⃣  Checking physical devices...\n");
        printf("   Device count: %d\n", ctx->physical_device_count());
        if (ctx->physical_device_count() == 0) {
            printf("❌ No Vulkan devices found\n");
            return 1;
        }
        printf("   Device name: %s\n", ctx->device_name().c_str());
        printf("   FP16 support: %s\n", ctx->supports_fp16() ? "Yes" : "No");
        printf("   Subgroups: %s\n", ctx->supports_subgroups() ? "Yes" : "No");
        printf("✅ Physical device OK\n\n");
        
        printf("3️⃣  Creating logical device...\n");
        auto physical_device = ctx->physical_device(0);
        auto device = std::make_shared<VulkanDevice>(ctx, physical_device);
        if (!device->initialize()) {
            printf("❌ Failed to initialize device\n");
            return 1;
        }
        printf("✅ Logical device created\n\n");
        
        printf("4️⃣  Testing simple compute (vector add)...\n");
        const int N = 1024;
        std::vector<float> a(N, 1.0f);
        std::vector<float> b(N, 2.0f);
        std::vector<float> c(N, 0.0f);
        
        // Create buffers (DeviceLocal for GPU compute)
        size_t size = N * sizeof(float);
        auto bufA = std::make_shared<VulkanBuffer>(*device, size, VulkanBuffer::Type::DeviceLocal);
        auto bufB = std::make_shared<VulkanBuffer>(*device, size, VulkanBuffer::Type::DeviceLocal);
        auto bufC = std::make_shared<VulkanBuffer>(*device, size, VulkanBuffer::Type::DeviceLocal);
        
        bufA->upload(a.data(), size);
        bufB->upload(b.data(), size);
        
        // Load pipeline (vector_add.spv should be in shaders/)
        VulkanPipeline pipeline(*device, "shaders/vector_add.spv", sizeof(uint32_t));
        pipeline.bind_buffer(0, bufA);
        pipeline.bind_buffer(1, bufB);
        pipeline.bind_buffer(2, bufC);
        
        uint32_t n = N;
        pipeline.set_push_constants(&n, sizeof(n));
        
        // Dispatch compute
        uint32_t workgroups = (N + 255) / 256;
        VulkanCommand cmd(*device);
        cmd.begin();
        cmd.bind_pipeline(pipeline);
        cmd.dispatch(workgroups, 1, 1);
        cmd.end();
        
        cmd.submit_and_wait();
        
        // Download result
        bufC->download(c.data(), size);
        
        // Verify
        bool success = true;
        for (int i = 0; i < N; i++) {
            if (std::abs(c[i] - 3.0f) > 1e-5f) {
                printf("❌ Mismatch at index %d: got %.3f, expected 3.0\n", i, c[i]);
                success = false;
                break;
            }
        }
        
        if (success) {
            printf("✅ Compute test PASSED! GPU is working!\n\n");
            printf("🎉 Vulkan GPU is fully functional!\n");
            printf("   Ready for LLM inference with GPU acceleration.\n");
            return 0;
        } else {
            printf("❌ Compute test FAILED\n");
            return 1;
        }
        
    } catch (const std::exception& e) {
        printf("❌ Exception: %s\n", e.what());
        return 1;
    } catch (...) {
        printf("❌ Unknown exception\n");
        return 1;
    }
#else
    printf("❌ Vulkan backend not enabled in build\n");
    return 1;
#endif
}
