// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>
#include <vector>

namespace mlx::backend::vulkan {

class VulkanDevice;
class VulkanPipeline;

/**
 * Vulkan command buffer recording helper.
 */
class VulkanCommand {
public:
    explicit VulkanCommand(VulkanDevice& device);
    ~VulkanCommand();

    void begin();
    void end();
    void submit();

    void bindPipeline(const VulkanPipeline& pipeline);
    void bindDescriptorSet(const VulkanPipeline& pipeline, VkDescriptorSet descriptorSet);
    
    void pushConstants(
        const VulkanPipeline& pipeline,
        const void* data,
        uint32_t size
    );

    void dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);
    void memoryBarrier();
    
    VkCommandBuffer handle() const { return cmd_buffer_; }

private:
    VulkanDevice& device_;
    VkCommandBuffer cmd_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
};

} // namespace mlx::backend::vulkan
