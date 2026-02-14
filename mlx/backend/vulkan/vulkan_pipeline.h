// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>

namespace mlx::backend::vulkan {

class VulkanDevice;

/**
 * Vulkan compute pipeline wrapper.
 * Handles shader loading, descriptor sets, and pipeline creation.
 */
class VulkanPipeline {
public:
    VulkanPipeline(
        VulkanDevice& device,
        const std::string& shaderName,
        uint32_t pushConstantSize = 0
    );

    ~VulkanPipeline();

    // Non-copyable
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;

    // Getters
    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout descriptor_set_layout() const { return descriptor_set_layout_; }
    VkDescriptorPool descriptor_pool() const { return descriptor_pool_; }

    VkDescriptorSet createDescriptorSet();
    
    void updateDescriptorSet(
        VkDescriptorSet descriptorSet,
        const std::vector<VkBuffer>& buffers
    );

private:
    void createShaderModule(const std::vector<uint32_t>& spirv);
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createPipelineLayout();
    void createPipeline();
    std::vector<uint32_t> loadShader(const std::string& name);

    VulkanDevice& device_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    uint32_t push_constant_size_ = 0;
};

} // namespace mlx::backend::vulkan
