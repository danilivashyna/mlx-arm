#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>

namespace mlx {
namespace backend {
namespace vulkan {

class VulkanDevice;

/**
 * Vulkan compute pipeline wrapper.
 * Handles shader loading, descriptor sets, and pipeline creation.
 */
class VulkanPipeline {
public:
    /**
     * Create compute pipeline from SPIR-V shader.
     * @param device Vulkan device
     * @param spirvPath Path to .spv file
     * @param pushConstantSize Size of push constants (optional)
     */
    VulkanPipeline(
        const VulkanDevice& device,
        const std::string& spirvPath,
        uint32_t pushConstantSize = 0
    );

    ~VulkanPipeline();

    // Non-copyable
    VulkanPipeline(const VulkanPipeline&) = delete;
    VulkanPipeline& operator=(const VulkanPipeline&) = delete;

    // Movable
    VulkanPipeline(VulkanPipeline&& other) noexcept;
    VulkanPipeline& operator=(VulkanPipeline&& other) noexcept;

    // Getters
    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return layout_; }
    VkDescriptorSetLayout descriptorSetLayout() const { return descriptor_set_layout_; }
    VkDescriptorPool descriptorPool() const { return descriptor_pool_; }

    /**
     * Create descriptor set for buffer bindings.
     * Call this for each dispatch with different buffers.
     * @return Descriptor set handle
     */
    VkDescriptorSet createDescriptorSet();

    /**
     * Update descriptor set with buffer bindings.
     * @param descriptorSet Descriptor set to update
     * @param buffers Array of buffers to bind (must match shader layout)
     * @param bufferCount Number of buffers
     */
    void updateDescriptorSet(
        VkDescriptorSet descriptorSet,
        VkBuffer* buffers,
        size_t bufferCount
    );

private:
    const VulkanDevice* device_;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkShaderModule shader_module_ = VK_NULL_HANDLE;
    uint32_t push_constant_size_ = 0;

    /**
     * Load SPIR-V shader from file.
     * @param path File path
     * @return SPIR-V bytecode
     */
    std::vector<uint32_t> loadSPIRV(const std::string& path);

    /**
     * Create shader module from SPIR-V.
     */
    void createShaderModule(const std::vector<uint32_t>& spirv);

    /**
     * Create descriptor set layout.
     * Hardcoded for 3 storage buffers (a, b, result).
     */
    void createDescriptorSetLayout();

    /**
     * Create descriptor pool.
     */
    void createDescriptorPool();

    /**
     * Create pipeline layout.
     */
    void createPipelineLayout();

    /**
     * Create compute pipeline.
     */
    void createPipeline();

    void cleanup();
};

} // namespace vulkan
} // namespace backend
} // namespace mlx
