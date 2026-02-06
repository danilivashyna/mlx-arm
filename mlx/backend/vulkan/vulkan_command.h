#pragma once

#include <vulkan/vulkan.h>
#include <functional>

namespace mlx {
namespace backend {
namespace vulkan {

class VulkanDevice;
class VulkanPipeline;

/**
 * Vulkan command buffer recording helper.
 * Simplifies compute dispatch operations.
 */
class VulkanCommandBuffer {
public:
    explicit VulkanCommandBuffer(const VulkanDevice& device);
    ~VulkanCommandBuffer();

    // Non-copyable
    VulkanCommandBuffer(const VulkanCommandBuffer&) = delete;
    VulkanCommandBuffer& operator=(const VulkanCommandBuffer&) = delete;

    /**
     * Begin recording commands.
     */
    void begin();

    /**
     * End recording.
     */
    void end();

    /**
     * Submit and execute on GPU, then wait for completion.
     * Blocks until GPU finishes.
     */
    void submit();

    /**
     * Bind compute pipeline.
     */
    void bindPipeline(const VulkanPipeline& pipeline);

    /**
     * Bind descriptor sets for buffer bindings.
     */
    void bindDescriptorSets(
        const VulkanPipeline& pipeline,
        VkDescriptorSet descriptorSet
    );

    /**
     * Push constants.
     */
    void pushConstants(
        const VulkanPipeline& pipeline,
        const void* data,
        uint32_t size
    );

    /**
     * Dispatch compute workgroups.
     * @param groupCountX Number of workgroups in X dimension
     * @param groupCountY Number of workgroups in Y dimension
     * @param groupCountZ Number of workgroups in Z dimension
     */
    void dispatch(uint32_t groupCountX, uint32_t groupCountY = 1, uint32_t groupCountZ = 1);

    /**
     * Add pipeline barrier for memory synchronization.
     * Use after dispatch to ensure GPU writes are visible.
     */
    void memoryBarrier();

    /**
     * Get command buffer handle.
     */
    VkCommandBuffer handle() const { return cmd_buffer_; }

private:
    const VulkanDevice* device_;
    VkCommandBuffer cmd_buffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    bool recording_ = false;
};

} // namespace vulkan
} // namespace backend
} // namespace mlx
