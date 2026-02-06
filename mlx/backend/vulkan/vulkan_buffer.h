#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>
#include <memory>

namespace mlx {
namespace backend {
namespace vulkan {

class VulkanDevice;

/**
 * RAII wrapper for Vulkan buffer and device memory.
 * Handles allocation, mapping, and transfer for GPU compute.
 */
class VulkanBuffer {
public:
    enum class Type {
        Staging,      // Host-visible, CPU → GPU transfer
        DeviceLocal,  // GPU-only, fast compute access
        Uniform       // Small constant data
    };

    /**
     * Create buffer with automatic memory allocation.
     * @param device Vulkan device
     * @param size Size in bytes
     * @param type Buffer type (staging/device/uniform)
     * @param usage Vulkan usage flags (additional to type defaults)
     */
    VulkanBuffer(
        const VulkanDevice& device,
        size_t size,
        Type type,
        VkBufferUsageFlags usage = 0
    );

    ~VulkanBuffer();

    // Non-copyable
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    // Movable
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;

    // Getters
    VkBuffer buffer() const { return buffer_; }
    VkDeviceMemory memory() const { return memory_; }
    size_t size() const { return size_; }
    Type type() const { return type_; }

    /**
     * Map buffer memory for CPU access.
     * Only works for staging buffers.
     * @return Pointer to mapped memory
     */
    void* map();

    /**
     * Unmap buffer memory.
     */
    void unmap();

    /**
     * Write data to buffer.
     * For staging buffers: direct write.
     * For device-local buffers: uses staging + command buffer copy.
     * @param data Source data
     * @param offset Offset in bytes
     * @param size Size in bytes (0 = all remaining)
     */
    void write(const void* data, size_t offset = 0, size_t size = 0);

    /**
     * Read data from buffer.
     * For staging buffers: direct read.
     * For device-local buffers: uses command buffer copy + staging.
     * @param data Destination buffer
     * @param offset Offset in bytes
     * @param size Size in bytes (0 = all remaining)
     */
    void read(void* data, size_t offset = 0, size_t size = 0);

private:
    const VulkanDevice* device_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    size_t size_ = 0;
    Type type_;
    void* mapped_ = nullptr;

    /**
     * Find suitable memory type index.
     * @param typeFilter Allowed memory types bitmask
     * @param properties Required properties
     * @return Memory type index
     */
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void cleanup();
};

} // namespace vulkan
} // namespace backend
} // namespace mlx
