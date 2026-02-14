// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <vulkan/vulkan.h>
#include <cstddef>
#include <memory>

namespace mlx::backend::vulkan {

class VulkanDevice;

/**
 * RAII wrapper for Vulkan buffer and device memory.
 * Handles allocation, mapping, and transfer for GPU compute.
 */
class VulkanBuffer {
public:
    enum class Type {
        Staging,      // Host-visible, CPU -> GPU transfer
        Device,       // GPU-only, fast compute access
        Uniform       // Small constant data
    };

    /**
     * Create buffer with automatic memory allocation.
     * @param device Vulkan device
     * @param size Size in bytes
     * @param type Buffer type
     * @param existing_ptr Optional data to copy after allocation
     */
    VulkanBuffer(VulkanDevice& device, size_t size, Type type, void* existing_ptr = nullptr);

    ~VulkanBuffer();

    // Non-copyable
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    // Getters
    VkBuffer buffer() const { return buffer_; }
    VkDeviceMemory memory() const { return memory_; }
    size_t size() const { return size_; }
    Type type() const { return type_; }
    void* mapped_data() const { return mapped_data_; }

    /**
     * Write data to buffer.
     * @param data Source data
     * @param offset Offset in bytes
     * @param size Size in bytes
     */
    void write(const void* data, size_t offset = 0, size_t size = 0);

    /**
     * Read data from buffer.
     * @param data Destination buffer
     * @param offset Offset in bytes
     * @param size Size in bytes
     */
    void read(void* data, size_t offset = 0, size_t size = 0);

private:
    void allocate_memory(void* existing_ptr);

    VulkanDevice& device_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    size_t size_ = 0;
    Type type_;
    void* mapped_data_ = nullptr;
};

} // namespace mlx::backend::vulkan
