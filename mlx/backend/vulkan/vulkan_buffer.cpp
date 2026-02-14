// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/backend/vulkan/vulkan_buffer.h"
#include "mlx/backend/vulkan/vulkan_device.h"
#include <stdexcept>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "MLX-VulkanBuffer"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#endif

namespace mlx::backend::vulkan {

VulkanBuffer::VulkanBuffer(VulkanDevice& device, size_t size, Type type, void* existing_ptr)
    : device_(device), size_(size), type_(type) {
    
    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_.device(), &buffer_info, nullptr, &buffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan buffer");
    }

    allocate_memory(existing_ptr);
}

VulkanBuffer::~VulkanBuffer() {
    if (mapped_data_ != nullptr) {
        vkUnmapMemory(device_.device(), memory_);
    }
    if (memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_.device(), memory_, nullptr);
    }
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_.device(), buffer_, nullptr);
    }
}

void VulkanBuffer::allocate_memory(void* existing_ptr) {
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_.device(), buffer_, &mem_requirements);

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    
    uint32_t properties = 0;
    if (type_ == Type::Device) {
        properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
        properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    
    alloc_info.memoryTypeIndex = device_.find_memory_type(mem_requirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device_.device(), &alloc_info, nullptr, &memory_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate Vulkan memory");
    }

    vkBindBufferMemory(device_.device(), buffer_, memory_, 0);
    
    if (type_ != Type::Device) {
        vkMapMemory(device_.device(), memory_, 0, size_, 0, &mapped_data_);
        if (existing_ptr) {
            memcpy(mapped_data_, existing_ptr, size_);
        }
    }
}

void VulkanBuffer::write(const void* data, size_t offset, size_t size) {
    if (mapped_data_) {
        memcpy((uint8_t*)mapped_data_ + offset, data, size);
    } else {
        // For device-only memory, we would need a staging buffer and a command
        throw std::runtime_error("Direct write only supported for host-visible buffers");
    }
}

void VulkanBuffer::read(void* data, size_t offset, size_t size) {
    if (mapped_data_) {
        memcpy(data, (uint8_t*)mapped_data_ + offset, size);
    } else {
        throw std::runtime_error("Direct read only supported for host-visible buffers");
    }
}

} // namespace mlx::backend::vulkan