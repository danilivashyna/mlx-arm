#include "vulkan_buffer.h"
#include "vulkan_device.h"
#include "vulkan_context.h"
#include <cstring>
#include <stdexcept>

// Cross-platform logging macros
#ifdef __ANDROID__
    #include <android/log.h>
    #define LOGD(fmt, ...) __android_log_print(ANDROID_LOG_DEBUG, "MLX-Buffer", fmt, ##__VA_ARGS__)
    #define LOGI(fmt, ...) __android_log_print(ANDROID_LOG_INFO, "MLX-Buffer", fmt, ##__VA_ARGS__)
    #define LOGE(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, "MLX-Buffer", fmt, ##__VA_ARGS__)
#else
    #include <cstdio>
    #define LOGD(fmt, ...) do { printf("DEBUG: " fmt "\n", ##__VA_ARGS__); } while(0)
    #define LOGI(fmt, ...) do { printf("INFO: " fmt "\n", ##__VA_ARGS__); } while(0)
    #define LOGE(fmt, ...) do { fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__); } while(0)
#endif

namespace mlx {
namespace backend {
namespace vulkan {

VulkanBuffer::VulkanBuffer(
    const VulkanDevice& device,
    size_t size,
    Type type,
    VkBufferUsageFlags usage
) : device_(&device), size_(size), type_(type) {
    
    // Determine usage flags based on type
    VkBufferUsageFlags bufferUsage = usage;
    VkMemoryPropertyFlags memoryProperties = 0;
    
    switch (type) {
        case Type::Staging:
            bufferUsage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
            
        case Type::DeviceLocal:
            bufferUsage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
            
        case Type::Uniform:
            bufferUsage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
    }
    
    // Create buffer
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = bufferUsage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device.device(), &bufferInfo, nullptr, &buffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan buffer");
    }
    
    // Query memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device.device(), buffer_, &memRequirements);
    
    // Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, memoryProperties);
    
    if (vkAllocateMemory(device.device(), &allocInfo, nullptr, &memory_) != VK_SUCCESS) {
        vkDestroyBuffer(device.device(), buffer_, nullptr);
        throw std::runtime_error("Failed to allocate Vulkan buffer memory");
    }
    
    // Bind memory to buffer
    vkBindBufferMemory(device.device(), buffer_, memory_, 0);
    
    LOGD("Created Vulkan buffer: size=%zu type=%d", size, static_cast<int>(type));
}

VulkanBuffer::~VulkanBuffer() {
    cleanup();
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : device_(other.device_),
      buffer_(other.buffer_),
      memory_(other.memory_),
      size_(other.size_),
      type_(other.type_),
      mapped_(other.mapped_) {
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.mapped_ = nullptr;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
        cleanup();
        
        device_ = other.device_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_ = other.size_;
        type_ = other.type_;
        mapped_ = other.mapped_;
        
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.mapped_ = nullptr;
    }
    return *this;
}

void* VulkanBuffer::map() {
    if (type_ != Type::Staging && type_ != Type::Uniform) {
        throw std::runtime_error("Can only map staging/uniform buffers");
    }
    
    if (!mapped_) {
        if (vkMapMemory(device_->device(), memory_, 0, size_, 0, &mapped_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map buffer memory");
        }
    }
    
    return mapped_;
}

void VulkanBuffer::unmap() {
    if (mapped_) {
        vkUnmapMemory(device_->device(), memory_);
        mapped_ = nullptr;
    }
}

void VulkanBuffer::write(const void* data, size_t offset, size_t writeSize) {
    if (writeSize == 0) {
        writeSize = size_ - offset;
    }
    
    if (offset + writeSize > size_) {
        throw std::out_of_range("Write exceeds buffer size");
    }
    
    if (type_ == Type::Staging || type_ == Type::Uniform) {
        // Direct write for host-visible buffers
        void* mapped = map();
        std::memcpy(static_cast<char*>(mapped) + offset, data, writeSize);
        unmap();
    } else {
        // Device-local buffer: need staging buffer and command buffer
        // TODO: Implement staging buffer copy (requires command queue)
        throw std::runtime_error("Device-local buffer write requires command queue (not yet implemented)");
    }
}

void VulkanBuffer::read(void* data, size_t offset, size_t readSize) {
    if (readSize == 0) {
        readSize = size_ - offset;
    }
    
    if (offset + readSize > size_) {
        throw std::out_of_range("Read exceeds buffer size");
    }
    
    if (type_ == Type::Staging || type_ == Type::Uniform) {
        // Direct read for host-visible buffers
        void* mapped = map();
        std::memcpy(data, static_cast<char*>(mapped) + offset, readSize);
        unmap();
    } else {
        // Device-local buffer: need staging buffer and command buffer
        // TODO: Implement staging buffer copy (requires command queue)
        throw std::runtime_error("Device-local buffer read requires command queue (not yet implemented)");
    }
}

uint32_t VulkanBuffer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device_->physicalDevice(), &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    
    throw std::runtime_error("Failed to find suitable memory type");
}

void VulkanBuffer::cleanup() {
    if (device_) {
        unmap();
        
        if (buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_->device(), buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
        }
        
        if (memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_->device(), memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
    }
}

} // namespace vulkan
} // namespace backend
} // namespace mlx
