#include "vulkan_command.h"
#include "vulkan_device.h"
#include "vulkan_pipeline.h"
#include <stdexcept>

// Cross-platform logging
#ifdef __ANDROID__
    #include <android/log.h>
    #define LOGD(fmt, ...) __android_log_print(ANDROID_LOG_DEBUG, "MLX-Command", fmt, ##__VA_ARGS__)
    #define LOGE(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, "MLX-Command", fmt, ##__VA_ARGS__)
#else
    #include <cstdio>
    #define LOGD(fmt, ...) do { printf("DEBUG: " fmt "\n", ##__VA_ARGS__); } while(0)
    #define LOGE(fmt, ...) do { fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__); } while(0)
#endif

namespace mlx {
namespace backend {
namespace vulkan {

VulkanCommandBuffer::VulkanCommandBuffer(const VulkanDevice& device)
    : device_(&device) {
    
    // Allocate command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = device.command_pool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    
    if (vkAllocateCommandBuffers(device.device(), &allocInfo, &cmd_buffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer");
    }
    
    // Create fence for synchronization
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    
    if (vkCreateFence(device.device(), &fenceInfo, nullptr, &fence_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence");
    }
}

VulkanCommandBuffer::~VulkanCommandBuffer() {
    if (device_) {
        if (fence_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_->device(), fence_, nullptr);
        }
        if (cmd_buffer_ != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(device_->device(), device_->command_pool(), 1, &cmd_buffer_);
        }
    }
}

void VulkanCommandBuffer::begin() {
    if (recording_) {
        throw std::runtime_error("Command buffer already recording");
    }
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    if (vkBeginCommandBuffer(cmd_buffer_, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer");
    }
    
    recording_ = true;
}

void VulkanCommandBuffer::end() {
    if (!recording_) {
        throw std::runtime_error("Command buffer not recording");
    }
    
    if (vkEndCommandBuffer(cmd_buffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer");
    }
    
    recording_ = false;
}

void VulkanCommandBuffer::submit() {
    if (recording_) {
        throw std::runtime_error("Must end() before submit()");
    }
    
    // Reset fence
    vkResetFences(device_->device(), 1, &fence_);
    
    // Submit
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd_buffer_;
    
    if (vkQueueSubmit(device_->compute_queue(), 1, &submitInfo, fence_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit command buffer");
    }
    
    // Wait for completion
    vkWaitForFences(device_->device(), 1, &fence_, VK_TRUE, UINT64_MAX);
}

void VulkanCommandBuffer::bindPipeline(const VulkanPipeline& pipeline) {
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
}

void VulkanCommandBuffer::bindDescriptorSets(
    const VulkanPipeline& pipeline,
    VkDescriptorSet descriptorSet
) {
    vkCmdBindDescriptorSets(
        cmd_buffer_,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout(),
        0,
        1,
        &descriptorSet,
        0,
        nullptr
    );
}

void VulkanCommandBuffer::pushConstants(
    const VulkanPipeline& pipeline,
    const void* data,
    uint32_t size
) {
    vkCmdPushConstants(
        cmd_buffer_,
        pipeline.layout(),
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        size,
        data
    );
}

void VulkanCommandBuffer::dispatch(uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
    vkCmdDispatch(cmd_buffer_, groupCountX, groupCountY, groupCountZ);
}

void VulkanCommandBuffer::memoryBarrier() {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    
    vkCmdPipelineBarrier(
        cmd_buffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &barrier,
        0,
        nullptr,
        0,
        nullptr
    );
}

} // namespace vulkan
} // namespace backend
} // namespace mlx
