// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/backend/vulkan/vulkan_command.h"
#include "mlx/backend/vulkan/vulkan_device.h"
#include "mlx/backend/vulkan/vulkan_pipeline.h"
#include <stdexcept>

namespace mlx::backend::vulkan {

VulkanCommand::VulkanCommand(VulkanDevice& device) : device_(device) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = device.command_pool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(device.device(), &allocInfo, &cmd_buffer_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffer");
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(device.device(), &fenceInfo, nullptr, &fence_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fence");
    }
}

VulkanCommand::~VulkanCommand() {
    vkDestroyFence(device_.device(), fence_, nullptr);
    vkFreeCommandBuffers(device_.device(), device_.command_pool(), 1, &cmd_buffer_);
}

void VulkanCommand::begin() {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buffer_, &beginInfo);
}

void VulkanCommand::end() {
    vkEndCommandBuffer(cmd_buffer_);
}

void VulkanCommand::submit() {
    vkResetFences(device_.device(), 1, &fence_);
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd_buffer_;
    vkQueueSubmit(device_.compute_queue(), 1, &submitInfo, fence_);
    vkWaitForFences(device_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
}

void VulkanCommand::bindPipeline(const VulkanPipeline& pipeline) {
    vkCmdBindPipeline(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline());
}

void VulkanCommand::bindDescriptorSet(const VulkanPipeline& pipeline, VkDescriptorSet descriptorSet) {
    vkCmdBindDescriptorSets(cmd_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1, &descriptorSet, 0, nullptr);
}

void VulkanCommand::pushConstants(const VulkanPipeline& pipeline, const void* data, uint32_t size) {
    vkCmdPushConstants(cmd_buffer_, pipeline.layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data);
}

void VulkanCommand::dispatch(uint32_t gx, uint32_t gy, uint32_t gz) {
    vkCmdDispatch(cmd_buffer_, gx, gy, gz);
}

void VulkanCommand::memoryBarrier() {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    
    vkCmdPipelineBarrier(
        cmd_buffer_,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );
}

} // namespace mlx::backend::vulkan
