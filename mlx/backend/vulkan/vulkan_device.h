// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vulkan/vulkan.h>

namespace mlx::backend::vulkan {

class VulkanContext;

/**
 * Vulkan logical device wrapper
 * Handles device creation, queues, and command pools
 */
class VulkanDevice {
public:
    VulkanDevice(std::shared_ptr<VulkanContext> context, VkPhysicalDevice physical_device);
    ~VulkanDevice();
    
    // Disable copy
    VulkanDevice(const VulkanDevice&) = delete;
    VulkanDevice& operator=(const VulkanDevice&) = delete;
    
    /**
     * Initialize logical device
     */
    bool initialize();
    
    /**
     * Get logical device handle
     */
    VkDevice device() const { return device_; }
    
    /**
     * Get physical device handle
     */
    VkPhysicalDevice physical_device() const { return physical_device_; }
    
    /**
     * Get compute queue
     */
    VkQueue compute_queue() const { return compute_queue_; }
    
    /**
     * Get compute queue family index
     */
    uint32_t compute_queue_family() const { return compute_queue_family_; }
    
    /**
     * Get command pool for compute operations
     */
    VkCommandPool command_pool() const { return command_pool_; }
    
    /**
     * Wait for all operations to complete
     */
    void wait_idle() const;
    
    /**
     * Get device properties
     */
    const VkPhysicalDeviceProperties& properties() const { return properties_; }
    
    /**
     * Get device features
     */
    const VkPhysicalDeviceFeatures& features() const { return features_; }

private:
    bool create_logical_device();
    bool create_command_pool();
    uint32_t find_compute_queue_family() const;
    
    std::shared_ptr<VulkanContext> context_;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    uint32_t compute_queue_family_ = 0;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    
    VkPhysicalDeviceProperties properties_;
    VkPhysicalDeviceFeatures features_;
};

}  // namespace mlx::backend::vulkan
