// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace mlx::backend::vulkan {

/**
 * Vulkan instance and context management
 * Handles Vulkan initialization, validation layers, and extensions
 */
class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();
    
    // Disable copy
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    
    /**
     * Initialize Vulkan instance
     * @param enable_validation Enable validation layers for debugging
     * @return true if initialization succeeded
     */
    bool initialize(bool enable_validation = false);
    
    /**
     * Check if Vulkan is available on this system
     */
    static bool is_vulkan_available();
    
    /**
     * Get Vulkan instance
     */
    VkInstance instance() const { return instance_; }
    
    /**
     * Get available physical devices
     */
    std::vector<VkPhysicalDevice> get_physical_devices() const;
    
    /**
     * Get physical device properties
     */
    VkPhysicalDeviceProperties get_device_properties(VkPhysicalDevice device) const;
    
    /**
     * Get physical device features
     */
    VkPhysicalDeviceFeatures get_device_features(VkPhysicalDevice device) const;
    
    /**
     * Check if device supports compute
     */
    bool supports_compute(VkPhysicalDevice device) const;
    
    /**
     * Find best compute device
     */
    VkPhysicalDevice find_best_compute_device() const;
    
    /**
     * Get required instance extensions
     */
    static std::vector<const char*> get_required_instance_extensions();
    
    /**
     * Get required device extensions
     */
    static std::vector<const char*> get_required_device_extensions();

private:
    bool create_instance(bool enable_validation);
    bool check_validation_layer_support() const;
    
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool validation_enabled_ = false;
    
    static constexpr const char* VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation";
};

}  // namespace mlx::backend::vulkan
