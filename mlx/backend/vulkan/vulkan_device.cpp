// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/backend/vulkan/vulkan_device.h"
#include "mlx/backend/vulkan/vulkan_context.h"
#include <stdexcept>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "MLX-VulkanDevice"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <iostream>
#include <cstdio>
#define LOGD(fmt, ...) do { printf("DEBUG: " fmt "\n", ##__VA_ARGS__); } while(0)
#define LOGI(fmt, ...) do { printf("INFO: " fmt "\n", ##__VA_ARGS__); } while(0)
#define LOGE(fmt, ...) do { fprintf(stderr, "ERROR: " fmt "\n", ##__VA_ARGS__); } while(0)
#endif

namespace mlx::backend::vulkan {

VulkanDevice::VulkanDevice(std::shared_ptr<VulkanContext> context,
                          VkPhysicalDevice physical_device)
    : context_(context), physical_device_(physical_device) {
    
    vkGetPhysicalDeviceProperties(physical_device_, &properties_);
    vkGetPhysicalDeviceFeatures(physical_device_, &features_);
    
    LOGI("Created VulkanDevice for: %s", properties_.deviceName);
}

VulkanDevice::~VulkanDevice() {
    if (device_ != VK_NULL_HANDLE) {
        wait_idle();
        
        if (command_pool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
        }
        
        vkDestroyDevice(device_, nullptr);
        LOGI("Destroyed Vulkan logical device");
    }
}

bool VulkanDevice::initialize() {
    if (!create_logical_device()) {
        return false;
    }
    
    if (!create_command_pool()) {
        return false;
    }
    
    LOGI("VulkanDevice initialized successfully");
    return true;
}

uint32_t VulkanDevice::find_compute_queue_family() const {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count,
                                            queue_families.data());
    
    // Find first queue family that supports compute
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            LOGI("Found compute queue family at index %u", i);
            return i;
        }
    }
    
    throw std::runtime_error("No compute queue family found");
}

bool VulkanDevice::create_logical_device() {
    compute_queue_family_ = find_compute_queue_family();
    
    // Queue create info
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    
    // Device features we want to enable
    VkPhysicalDeviceFeatures device_features = {};
    device_features.shaderFloat64 = features_.shaderFloat64;
    device_features.shaderInt16 = features_.shaderInt16;
    device_features.shaderInt64 = features_.shaderInt64;
    
    // Device extensions
    auto extensions = VulkanContext::get_required_device_extensions();
    
    // Device create info
    VkDeviceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.queueCreateInfoCount = 1;
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    
    VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create logical device: %d", result);
        return false;
    }
    
    // Get compute queue
    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
    
    LOGI("Logical device created successfully");
    return true;
}

bool VulkanDevice::create_command_pool() {
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = compute_queue_family_;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    VkResult result = vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create command pool: %d", result);
        return false;
    }
    
    LOGI("Command pool created successfully");
    return true;
}

void VulkanDevice::wait_idle() const {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
}

}  // namespace mlx::backend::vulkan
