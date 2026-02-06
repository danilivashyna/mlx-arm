// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/backend/vulkan/vulkan_context.h"
#include <cstring>
#include <stdexcept>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "MLX-Vulkan"
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

// Debug callback for validation layers
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
    
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        LOGE("Validation layer: %s", callback_data->pMessage);
    } else {
        LOGD("Validation layer: %s", callback_data->pMessage);
    }
    
    return VK_FALSE;
}

VulkanContext::VulkanContext() {}

VulkanContext::~VulkanContext() {
    if (debug_messenger_ != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance_, debug_messenger_, nullptr);
        }
    }
    
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

bool VulkanContext::is_vulkan_available() {
    uint32_t version = 0;
    VkResult result = vkEnumerateInstanceVersion(&version);
    
    if (result != VK_SUCCESS) {
        return false;
    }
    
    LOGI("Vulkan version: %d.%d.%d",
         VK_API_VERSION_MAJOR(version),
         VK_API_VERSION_MINOR(version),
         VK_API_VERSION_PATCH(version));
    
    return VK_API_VERSION_MAJOR(version) >= 1 && VK_API_VERSION_MINOR(version) >= 1;
}

bool VulkanContext::initialize(bool enable_validation) {
    validation_enabled_ = enable_validation;
    
    if (!is_vulkan_available()) {
        LOGE("Vulkan 1.1+ is not available on this system");
        return false;
    }
    
    return create_instance(enable_validation);
}

bool VulkanContext::create_instance(bool enable_validation) {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "MLX-ARM Application";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "MLX-ARM";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_1;
    
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    
    // Get required extensions
    auto extensions = get_required_instance_extensions();
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    
    // Add debug extension if validation enabled
    if (enable_validation) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.data();
    }
    
    // Enable validation layers
    const char* validation_layers[] = {VALIDATION_LAYER};
    if (enable_validation && check_validation_layer_support()) {
        create_info.enabledLayerCount = 1;
        create_info.ppEnabledLayerNames = validation_layers;
        
        // Setup debug messenger
        VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {};
        debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_create_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_create_info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_create_info.pfnUserCallback = debug_callback;
        
        create_info.pNext = &debug_create_info;
    }
    
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    if (result != VK_SUCCESS) {
        LOGE("Failed to create Vulkan instance: %d", result);
        return false;
    }
    
    LOGI("Vulkan instance created successfully");
    
    // Create debug messenger if validation enabled
    if (enable_validation && check_validation_layer_support()) {
        VkDebugUtilsMessengerCreateInfoEXT debug_create_info = {};
        debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_create_info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_create_info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_create_info.pfnUserCallback = debug_callback;
        
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance_, "vkCreateDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance_, &debug_create_info, nullptr, &debug_messenger_);
        }
    }
    
    return true;
}

bool VulkanContext::check_validation_layer_support() const {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());
    
    for (const auto& layer : available_layers) {
        if (strcmp(VALIDATION_LAYER, layer.layerName) == 0) {
            return true;
        }
    }
    
    LOGE("Validation layer %s not available", VALIDATION_LAYER);
    return false;
}

std::vector<VkPhysicalDevice> VulkanContext::get_physical_devices() const {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    
    if (device_count == 0) {
        LOGE("No Vulkan physical devices found");
        return {};
    }
    
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    
    LOGI("Found %u Vulkan physical device(s)", device_count);
    
    return devices;
}

VkPhysicalDeviceProperties VulkanContext::get_device_properties(VkPhysicalDevice device) const {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(device, &properties);
    return properties;
}

VkPhysicalDeviceFeatures VulkanContext::get_device_features(VkPhysicalDevice device) const {
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(device, &features);
    return features;
}

bool VulkanContext::supports_compute(VkPhysicalDevice device) const {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
    
    for (const auto& queue_family : queue_families) {
        if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return true;
        }
    }
    
    return false;
}

VkPhysicalDevice VulkanContext::find_best_compute_device() const {
    auto devices = get_physical_devices();
    
    for (const auto& device : devices) {
        if (!supports_compute(device)) {
            continue;
        }
        
        auto properties = get_device_properties(device);
        LOGI("Selected device: %s", properties.deviceName);
        
        return device;
    }
    
    throw std::runtime_error("No suitable Vulkan compute device found");
}

std::vector<const char*> VulkanContext::get_required_instance_extensions() {
    std::vector<const char*> extensions;
    
    // No platform-specific extensions needed for compute-only
    // If we add graphics later, we'll need surface extensions
    
    return extensions;
}

std::vector<const char*> VulkanContext::get_required_device_extensions() {
    // No required device extensions for basic compute
    return {};
}

}  // namespace mlx::backend::vulkan
