// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/device.h"
#include <sstream>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "MLX-Device"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#include <iostream>
#define LOGD(...) std::cout << "DEBUG: " << __VA_ARGS__ << std::endl
#define LOGI(...) std::cout << "INFO: " << __VA_ARGS__ << std::endl
#define LOGE(...) std::cerr << "ERROR: " << __VA_ARGS__ << std::endl
#endif

namespace mlx::core {

// Static member initialization
Device DeviceManager::default_device_ = Device(DeviceType::CPU, 0);

Device::Device(DeviceType type, int index) : type_(type), index_(index) {
#ifdef __ANDROID__
    LOGD("Device created: %s (index: %d)", to_string().c_str(), index);
#else
    std::cout << "DEBUG: Device created: " << to_string() << " (index: " << index << ")" << std::endl;
#endif
}

std::string Device::name() const {
    switch (type_) {
        case DeviceType::CPU:
            return "ARM CPU";
        case DeviceType::GPU:
            return "GPU (Vulkan)";
        case DeviceType::NPU:
            return "NPU (Reserved)";
        default:
            return "Unknown";
    }
}

size_t Device::memory_capacity() const {
    // TODO: Query actual device memory capacity
    switch (type_) {
        case DeviceType::CPU:
            return 8ULL * 1024 * 1024 * 1024;  // 8GB default
        case DeviceType::GPU:
            return 4ULL * 1024 * 1024 * 1024;  // 4GB default (shared on mobile)
        case DeviceType::NPU:
            return 0;
        default:
            return 0;
    }
}

bool Device::supports_fp16() const {
    switch (type_) {
        case DeviceType::CPU:
            // ARM NEON supports FP16 on ARMv8.2+
            return true;  // TODO: Runtime detection
        case DeviceType::GPU:
            return true;  // Most mobile GPUs support FP16
        case DeviceType::NPU:
            return true;
        default:
            return false;
    }
}

bool Device::supports_bf16() const {
    switch (type_) {
        case DeviceType::CPU:
            // BF16 requires special CPU support
            return false;  // TODO: Runtime detection
        case DeviceType::GPU:
            return false;  // Limited mobile GPU support
        case DeviceType::NPU:
            return false;
        default:
            return false;
    }
}

bool Device::supports_int8() const {
    // INT8 widely supported across all device types
    return true;
}

bool Device::supports_unified_memory() const {
    // On Android/Linux ARM, unified memory requires special allocator
#ifdef __ANDROID__
    return type_ == DeviceType::GPU;  // Via Gralloc/AHardwareBuffer
#else
    return type_ == DeviceType::CPU;  // CPU always has unified memory
#endif
}

std::string Device::to_string() const {
    std::ostringstream oss;
    oss << "Device(" << name() << ", index=" << index_ << ")";
    return oss.str();
}

// DeviceManager implementation

Device DeviceManager::default_device() {
    return default_device_;
}

std::vector<Device> DeviceManager::available_devices() {
    std::vector<Device> devices;
    
    // CPU is always available
    devices.emplace_back(DeviceType::CPU, 0);
    LOGI("Detected CPU device");
    
    // Check for GPU (Vulkan)
#ifdef MLX_HAS_VULKAN
    // TODO: Actual Vulkan device enumeration
    // For now, assume GPU is available
    devices.emplace_back(DeviceType::GPU, 0);
    LOGI("Detected GPU device (Vulkan)");
#endif
    
    // NPU support reserved for future
    
    return devices;
}

void DeviceManager::set_default_device(const Device& device) {
#ifdef __ANDROID__
    LOGI("Setting default device to: %s", device.to_string().c_str());
#else
    std::cout << "INFO: Setting default device to: " << device.to_string() << std::endl;
#endif
    default_device_ = device;
}

Device DeviceManager::get_default_device() {
    return default_device_;
}

bool DeviceManager::has_gpu() {
#ifdef MLX_HAS_VULKAN
    return true;  // TODO: Actual Vulkan availability check
#else
    return false;
#endif
}

Device DeviceManager::cpu() {
    return Device(DeviceType::CPU, 0);
}

Device DeviceManager::gpu(int index) {
#ifdef MLX_HAS_VULKAN
    return Device(DeviceType::GPU, index);
#else
    LOGE("GPU requested but Vulkan backend not available");
    throw std::runtime_error("GPU device not available (Vulkan not compiled)");
#endif
}

}  // namespace mlx::core
