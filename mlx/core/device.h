// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace mlx::core {

/**
 * Device types supported by MLX-ARM
 */
enum class DeviceType {
    CPU,  ///< ARM CPU (NEON/SVE2)
    GPU,  ///< GPU via Vulkan or OpenCL
    NPU   ///< Reserved for future NPU support
};

/**
 * Represents a compute device (CPU, GPU, or NPU)
 */
class Device {
public:
    Device(DeviceType type, int index = 0);
    
    DeviceType type() const { return type_; }
    int index() const { return index_; }
    
    std::string name() const;
    size_t memory_capacity() const;
    
    // Capabilities
    bool supports_fp16() const;
    bool supports_bf16() const;
    bool supports_int8() const;
    bool supports_unified_memory() const;
    
    // Comparison operators
    bool operator==(const Device& other) const {
        return type_ == other.type_ && index_ == other.index_;
    }
    
    bool operator!=(const Device& other) const {
        return !(*this == other);
    }
    
    // String representation
    std::string to_string() const;
    
    // Hash support for unordered containers
    struct Hash {
        size_t operator()(const Device& d) const {
            return static_cast<size_t>(d.type_) * 1000 + d.index_;
        }
    };
    
private:
    DeviceType type_;
    int index_;
};

/**
 * Global device management
 */
class DeviceManager {
public:
    // Get default device (usually GPU if available, otherwise CPU)
    static Device default_device();
    
    // Get list of all available devices
    static std::vector<Device> available_devices();
    
    // Set the default device for new arrays
    static void set_default_device(const Device& device);
    
    // Get current default device
    static Device get_default_device();
    
    // Device queries
    static bool has_gpu();
    static Device cpu();
    static Device gpu(int index = 0);
    
private:
    static Device default_device_;
};

// Convenience functions
inline Device cpu() { return DeviceManager::cpu(); }
inline Device gpu(int index = 0) { return DeviceManager::gpu(index); }
inline void set_default_device(const Device& device) {
    DeviceManager::set_default_device(device);
}

} // namespace mlx::core
