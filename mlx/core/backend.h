// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/core/device.h"
#include <memory>
#include <vector>

namespace mlx::core {

class Array;
class Stream;

/**
 * Abstract backend interface for executing operations
 * Each backend (CPU, Vulkan, OpenCL) implements this interface
 */
class Backend {
public:
    virtual ~Backend() = default;
    
    /**
     * Get the device type this backend handles
     */
    virtual DeviceType device_type() const = 0;
    
    /**
     * Get human-readable backend name
     */
    virtual std::string name() const = 0;
    
    /**
     * Initialize the backend
     * @return true if initialization succeeded
     */
    virtual bool initialize() = 0;
    
    /**
     * Check if backend is available on this device
     */
    virtual bool is_available() const = 0;
    
    /**
     * Execute a single operation
     */
    virtual void execute(
        const std::string& op_name,
        const std::vector<Array>& inputs,
        std::vector<Array>& outputs,
        const Stream& stream
    ) = 0;
    
    /**
     * Synchronize all pending operations
     */
    virtual void synchronize() = 0;
};

/**
 * Backend registry - manages available backends
 */
class BackendRegistry {
public:
    static BackendRegistry& instance();
    
    /**
     * Register a backend
     */
    void register_backend(std::shared_ptr<Backend> backend);
    
    /**
     * Get backend for a specific device type
     */
    std::shared_ptr<Backend> get_backend(DeviceType type);
    
    /**
     * List all available backends
     */
    std::vector<std::shared_ptr<Backend>> available_backends() const;
    
private:
    BackendRegistry() = default;
    std::vector<std::shared_ptr<Backend>> backends_;
};

} // namespace mlx::core
