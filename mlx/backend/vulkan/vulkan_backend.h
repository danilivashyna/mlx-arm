// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/core/backend.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace mlx::backend::vulkan {

class VulkanContext;
class VulkanDevice;

/**
 * Vulkan compute backend implementation
 * Primary GPU backend for MLX-ARM
 */
class VulkanBackend : public core::Backend {
public:
    VulkanBackend();
    ~VulkanBackend() override;
    
    // Backend interface
    core::DeviceType device_type() const override { return core::DeviceType::GPU; }
    std::string name() const override { return "Vulkan"; }
    bool initialize() override;
    bool is_available() const override;
    
    void execute(
        const std::string& op_name,
        const std::vector<core::Array>& inputs,
        std::vector<core::Array>& outputs,
        const core::Stream& stream
    ) override;
    
    void synchronize() override;
    
    // Vulkan-specific accessors
    VkInstance instance() const;
    std::shared_ptr<VulkanDevice> device() const { return device_; }
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::shared_ptr<VulkanDevice> device_;
    bool initialized_ = false;
};

} // namespace mlx::backend::vulkan
