// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/backend/vulkan/vulkan_backend.h"
#include "mlx/backend/vulkan/vulkan_context.h"
#include "mlx/backend/vulkan/vulkan_device.h"
#include "mlx/backend/vulkan/vulkan_buffer.h"
#include "mlx/backend/vulkan/vulkan_pipeline.h"
#include "mlx/backend/vulkan/vulkan_command.h"
#include "mlx/core/array.h"
#include <map>

namespace mlx::backend::vulkan {

class VulkanBackend::Impl {
public:
    std::map<std::string, std::unique_ptr<VulkanPipeline>> pipelines;
};

VulkanBackend& VulkanBackend::get_instance() {
    static VulkanBackend backend;
    return backend;
}

VulkanBackend::VulkanBackend() : impl_(std::make_unique<Impl>()) {
    initialize();
}

VulkanBackend::~VulkanBackend() = default;

bool VulkanBackend::initialize() {
    if (initialized_) return true;
    
    auto context = std::make_shared<VulkanContext>();
    if (!context->initialize()) return false;
    
    auto physical_device = context->find_best_compute_device();
    if (physical_device == VK_NULL_HANDLE) return false;
    
    device_ = std::make_shared<VulkanDevice>(context, physical_device);
    if (!device_->initialize()) return false;

    // Pre-create pipelines
    impl_->pipelines["matmul"] = std::make_unique<VulkanPipeline>(*device_, "matmul_naive", 12); // 12 bytes for 3 uint32

    initialized_ = true;
    return true;
}

bool VulkanBackend::is_available() const {
    return initialized_;
}

void VulkanBackend::execute(
    const std::string& op_name,
    const std::vector<core::Array>& inputs,
    std::vector<core::Array>& outputs,
    const core::Stream& stream) {
    
    if (!initialized_) return;

    if (op_name == "matmul") {
        auto& a = inputs[0];
        auto& b = inputs[1];
        auto& res = outputs[0];

        int M = a.shape()[a.ndim()-2];
        int K = a.shape().back();
        int N = b.shape().back();

        // 1. Create buffers
        VulkanBuffer buf_a(*device_, a.size() * sizeof(float), VulkanBuffer::Type::Staging, (void*)a.data());
        VulkanBuffer buf_b(*device_, b.size() * sizeof(float), VulkanBuffer::Type::Staging, (void*)b.data());
        VulkanBuffer buf_res(*device_, res.size() * sizeof(float), VulkanBuffer::Type::Staging);

        // 2. Dispatch
        auto& pipeline = *impl_->pipelines["matmul"];
        auto descriptorSet = pipeline.createDescriptorSet();
        pipeline.updateDescriptorSet(descriptorSet, {buf_a.buffer(), buf_b.buffer(), buf_res.buffer()});

        VulkanCommand cmd(*device_);
        cmd.begin();
        cmd.bindPipeline(pipeline);
        cmd.bindDescriptorSet(pipeline, descriptorSet);
        
        uint32_t dims[3] = {(uint32_t)M, (uint32_t)K, (uint32_t)N};
        cmd.pushConstants(pipeline, dims, sizeof(dims));
        
        // Local size is 16x16
        cmd.dispatch((N + 15) / 16, (M + 15) / 16, 1);
        cmd.end();
        cmd.submit();

        // 3. Download results
        buf_res.read(res.data(), 0, res.size() * sizeof(float));
    }
}

void VulkanBackend::synchronize() {
    if (device_) {
        device_->wait_idle();
    }
}

VkInstance VulkanBackend::instance() const {
    // This is a bit of a hack but we need the instance for some APIs
    // However, VulkanContext handles it. For now return null or add a getter to Context.
    return VK_NULL_HANDLE;
}

}  // namespace mlx::backend::vulkan
