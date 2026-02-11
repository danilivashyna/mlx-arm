// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/backend.h"

namespace mlx::backend::cpu {

class CPUBackend : public core::Backend {
public:
    core::DeviceType device_type() const override { return core::DeviceType::CPU; }
    std::string name() const override { return "CPU"; }
    bool initialize() override { return true; }
    bool is_available() const override { return true; }
    
    void execute(
        const std::string& op_name,
        const std::vector<core::Array>& inputs,
        std::vector<core::Array>& outputs,
        const core::Stream& stream) override {
        // TODO: CPU operations
    }
    
    void synchronize() override {
        // CPU is always synchronized
    }
};

}  // namespace mlx::backend::cpu
