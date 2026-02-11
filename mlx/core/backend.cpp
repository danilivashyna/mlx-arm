// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/backend.h"
#include <stdexcept>

namespace mlx::core {

// Backend Registry Implementation

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry instance;
    return instance;
}

void BackendRegistry::register_backend(std::shared_ptr<Backend> backend) {
    backends_.push_back(backend);
}

std::shared_ptr<Backend> BackendRegistry::get_backend(DeviceType type) {
    for (auto& backend : backends_) {
        if (backend->device_type() == type) {
            return backend;
        }
    }
    throw std::runtime_error("Backend not found for device type");
}

std::vector<std::shared_ptr<Backend>> BackendRegistry::available_backends() const {
    return backends_;
}

}  // namespace mlx::core
