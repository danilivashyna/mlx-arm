// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/core/array.h"
#include <string>
#include <map>
#include <unordered_map>

namespace mlx::core {

// Loads weights from a .safetensors file
std::map<std::string, Array> load_safetensors(const std::string& filename);

} // namespace mlx::core