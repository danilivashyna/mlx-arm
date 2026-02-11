// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mlx/core/array.h"
#include <vector>

namespace mlx::core {

Array add(const Array& a, const Array& b);
Array multiply(const Array& a, const Array& b);
Array multiply(const Array& a, float b);
Array matmul(const Array& a, const Array& b);
Array rms_norm(const Array& x, const Array& weight, float eps);
Array softmax(const Array& a, int axis = -1);
Array silu(const Array& a);
Array rope(const Array& x, int dims, int offset, float theta, float scale = 1.0f);
Array reshape(const Array& a, const std::vector<int>& shape);
Array transpose(const Array& a, const std::vector<int>& axes);
Array cross_entropy(const Array& logits, const Array& targets);
Array concat(const std::vector<Array>& arrays, int axis = 0);
Array slice(const Array& a, const std::vector<int>& start, const std::vector<int>& end);

} // namespace mlx::core