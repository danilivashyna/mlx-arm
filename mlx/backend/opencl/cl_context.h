// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <string>
#include <memory>

// Forward declarations for OpenCL types to avoid header dependency issues
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;

namespace mlx::core::opencl {

class OpenCLContext {
public:
    static OpenCLContext& instance();

    OpenCLContext();
    ~OpenCLContext();

    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const { return device_; }

    void synchronize();

private:
    void init();

    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    bool initialized_ = false;
};

} // namespace mlx::core::opencl