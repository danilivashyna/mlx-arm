// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/backend/opencl/cl_context.h"
#include <iostream>
#include <dlfcn.h>
#include <stdexcept>

namespace mlx::core::opencl {

// Function pointers for OpenCL
typedef int32_t (*clGetPlatformIDs_ptr)(uint32_t, cl_platform_id*, uint32_t*);
typedef int32_t (*clGetDeviceIDs_ptr)(cl_platform_id, uint64_t, uint32_t, cl_device_id*, uint32_t*);
typedef cl_context (*clCreateContext_ptr)(void*, uint32_t, const cl_device_id*, void*, void*, int32_t*);
typedef cl_command_queue (*clCreateCommandQueueWithProperties_ptr)(cl_context, cl_device_id, const void*, int32_t*);
typedef int32_t (*clFinish_ptr)(cl_command_queue);

static void* cl_handle = nullptr;
static clGetPlatformIDs_ptr clGetPlatformIDs_f = nullptr;
static clGetDeviceIDs_ptr clGetDeviceIDs_f = nullptr;
static clCreateContext_ptr clCreateContext_f = nullptr;
static clCreateCommandQueueWithProperties_ptr clCreateCommandQueue_f = nullptr;
static clFinish_ptr clFinish_f = nullptr;

OpenCLContext& OpenCLContext::instance() {
    static OpenCLContext instance;
    return instance;
}

OpenCLContext::OpenCLContext() {
    init();
}

OpenCLContext::~OpenCLContext() {
    // Release resources if needed
}

void OpenCLContext::init() {
    if (initialized_) return;

    // Try multiple paths for libOpenCL.so
    const char* paths[] = {
        "/system/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "libOpenCL.so"
    };

    for (const char* path : paths) {
        cl_handle = dlopen(path, RTLD_LAZY);
        if (cl_handle) break;
    }

    if (!cl_handle) {
        std::cerr << "OpenCL WARNING: Could not load libOpenCL.so. GPU acceleration disabled." << std::endl;
        return;
    }

    // Load symbols
    clGetPlatformIDs_f = (clGetPlatformIDs_ptr)dlsym(cl_handle, "clGetPlatformIDs");
    clGetDeviceIDs_f = (clGetDeviceIDs_ptr)dlsym(cl_handle, "clGetDeviceIDs");
    clCreateContext_f = (clCreateContext_ptr)dlsym(cl_handle, "clCreateContext");
    clCreateCommandQueue_f = (clCreateCommandQueueWithProperties_ptr)dlsym(cl_handle, "clCreateCommandQueueWithProperties");
    clFinish_f = (clFinish_ptr)dlsym(cl_handle, "clFinish");

    if (!clGetPlatformIDs_f || !clGetDeviceIDs_f || !clCreateContext_f) {
        std::cerr << "OpenCL ERROR: Required symbols not found." << std::endl;
        return;
    }

    // Initialize OpenCL
    uint32_t num_platforms = 0;
    clGetPlatformIDs_f(1, &platform_, &num_platforms);
    if (num_platforms == 0) return;

    uint32_t num_devices = 0;
    // CL_DEVICE_TYPE_GPU = (1 << 2)
    clGetDeviceIDs_f(platform_, (1 << 2), 1, &device_, &num_devices);
    if (num_devices == 0) return;

    int32_t err = 0;
    context_ = clCreateContext_f(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != 0) return;

    queue_ = clCreateCommandQueue_f(context_, device_, nullptr, &err);
    if (err != 0) return;

    initialized_ = true;
    printf("DEBUG: OpenCL initialized successfully on GPU Adreno.\n");
}

void OpenCLContext::synchronize() {
    if (initialized_ && clFinish_f) {
        clFinish_f(queue_);
    }
}

} // namespace mlx::core::opencl
