// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/allocator.h"
#include "mlx/core/device.h"
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <mutex>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <sys/mman.h>
#endif

namespace mlx::core {

// CPU Allocator Implementation

CPUAllocator::CPUAllocator(const Device& device) : device_(device) {}

void* CPUAllocator::allocate(size_t size, size_t alignment) {
    void* ptr = nullptr;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        throw std::bad_alloc();
    }
#endif
    
    if (!ptr) {
        throw std::bad_alloc();
    }
    
    return ptr;
}

void CPUAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// Android Unified Allocator Implementation

#ifdef __ANDROID__

struct BufferInfo {
    AHardwareBuffer* buffer;
    void* mapped_ptr;
    size_t size;
};

class AndroidUnifiedAllocator::Impl {
public:
    std::unordered_map<void*, BufferInfo> buffers_;
    std::mutex mutex_;
};

AndroidUnifiedAllocator::AndroidUnifiedAllocator(const Device& device) 
    : device_(device), impl_(std::make_unique<Impl>()) {}

AndroidUnifiedAllocator::~AndroidUnifiedAllocator() {
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    
    // Clean up all allocated buffers
    for (auto& [ptr, info] : impl_->buffers_) {
        if (info.mapped_ptr) {
            AHardwareBuffer_unlock(info.buffer, nullptr);
        }
        if (info.buffer) {
            AHardwareBuffer_release(info.buffer);
        }
    }
}

void* AndroidUnifiedAllocator::allocate(size_t size, size_t alignment) {
    AHardwareBuffer_Desc desc = {};
    desc.width = size;
    desc.height = 1;
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_BLOB;
    desc.usage = AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                 AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN |
                 AHARDWAREBUFFER_USAGE_GPU_DATA_BUFFER;
    
    AHardwareBuffer* buffer = nullptr;
    if (AHardwareBuffer_allocate(&desc, &buffer) != 0) {
        throw std::bad_alloc();
    }
    
    // Map the buffer for CPU access
    void* mapped_ptr = nullptr;
    if (AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                                     AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                            -1, nullptr, &mapped_ptr) != 0) {
        AHardwareBuffer_release(buffer);
        throw std::runtime_error("Failed to map AHardwareBuffer");
    }
    
    // Store buffer info
    {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->buffers_[mapped_ptr] = BufferInfo{buffer, mapped_ptr, size};
    }
    
    return mapped_ptr;
}

void AndroidUnifiedAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    
    auto it = impl_->buffers_.find(ptr);
    if (it == impl_->buffers_.end()) {
        return;
    }
    
    auto& info = it->second;
    if (info.mapped_ptr) {
        AHardwareBuffer_unlock(info.buffer, nullptr);
    }
    if (info.buffer) {
        AHardwareBuffer_release(info.buffer);
    }
    
    impl_->buffers_.erase(it);
}

void AndroidUnifiedAllocator::sync_to_device(void* ptr, size_t size) {
    // On Android, we need to flush CPU cache to ensure GPU sees updated data
    // AHardwareBuffer handles this internally when unlocking/locking
    
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    auto it = impl_->buffers_.find(ptr);
    if (it != impl_->buffers_.end()) {
        // Unlock and re-lock to ensure cache coherency
        AHardwareBuffer_unlock(it->second.buffer, nullptr);
        AHardwareBuffer_lock(it->second.buffer,
                            AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN |
                            AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,
                            -1, nullptr, &it->second.mapped_ptr);
    }
}

void AndroidUnifiedAllocator::sync_to_host(void* ptr, size_t size) {
    // Similar to sync_to_device, ensure CPU cache is invalidated
    sync_to_device(ptr, size);
}

#endif  // __ANDROID__

// Allocator Factory

static std::unordered_map<Device, std::shared_ptr<Allocator>, Device::Hash> allocator_cache_;
static std::mutex allocator_mutex_;

std::shared_ptr<Allocator> AllocatorFactory::get_allocator(const Device& device) {
    std::lock_guard<std::mutex> lock(allocator_mutex_);
    
    auto it = allocator_cache_.find(device);
    if (it != allocator_cache_.end()) {
        return it->second;
    }
    
    std::shared_ptr<Allocator> allocator;
    
    switch (device.type()) {
        case DeviceType::CPU:
            allocator = std::make_shared<CPUAllocator>(device);
            break;
            
        case DeviceType::GPU:
#ifdef __ANDROID__
            allocator = std::make_shared<AndroidUnifiedAllocator>(device);
#else
            allocator = std::make_shared<CPUAllocator>(device);
#endif
            break;
            
        default:
            throw std::runtime_error("Unsupported device type for allocator");
    }
    
    allocator_cache_[device] = allocator;
    return allocator;
}

}  // namespace mlx::core
