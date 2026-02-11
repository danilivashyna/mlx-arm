// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#include "mlx/core/allocator.h"
#include "mlx/core/device.h"
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <mutex>

#ifdef __ANDROID__
// #include <android/hardware_buffer.h>
#include <sys/mman.h>
#endif

namespace mlx::core {

// ... (CPU Allocator remains same)

// Android Unified Allocator Implementation

#ifdef __ANDROID__

struct BufferInfo {
    // AHardwareBuffer* buffer;
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
    for (auto& [ptr, info] : impl_->buffers_) {
        free(ptr); // Simple free for now
    }
}

void* AndroidUnifiedAllocator::allocate(size_t size, size_t alignment) {
    // FALLBACK TO STANDARD MALLOC FOR TERMUX BUILD
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) throw std::bad_alloc();
    
    {
        std::lock_guard<std::mutex> lock(impl_->mutex_);
        impl_->buffers_[ptr] = BufferInfo{ptr, size};
    }
    return ptr;
}

void AndroidUnifiedAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(impl_->mutex_);
    if (impl_->buffers_.erase(ptr)) {
        free(ptr);
    }
}

void AndroidUnifiedAllocator::sync_to_device(void* ptr, size_t size) { }
void AndroidUnifiedAllocator::sync_to_host(void* ptr, size_t size) { }

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
