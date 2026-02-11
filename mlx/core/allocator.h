// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include "mlx/core/device.h"

namespace mlx::core {

/**
 * Memory allocator interface for unified memory management
 */
class Allocator {
public:
    virtual ~Allocator() = default;
    
    /**
     * Allocate memory with specified size and alignment
     * @param size Size in bytes
     * @param alignment Alignment requirement (default: 64 for NEON)
     * @return Pointer to allocated memory
     */
    virtual void* allocate(size_t size, size_t alignment = 64) = 0;
    
    /**
     * Deallocate previously allocated memory
     * @param ptr Pointer to memory to deallocate
     */
    virtual void deallocate(void* ptr) = 0;
    
    /**
     * Check if this allocator provides unified (shared) memory
     * @return true if memory is accessible from both CPU and GPU without copies
     */
    virtual bool is_unified() const = 0;
    
    /**
     * Synchronize memory from host to device
     * Only needed for non-unified memory allocators
     */
    virtual void sync_to_device(void* ptr, size_t size) = 0;
    
    /**
     * Synchronize memory from device to host
     * Only needed for non-unified memory allocators
     */
    virtual void sync_to_host(void* ptr, size_t size) = 0;
    
    /**
     * Get the device this allocator is associated with
     */
    virtual const Device& device() const = 0;
};

/**
 * CPU allocator using standard aligned allocation
 */
class CPUAllocator : public Allocator {
public:
    explicit CPUAllocator(const Device& device);
    ~CPUAllocator() override = default;
    
    void* allocate(size_t size, size_t alignment = 64) override;
    void deallocate(void* ptr) override;
    bool is_unified() const override { return true; }
    void sync_to_device(void*, size_t) override {}  // No-op for CPU
    void sync_to_host(void*, size_t) override {}    // No-op for CPU
    const Device& device() const override { return device_; }
    
private:
    Device device_;
};

/**
 * Android unified allocator using Gralloc/AHardwareBuffer
 * Provides zero-copy memory shared between CPU and GPU
 */
#ifdef __ANDROID__
class AndroidUnifiedAllocator : public Allocator {
public:
    explicit AndroidUnifiedAllocator(const Device& device);
    ~AndroidUnifiedAllocator() override;
    
    void* allocate(size_t size, size_t alignment = 64) override;
    void deallocate(void* ptr) override;
    bool is_unified() const override { return true; }
    void sync_to_device(void* ptr, size_t size) override;  // Cache flush
    void sync_to_host(void* ptr, size_t size) override;    // Cache invalidate
    const Device& device() const override { return device_; }
    
private:
    Device device_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

/**
 * Factory for creating appropriate allocator for a device
 */
class AllocatorFactory {
public:
    static std::shared_ptr<Allocator> get_allocator(const Device& device);
};

} // namespace mlx::core
