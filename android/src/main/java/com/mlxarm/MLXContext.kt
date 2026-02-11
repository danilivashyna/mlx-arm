// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

package com.mlxarm

/**
 * Main MLX context for Android
 * Manages Vulkan device and computation
 */
class MLXContext {
    private var nativeHandle: Long = 0
    
    init {
        System.loadLibrary("mlx-jni")
        nativeHandle = nativeCreate()
    }
    
    /**
     * Initialize MLX with Vulkan backend
     * @return true if initialization succeeded
     */
    fun initialize(): Boolean {
        return nativeInitialize(nativeHandle)
    }
    
    /**
     * Get GPU device name
     */
    fun getDeviceName(): String {
        return nativeGetDeviceName(nativeHandle)
    }
    
    /**
     * Check if device supports FP16
     */
    fun supportsFP16(): Boolean {
        return nativeSupportsFP16(nativeHandle)
    }
    
    /**
     * Clean up resources
     */
    fun destroy() {
        if (nativeHandle != 0L) {
            nativeDestroy(nativeHandle)
            nativeHandle = 0
        }
    }
    
    override fun finalize() {
        destroy()
    }
    
    // Native methods
    private external fun nativeCreate(): Long
    private external fun nativeDestroy(handle: Long)
    private external fun nativeInitialize(handle: Long): Boolean
    private external fun nativeGetDeviceName(handle: Long): String
    private external fun nativeSupportsFP16(handle: Long): Boolean
}

/**
 * Vector operations
 */
class MLXOps {
    companion object {
        init {
            System.loadLibrary("mlx-jni")
        }
        
        /**
         * Add two vectors element-wise
         * @param context MLX context
         * @param a First vector
         * @param b Second vector
         * @return Result vector (a + b)
         */
        @JvmStatic
        external fun nativeVectorAdd(context: Long, a: FloatArray, b: FloatArray): FloatArray
    }
}

/**
 * Device information
 */
class MLXDevice {
    companion object {
        init {
            System.loadLibrary("mlx-jni")
        }
        
        /**
         * Get list of available compute devices
         * @return Array of device names
         */
        @JvmStatic
        external fun nativeGetAvailableDevices(): Array<String>
        
        /**
         * Get available devices as list
         */
        fun getAvailableDevices(): List<String> {
            return nativeGetAvailableDevices().toList()
        }
    }
}
