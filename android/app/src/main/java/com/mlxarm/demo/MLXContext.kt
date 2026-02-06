package com.mlxarm.demo

/**
 * MLX-ARM Java/Kotlin API
 * JNI bridge to C++ MLX implementation
 */
class MLXContext {
    
    data class DeviceInfo(
        val cpuName: String,
        val gpuName: String,
        val vulkanVersion: String,
        val isInitialized: Boolean
    )
    
    /**
     * Initialize MLX context
     */
    init {
        nativeInit()
    }
    
    /**
     * Get device information
     */
    fun getDeviceInfo(): DeviceInfo {
        val info = nativeGetDeviceInfo()
        return DeviceInfo(
            cpuName = info[0],
            gpuName = info[1],
            vulkanVersion = info[2],
            isInitialized = info[3].toBoolean()
        )
    }
    
    /**
     * Vector addition using Vulkan compute
     * @param a First input vector
     * @param b Second input vector
     * @return Result vector (a + b)
     */
    fun vectorAdd(a: FloatArray, b: FloatArray): FloatArray {
        require(a.size == b.size) { "Input arrays must have same size" }
        return nativeVectorAdd(a, b)
    }
    
    /**
     * Matrix multiplication
     * @param a First matrix (m × k)
     * @param b Second matrix (k × n)
     * @param m Number of rows in A
     * @param k Shared dimension
     * @param n Number of columns in B
     * @return Result matrix (m × n)
     */
    fun matmul(a: FloatArray, b: FloatArray, m: Int, k: Int, n: Int): FloatArray {
        require(a.size == m * k) { "Matrix A size mismatch" }
        require(b.size == k * n) { "Matrix B size mismatch" }
        return nativeMatmul(a, b, m, k, n)
    }
    
    /**
     * Cleanup resources
     */
    fun destroy() {
        nativeDestroy()
    }
    
    // Native methods
    private external fun nativeInit()
    private external fun nativeGetDeviceInfo(): Array<String>
    private external fun nativeVectorAdd(a: FloatArray, b: FloatArray): FloatArray
    private external fun nativeMatmul(a: FloatArray, b: FloatArray, m: Int, k: Int, n: Int): FloatArray
    private external fun nativeDestroy()
}
