// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

/**
 * JNI Bridge for MLX-ARM Android Integration
 * Provides Java/Kotlin access to C++ MLX functionality
 */

#include <jni.h>
#include <android/log.h>
#include <vector>
#include <memory>
#include <string>

#include "mlx/core/device.h"
#include "mlx/backend/vulkan/vulkan_context.h"
#include "mlx/backend/vulkan/vulkan_device.h"

#define LOG_TAG "MLX-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace mlx::core;
using namespace mlx::backend::vulkan;

// Global context (TODO: make per-instance)
static std::shared_ptr<VulkanContext> g_vulkan_context;
static std::shared_ptr<VulkanDevice> g_vulkan_device;

extern "C" {

JNIEXPORT void JNICALL
Java_com_mlxarm_demo_MLXContext_nativeInit(JNIEnv* env, jobject thiz) {
    LOGI("Initializing MLX-ARM");
    
    try {
        // Initialize Vulkan context
        g_vulkan_context = std::make_shared<VulkanContext>();
        if (!g_vulkan_context->initialize(false)) {
            LOGE("Failed to initialize Vulkan context");
            return;
        }
        
        // Find and create device
        VkPhysicalDevice physical_device = g_vulkan_context->find_best_compute_device();
        g_vulkan_device = std::make_shared<VulkanDevice>(g_vulkan_context, physical_device);
        
        if (!g_vulkan_device->initialize()) {
            LOGE("Failed to initialize Vulkan device");
            return;
        }
        
        LOGI("MLX-ARM initialized successfully");
    } catch (const std::exception& e) {
        LOGE("Exception during initialization: %s", e.what());
    }
}

JNIEXPORT jobjectArray JNICALL
Java_com_mlxarm_demo_MLXContext_nativeGetDeviceInfo(JNIEnv* env, jobject thiz) {
    // Create String array [cpuName, gpuName, vulkanVersion, isInitialized]
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(4, stringClass, nullptr);
    
    // CPU info
    env->SetObjectArrayElement(result, 0, env->NewStringUTF("ARM CPU (NEON)"));
    
    // GPU info
    if (g_vulkan_device) {
        auto props = g_vulkan_device->properties();
        env->SetObjectArrayElement(result, 1, env->NewStringUTF(props.deviceName));
        
        // Vulkan version
        char version[32];
        snprintf(version, sizeof(version), "%d.%d.%d",
                VK_API_VERSION_MAJOR(props.apiVersion),
                VK_API_VERSION_MINOR(props.apiVersion),
                VK_API_VERSION_PATCH(props.apiVersion));
        env->SetObjectArrayElement(result, 2, env->NewStringUTF(version));
        
        env->SetObjectArrayElement(result, 3, env->NewStringUTF("true"));
    } else {
        env->SetObjectArrayElement(result, 1, env->NewStringUTF("N/A"));
        env->SetObjectArrayElement(result, 2, env->NewStringUTF("N/A"));
        env->SetObjectArrayElement(result, 3, env->NewStringUTF("false"));
    }
    
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_mlxarm_demo_MLXContext_nativeVectorAdd(
    JNIEnv* env, jobject thiz, jfloatArray a, jfloatArray b) {
    
    jsize len = env->GetArrayLength(a);
    
    // Get input arrays
    jfloat* a_data = env->GetFloatArrayElements(a, nullptr);
    jfloat* b_data = env->GetFloatArrayElements(b, nullptr);
    
    // Create output array
    jfloatArray result = env->NewFloatArray(len);
    jfloat* result_data = env->GetFloatArrayElements(result, nullptr);
    
    // TODO: Run on Vulkan compute
    // For now, simple CPU implementation
    for (jsize i = 0; i < len; i++) {
        result_data[i] = a_data[i] + b_data[i];
    }
    
    // Release arrays
    env->ReleaseFloatArrayElements(a, a_data, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, b_data, JNI_ABORT);
    env->ReleaseFloatArrayElements(result, result_data, 0);
    
    return result;
}

JNIEXPORT jfloatArray JNICALL
Java_com_mlxarm_demo_MLXContext_nativeMatmul(
    JNIEnv* env, jobject thiz,
    jfloatArray a, jfloatArray b,
    jint m, jint k, jint n) {
    
    // TODO: Implement matrix multiplication
    // For now, return empty array
    jfloatArray result = env->NewFloatArray(m * n);
    return result;
}

JNIEXPORT void JNICALL
Java_com_mlxarm_demo_MLXContext_nativeDestroy(JNIEnv* env, jobject thiz) {
    LOGI("Destroying MLX-ARM context");
    
    if (g_vulkan_device) {
        g_vulkan_device->wait_idle();
        g_vulkan_device.reset();
    }
    
    g_vulkan_context.reset();
    
    LOGI("MLX-ARM context destroyed");
}

} // extern "C"
