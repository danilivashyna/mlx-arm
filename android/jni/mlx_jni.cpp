// Copyright © 2026 MLX-ARM Contributors
// SPDX-License-Identifier: MIT

/**
 * JNI bridge for MLX-ARM on Android
 * Provides Java/Kotlin access to C++ MLX functionality
 */

#include <jni.h>
#include <string>
#include <vector>

#include "mlx/core/device.h"
#include "mlx/backend/vulkan/vulkan_context.h"

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "MLX-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#endif

extern "C" {

//==============================================================================
// MLX Context Management
//==============================================================================

JNIEXPORT jlong JNICALL
Java_com_mlxarm_MLXContext_nativeCreate(JNIEnv* env, jobject /* this */) {
    try {
        auto* ctx = new mlx::backend::vulkan::VulkanContext();
        return reinterpret_cast<jlong>(ctx);
    } catch (const std::exception& e) {
        LOGE("Failed to create MLX context: %s", e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL
Java_com_mlxarm_MLXContext_nativeDestroy(JNIEnv* env, jobject /* this */, jlong handle) {
    if (handle == 0) return;
    
    auto* ctx = reinterpret_cast<mlx::backend::vulkan::VulkanContext*>(handle);
    delete ctx;
}

JNIEXPORT jboolean JNICALL
Java_com_mlxarm_MLXContext_nativeInitialize(JNIEnv* env, jobject /* this */, jlong handle) {
    if (handle == 0) {
        LOGE("Invalid context handle");
        return JNI_FALSE;
    }
    
    auto* ctx = reinterpret_cast<mlx::backend::vulkan::VulkanContext*>(handle);
    
    try {
        bool success = ctx->initialize(false); // No validation layers in release
        if (success) {
            LOGI("MLX initialized successfully");
            LOGI("Device: %s", ctx->device_name().c_str());
        }
        return success ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGE("Failed to initialize MLX: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jstring JNICALL
Java_com_mlxarm_MLXContext_nativeGetDeviceName(JNIEnv* env, jobject /* this */, jlong handle) {
    if (handle == 0) {
        return env->NewStringUTF("Unknown");
    }
    
    auto* ctx = reinterpret_cast<mlx::backend::vulkan::VulkanContext*>(handle);
    std::string name = ctx->device_name();
    return env->NewStringUTF(name.c_str());
}

JNIEXPORT jboolean JNICALL
Java_com_mlxarm_MLXContext_nativeSupportsFP16(JNIEnv* env, jobject /* this */, jlong handle) {
    if (handle == 0) return JNI_FALSE;
    
    auto* ctx = reinterpret_cast<mlx::backend::vulkan::VulkanContext*>(handle);
    return ctx->supports_fp16() ? JNI_TRUE : JNI_FALSE;
}

//==============================================================================
// Vector Operations (Simple demo)
//==============================================================================

JNIEXPORT jfloatArray JNICALL
Java_com_mlxarm_MLXOps_nativeVectorAdd(
    JNIEnv* env,
    jobject /* this */,
    jlong context_handle,
    jfloatArray a,
    jfloatArray b
) {
    if (context_handle == 0) {
        LOGE("Invalid context handle");
        return nullptr;
    }
    
    jsize length = env->GetArrayLength(a);
    if (length != env->GetArrayLength(b)) {
        LOGE("Array length mismatch");
        return nullptr;
    }
    
    // Get array data
    jfloat* a_data = env->GetFloatArrayElements(a, nullptr);
    jfloat* b_data = env->GetFloatArrayElements(b, nullptr);
    
    // Create result array
    jfloatArray result = env->NewFloatArray(length);
    jfloat* result_data = env->GetFloatArrayElements(result, nullptr);
    
    // Perform addition (CPU for now - GPU dispatch coming soon)
    for (jsize i = 0; i < length; i++) {
        result_data[i] = a_data[i] + b_data[i];
    }
    
    // Release arrays
    env->ReleaseFloatArrayElements(a, a_data, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, b_data, JNI_ABORT);
    env->ReleaseFloatArrayElements(result, result_data, 0);
    
    return result;
}

//==============================================================================
// Device Info
//==============================================================================

JNIEXPORT jobjectArray JNICALL
Java_com_mlxarm_MLXDevice_nativeGetAvailableDevices(JNIEnv* env, jclass /* class */) {
    try {
        auto devices = mlx::core::DeviceManager::available_devices();
        
        // Create Java String array
        jclass string_class = env->FindClass("java/lang/String");
        jobjectArray result = env->NewObjectArray(
            static_cast<jsize>(devices.size()),
            string_class,
            nullptr
        );
        
        for (size_t i = 0; i < devices.size(); i++) {
            std::string device_str = devices[i].to_string();
            jstring j_str = env->NewStringUTF(device_str.c_str());
            env->SetObjectArrayElement(result, static_cast<jsize>(i), j_str);
            env->DeleteLocalRef(j_str);
        }
        
        return result;
    } catch (const std::exception& e) {
        LOGE("Failed to get available devices: %s", e.what());
        return nullptr;
    }
}

} // extern "C"
