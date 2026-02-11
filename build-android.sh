#!/bin/bash
# MLX-ARM Android Build Script (Termux Fixed)
set -e

# Hardcoded paths for your Termux environment
export ANDROID_NDK=/data/data/com.termux/files/home/code/android-ndk-r29
export ANDROID_PLATFORM=android-33
export ANDROID_ABI=arm64-v8a

echo "=== MLX-ARM Local Build ==="
echo "NDK: $ANDROID_NDK"

if [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: NDK not found at $ANDROID_NDK"
    exit 1
fi

BUILD_DIR="build-android"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DANDROID_STL=c++_static \
    -DMLX_BUILD_VULKAN=ON \
    -DMLX_BUILD_OPENCL=OFF \
    -DMLX_BUILD_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release

make -j8
echo "✅ Done!"