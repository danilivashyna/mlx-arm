#!/bin/bash
# Build script using Android NDK r29

NDK_PATH=$HOME/code/android-ndk-r29
BUILD_DIR=build-android-ndk

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-34 \
    -DMLX_BUILD_VULKAN=ON \
    -DMLX_BUILD_OPENCL=ON \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

