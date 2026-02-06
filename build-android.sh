#!/bin/bash
# MLX-ARM Android Build Script
# For Galaxy Fold 5 (Snapdragon 8 Gen 2, Adreno 740, Android 13+)

set -e

# Configuration
export ANDROID_NDK=~/Library/Android/sdk/ndk/27.0.12077973
export ANDROID_PLATFORM=android-33  # Android 13
export ANDROID_ABI=arm64-v8a        # Galaxy Fold 5 is 64-bit ARM
export VULKAN_SDK=/Users/jBarton43/VulkanSDK/1.4.341.0/macOS

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== MLX-ARM Android Build for Galaxy Fold 5 ===${NC}"
echo -e "${YELLOW}Target: Snapdragon 8 Gen 2, Adreno 740, Vulkan 1.3${NC}"
echo ""

# Check NDK
if [ ! -d "$ANDROID_NDK" ]; then
    echo -e "${RED}Error: Android NDK not found at $ANDROID_NDK${NC}"
    echo "Please install NDK r27 via Android Studio SDK Manager"
    exit 1
fi

echo -e "${GREEN}✓ NDK found: $ANDROID_NDK${NC}"

# Create build directory
BUILD_DIR="build-android"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
echo -e "${YELLOW}Configuring CMake for Android...${NC}"
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DANDROID_PLATFORM=$ANDROID_PLATFORM \
    -DANDROID_STL=c++_shared \
    -DMLX_BUILD_VULKAN=ON \
    -DMLX_BUILD_OPENCL=OFF \
    -DMLX_BUILD_PYTHON=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DVulkan_INCLUDE_DIR=$VULKAN_SDK/include \
    -DVulkan_LIBRARY=$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib/aarch64-linux-android/libvulkan.so

# Build
echo -e "${YELLOW}Building MLX-ARM...${NC}"
make -j$(sysctl -n hw.ncpu)

echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo -e "${YELLOW}Binary location: bin/vector_add_example${NC}"
echo ""
echo -e "${GREEN}To deploy to Galaxy Fold 5:${NC}"
echo "  adb push bin/vector_add_example /data/local/tmp/"
echo "  adb shell chmod +x /data/local/tmp/vector_add_example"
echo "  adb shell /data/local/tmp/vector_add_example"
