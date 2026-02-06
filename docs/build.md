# Building MLX-ARM for Android

This guide will help you build MLX-ARM for Android devices.

## Prerequisites

1. **Android NDK r25 or later**
   ```bash
   # Download from https://developer.android.com/ndk/downloads
   # Or via Android Studio SDK Manager
   export ANDROID_NDK=/path/to/ndk
   ```

2. **CMake 3.20+**
   ```bash
   # Linux
   sudo apt install cmake
   
   # macOS
   brew install cmake
   ```

3. **Vulkan SDK** (for shader compilation)
   ```bash
   # Linux
   wget https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz
   # Follow installation instructions
   
   # macOS
   brew install vulkan-headers vulkan-tools glslang
   ```

4. **Ninja** (recommended)
   ```bash
   # Linux
   sudo apt install ninja-build
   
   # macOS
   brew install ninja
   ```

## Building

### Quick Build

```bash
# Clone repository
git clone https://github.com/yourorg/mlx-arm.git
cd mlx-arm

# Create build directory
mkdir build-android && cd build-android

# Configure
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DMLX_BUILD_VULKAN=ON \
  -DMLX_BUILD_OPENCL=OFF \
  -DMLX_BUILD_PYTHON=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja

# Build
ninja

# Optional: Install
ninja install
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `MLX_BUILD_VULKAN` | ON | Build Vulkan backend |
| `MLX_BUILD_OPENCL` | OFF | Build OpenCL backend |
| `MLX_ENABLE_NEON` | ON | Enable NEON optimizations |
| `MLX_ENABLE_SVE` | ON | Enable SVE2 if available |
| `MLX_BUILD_TESTS` | ON | Build unit tests |

### Android ABI Options

| ABI | Description | Recommended |
|-----|-------------|-------------|
| `arm64-v8a` | 64-bit ARM | ✅ Yes |
| `armeabi-v7a` | 32-bit ARM | ❌ Not recommended |
| `x86_64` | Emulator (64-bit) | For testing only |

### Minimum Android API Level

- **API 26 (Android 8.0)**: Minimum for Vulkan support
- **API 29 (Android 10)**: Recommended for full Vulkan 1.1 features

## Integration with Android Studio

### 1. Add as CMake module

In your `app/build.gradle`:

```gradle
android {
    ...
    externalNativeBuild {
        cmake {
            path "src/main/cpp/CMakeLists.txt"
            version "3.22.1"
        }
    }
    
    defaultConfig {
        ...
        minSdkVersion 26
        ndk {
            abiFilters 'arm64-v8a'
        }
    }
}
```

### 2. Link MLX-ARM library

In your `src/main/cpp/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.22)
project(myapp)

# Add MLX-ARM as subdirectory
add_subdirectory(mlx-arm)

# Your native library
add_library(myapp SHARED
    native-lib.cpp
)

target_link_libraries(myapp
    mlx-core
    log
    android
)
```

## Running Tests

```bash
# After building, tests are in build-android/bin/
adb push build-android/bin/mlx_tests /data/local/tmp/
adb shell /data/local/tmp/mlx_tests
```

## Troubleshooting

### Vulkan not found

Make sure your Android device supports Vulkan:

```bash
adb shell dumpsys vulkan
```

If not available, the device doesn't support Vulkan.

### Shader compilation fails

Ensure `glslangValidator` is in your PATH:

```bash
which glslangValidator
# Should output path to the tool
```

### CMake configuration fails

Check NDK path:

```bash
ls $ANDROID_NDK/build/cmake/android.toolchain.cmake
# Should exist
```

## Next Steps

- [Run the demo app](examples/android/README.md)
- [API documentation](api.md)
- [Contributing guidelines](../CONTRIBUTING.md)
