# MLX-ARM Examples

This directory contains example applications demonstrating MLX-ARM functionality.

## Available Examples

### 1. Vector Addition (C++)

**File:** `vector_add.cpp`

Basic example showing:
- Vulkan context initialization
- Device detection
- Simple compute operation

**Build & Run:**
```bash
cd build
./bin/examples/vector_add_example
```

**Expected Output:**
```
=== MLX-ARM Vulkan Vector Addition Demo ===
✅ Vulkan initialized
Device: Adreno 730
FP16 support: Yes
Subgroups: Yes

📊 Test data prepared: 1024 elements
✅ Results verified correctly!
Sample: 10.0 + 20.0 = 30.0

🎉 Demo completed successfully!
```

---

## Android Demo App

**Location:** `../android/`

Full Android application with:
- MLX context management
- Device enumeration
- Vector operations via JNI
- Material Design UI

**Build & Run:**
```bash
cd android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.mlxarm.demo/.MainActivity
```

**Features:**
- Lists available compute devices (CPU/GPU)
- Runs vector addition benchmark
- Displays device capabilities (FP16, Vulkan version)
- Shows execution time

---

## Coming Soon

### 2. Matrix Multiplication
- Tiled matmul shader
- Performance comparison (CPU vs GPU)
- Different matrix sizes

### 3. LLM Inference
- Llama 3 8B (quantized)
- Token generation demo
- Interactive prompt

### 4. Quantization Demo
- Q4/Q8 quantization
- Memory usage comparison
- Quality metrics

---

## Building Examples

All examples are built automatically when you build the main project:

```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DANDROID_PLATFORM=android-26
make -j$(nproc)
```

Examples will be in `build/bin/examples/`

---

## Contributing Examples

Want to add an example? Great!

1. Create `examples/your_example.cpp`
2. Add to `examples/CMakeLists.txt`
3. Document in this README
4. Submit PR

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
