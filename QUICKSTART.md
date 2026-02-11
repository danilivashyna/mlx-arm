# MLX-ARM Quick Start Guide

Get MLX-ARM running on your Android device in 5 minutes!

## Prerequisites

- Android device with:
  - Android 8.0+ (API 26+)
  - ARM64 processor
  - Vulkan 1.1+ support
- Android Studio (latest)
- Android NDK r25+

## Step 1: Check Device Compatibility

Verify your device supports Vulkan:

```bash
adb shell dumpsys vulkan
```

Should show Vulkan version 1.1.0 or higher.

## Step 2: Clone Repository

```bash
git clone https://github.com/yourorg/mlx-arm.git
cd mlx-arm
```

## Step 3: Build Library

### Option A: Android Studio (Recommended)

1. Open `android/` folder in Android Studio
2. Let Gradle sync
3. Build → Make Project
4. Run on device

### Option B: Command Line

```bash
cd android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Step 4: Run Demo

Launch the app on your device. You should see:

```
=== MLX-ARM Demo ===

📱 Available Devices:
  • Device(CPU, index=0, name="ARM CPU (ARMv8-A + NEON)")
  • Device(GPU, index=0, name="Adreno 730")

✅ MLX initialized successfully
Device: Adreno 730
FP16 support: true

🧮 Running Vector Addition Demo...
✅ Results verified correctly!
Time: 2.34 ms
Sample: 10.0 + 20.0 = 30.0
```

## Step 5: Run from Code

### Kotlin

```kotlin
import com.mlxarm.MLXContext

class MyActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize MLX
        val mlx = MLXContext()
        if (mlx.initialize()) {
            Log.i("MLX", "Device: ${mlx.getDeviceName()}")
            
            // Your ML code here
            
            mlx.destroy()
        }
    }
}
```

### C++ (via JNI)

```cpp
#include "mlx/core/device.h"
#include "mlx/backend/vulkan/vulkan_context.h"

void run_inference() {
    mlx::backend::vulkan::VulkanContext ctx;
    ctx.initialize();
    
    // Your code here
}
```

## Next Steps

- [Examples](examples/README.md) - More demos
- [API Reference](docs/api.md) - Full API documentation
- [Build Guide](docs/build.md) - Advanced build options
- [Contributing](CONTRIBUTING.md) - Join development

## Troubleshooting

### "Vulkan not found"

Your device doesn't support Vulkan. Try:
- Update to latest Android version
- Check manufacturer's GPU driver updates

### "Failed to initialize"

Check logcat:
```bash
adb logcat | grep MLX
```

Common issues:
- Insufficient GPU memory
- Driver compatibility
- App permissions

### Build Errors

Make sure you have:
- NDK r25 or later
- CMake 3.22+
- Vulkan SDK headers

## Get Help

- GitHub Issues: Report bugs
- Discussions: Ask questions
- Discord: Real-time chat (coming soon)

---

**Welcome to MLX-ARM! 🚀**
