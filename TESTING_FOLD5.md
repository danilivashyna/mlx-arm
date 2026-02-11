# Testing MLX-ARM on Galaxy Fold 5

## Device Specifications

- **Model:** Samsung Galaxy Fold 5
- **SoC:** Snapdragon 8 Gen 2
- **GPU:** Adreno 740
- **RAM:** 12GB
- **Android:** 13+ (API 33)
- **Vulkan:** 1.3.x (expected)

## Test Plan

### 1. Pre-Test Verification

```bash
# Connect device
adb devices

# Check Vulkan support
adb shell dumpsys vulkan

# Expected output should show:
# - Vulkan 1.3.x
# - Physical device: Adreno 740
# - Compute queue support: Yes
```

### 2. Build for Device

```bash
cd /Users/jbarton43/llm_train/mlx_arm
mkdir -p build-fold5 && cd build-fold5

# Configure
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-33 \
  -DMLX_BUILD_VULKAN=ON \
  -DMLX_BUILD_OPENCL=OFF \
  -DMLX_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -GNinja

# Build
ninja -j8

# Check build output
ls -lh bin/
ls -lh lib/
```

### 3. Run Command Line Tests

```bash
# Push test binary
adb push bin/vector_add_example /data/local/tmp/
adb shell chmod +x /data/local/tmp/vector_add_example

# Run test
adb shell /data/local/tmp/vector_add_example

# Capture logs
adb logcat -d | grep -E "MLX|Vulkan" > test_logs_fold5.txt
```

**Expected Output:**
```
MLX-ARM Vector Addition Example
================================

Input vectors size: 1024 elements

[CPU] Computing vector addition...
[CPU] First 5 results: 0 3 6 9 12 

[Vulkan] Initializing...
[Vulkan] Context created
[Vulkan] Using device: Adreno (TM) 740
[Vulkan] Vendor ID: 0x5143
[Vulkan] Device type: 1
[Vulkan] Logical device created
[Vulkan] Compute queue family: 0

[Note] Full compute pipeline implementation coming in next iteration
[Note] For now, this demonstrates Vulkan initialization and device setup

[Vulkan] Cleanup complete

✓ Example completed successfully!
```

### 4. Android App Test

```bash
# Install APK (once demo app is built)
cd android
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

# Launch app
adb shell am start -n com.mlxarm.demo/.MainActivity

# Check for crashes
adb logcat | grep -E "AndroidRuntime|MLX"
```

### 5. Performance Metrics to Collect

#### Vulkan Initialization
- [ ] Context creation time: ____ ms
- [ ] Device enumeration time: ____ ms
- [ ] Logical device creation: ____ ms
- [ ] Total initialization: ____ ms

#### Device Properties
- [ ] GPU Name: ____________________
- [ ] Vulkan Version: ____________________
- [ ] Max Compute Workgroup Size: ____________________
- [ ] Max Memory Allocation: ____ MB
- [ ] Subgroup support: Yes/No
- [ ] FP16 support: Yes/No

#### Memory
- [ ] App memory usage (idle): ____ MB
- [ ] App memory usage (after init): ____ MB
- [ ] GPU memory allocated: ____ MB

### 6. Screenshots Needed

- [ ] App launch screen showing device info
- [ ] "Run Vector Add Test" button and results
- [ ] logcat output showing successful Vulkan init
- [ ] (Optional) GPU profiler snapshot

### 7. Known Issues to Check

- [ ] App crashes on launch?
- [ ] Vulkan initialization fails?
- [ ] "libvulkan.so not found" errors?
- [ ] JNI binding issues?
- [ ] Memory leaks?
- [ ] Screen rotation crashes?

### 8. Test Results Template

```markdown
## Test Results - Galaxy Fold 5

**Date:** 2026-02-06
**Build:** v0.1.0-alpha
**Android Version:** [version]
**Kernel:** [check with `adb shell uname -a`]

### Initialization
✅ Vulkan context created successfully
✅ Device detected: Adreno 740
✅ Compute queue available
⏱️ Total init time: XX ms

### App Test
✅ App launches without crash
✅ Device info displays correctly
✅ Vector add test runs
⏱️ Vector add (1024 elements): XX ms

### Issues Found
- [List any issues]

### Logs
[Attach test_logs_fold5.txt]

### Screenshots
[Attach screenshots]
```

## Troubleshooting

### If Vulkan initialization fails:

```bash
# Check Vulkan layers
adb shell pm list packages | grep vulkan

# Check GPU info
adb shell dumpsys SurfaceFlinger | grep GLES

# Try with validation layers disabled
# Edit vulkan_context.cpp: initialize(false)
```

### If app crashes:

```bash
# Get crash log
adb logcat -b crash > crash.log

# Get native stacktrace
adb shell run-as com.mlxarm.demo cat /data/data/com.mlxarm.demo/files/native_crash.txt

# Check JNI issues
adb logcat | grep "JNI ERROR"
```

### If build fails:

```bash
# Clean build
rm -rf build-fold5
mkdir build-fold5 && cd build-fold5

# Verbose build
ninja -v

# Check NDK path
echo $ANDROID_NDK
```

## Success Criteria

For v0.1.0-alpha release, we need:

- [x] ✅ Project compiles for Android arm64-v8a
- [ ] ✅ App launches on Fold 5 without crash
- [ ] ✅ Vulkan initializes successfully
- [ ] ✅ Device info displayed correctly
- [ ] ✅ At least one screenshot for README
- [ ] ✅ Logs captured and clean (no critical errors)

## After Testing

1. Update README with:
   - "Tested on Galaxy Fold 5" badge
   - Screenshot of running app
   - Performance metrics
   
2. Update CHANGELOG with actual test results

3. Create GitHub release with:
   - Test logs attached
   - Known limitations documented
   - APK available for download

4. Post results in discussions for community feedback

---

**Ready to test?** 🚀

Let's get those metrics and screenshots!
