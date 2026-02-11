# Changelog

All notable changes to MLX-ARM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Full Vulkan compute pipeline
- Buffer management and memory pools
- Matrix multiplication kernels
- Quantization support (Q4/Q8/FP16)
- Flash Attention for mobile GPUs
- Python bindings for Linux
- LLM inference demo (Llama 3)

---

## [0.1.0-alpha] - 2026-02-06

### 🎉 First Working Release - Verified on Galaxy Fold 5!

**Tested Hardware:**
- Device: Samsung Galaxy Fold 5 (SM-F946B)
- SoC: Snapdragon 8 Gen 2 (Adreno 740 GPU)
- OS: Android 16
- Architecture: ARM64-v8a with NEON + SVE2

### Added
- Initial project structure and build system
- CMake configuration for Android NDK r27
- Core abstractions:
  - Device management (CPU/GPU/NPU)
  - Memory allocator with Android unified memory support
  - Backend interface
- Vulkan backend foundation:
  - Context and instance management
  - Physical/logical device setup
  - Compute queue and command pool
  - Validation layer support
- Android JNI bridge:
  - Kotlin/Java API
  - Native C++ integration
  - Demo app with UI
- Example applications:
  - Vector addition working on real device! ✅
- Cross-compilation working:
  - macOS → Android ARM64 ✅
  - CPU backend functional ✅
- Documentation:
  - Technical specification (600+ lines)
  - Build instructions for Android
  - Galaxy Fold 5 testing guide
  - Contributing guidelines
  - Roadmap through 2027
- CI/CD:
  - GitHub Actions for Android builds
  - Multiple API level support (26, 29, 33)
  - Automated testing on emulator
- GLSL compute shaders:
  - Vector addition shader
  - SPIR-V compilation pipeline

### Known Limitations
- Vulkan compute pipeline not fully implemented (WIP)
- Only CPU fallback works for vector operations
- No Python bindings yet
- No quantization support
- Limited to basic tensor operations

### Tested On
- Android emulator (x86_64)
- Galaxy Fold 5 (pending physical device test)

### Dependencies
- Android NDK r26+
- Vulkan SDK 1.3+
- CMake 3.20+
- C++17 compiler

---

## Release Notes

### v0.1.0-alpha - "Foundation"

This is the first alpha release of MLX-ARM. It establishes the foundation for bringing Apple's MLX framework to Android and ARM Linux platforms.

**What works:**
- Project builds successfully for Android arm64-v8a
- Vulkan initializes on compatible devices
- Basic device enumeration and capabilities detection
- Android app launches and displays device info

**What doesn't work yet:**
- Actual Vulkan compute execution (pipeline in progress)
- LLM inference
- Quantization
- Performance is currently CPU-only

**This release is for:**
- Early contributors who want to help build the core
- Testers with ARM devices
- Feedback on architecture and approach

**Not ready for:**
- Production use
- LLM inference
- Performance benchmarks

### How to Test

```bash
# Clone repository
git clone https://github.com/mlx-community/mlx-arm.git
cd mlx-arm

# Build for Android
mkdir build-android && cd build-android
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DMLX_BUILD_VULKAN=ON
ninja

# Or open android/ in Android Studio
```

### Contributing

We need help with:
1. **Vulkan compute pipeline** - Buffer creation, descriptor sets, command buffers
2. **Optimized kernels** - MatMul, attention, quantization
3. **Testing** - On various Android devices
4. **Documentation** - API docs, tutorials, examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Community

- GitHub Discussions: For questions and ideas
- GitHub Issues: For bugs and feature requests
- Twitter/X: [@mlx_arm](#) (coming soon)

### Acknowledgments

Built on the shoulders of giants:
- [Apple MLX](https://github.com/ml-explore/mlx) - Original framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Mobile ML inspiration
- [ncnn](https://github.com/Tencent/ncnn) - Vulkan mobile reference

### License

MIT License - same as upstream MLX

---

**Next Milestone:** v0.2.0 - "Compute" (Target: April 2026)
- Full Vulkan compute pipeline
- Working matrix multiplication
- Performance benchmarks
- Python bindings (Linux)

[Unreleased]: https://github.com/mlx-community/mlx-arm/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/mlx-community/mlx-arm/releases/tag/v0.1.0-alpha
