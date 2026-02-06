# MLX-ARM: Universal ARM Machine Learning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/Platform-Android%20%7C%20Linux%20%7C%20ARM-blue)]()
[![GPU](https://img.shields.io/badge/GPU-Vulkan%20%7C%20OpenCL-green)]()
[![Version](https://img.shields.io/badge/Version-0.1.0--alpha-orange)]()

> Bringing MLX to Android, ARM Linux, and beyond - with a path to Steam/Proton gaming ecosystem

**🤝 Developed in collaboration:** This project is a human-AI collaboration between [@jbarton43](https://github.com/jbarton43) and **GitHub Copilot / Claude AI**. All architecture decisions, code implementation, and testing were done through iterative pair programming between human creativity and AI assistance.

## 🎉 v0.1.0-alpha Released!

**✅ First working release verified on Samsung Galaxy Fold 5!**

- ✅ Cross-compilation working (macOS → Android ARM64)
- ✅ CPU backend with NEON/SVE2 optimizations
- ✅ Binary runs on real device (Snapdragon 8 Gen 2)
- ✅ Vector addition verified (100% accurate)
- ⚠️ GPU acceleration coming in v0.2.0

See [FIRST_RUN.md](FIRST_RUN.md) for detailed test results.

## 🎯 Vision

MLX-ARM is a full-featured port of Apple's [MLX framework](https://github.com/ml-explore/mlx) for universal ARM platforms. Unlike the original MLX which is tightly coupled to Apple Silicon and Metal, MLX-ARM provides:

- **Android-first approach**: Run LLMs on your smartphone or tablet
- **Vulkan-powered GPU acceleration**: Cross-vendor support (Mali, Adreno, PowerVR)
- **Gaming ecosystem integration**: Proton/Steam compatibility for AI-enhanced gaming
- **100% API compatibility**: Drop-in replacement for existing MLX Python code

## 🚀 Current Status

**🎉 v0.1.0-alpha - First Working Release!**

Successfully tested on **Samsung Galaxy Fold 5** (Snapdragon 8 Gen 2, Android 16)

### What Works Now ✅

- ✅ **Cross-compilation**: macOS → Android ARM64
- ✅ **CPU Backend**: NEON + SVE2 optimizations enabled
- ✅ **Device Abstraction**: CPU/GPU/NPU types
- ✅ **Memory Management**: Android unified memory support
- ✅ **Build System**: CMake + Android NDK r27
- ✅ **Real Device Testing**: Binary runs on Galaxy Fold 5
- ✅ **Computation Verified**: Vector addition 100% accurate

### Coming in v0.2.0 🚧

- ⏳ **Vulkan GPU Pipeline**: Full compute shader implementation
- ⏳ **GPU Acceleration**: Leverage Adreno 740 on Snapdragon 8 Gen 2
- ⏳ **Performance Benchmarks**: CPU vs GPU comparisons
- ⏳ **Matrix Operations**: GEMM kernels for ML workloads

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│      Python/Kotlin API (mlx.core)       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          C++ Core (mlx/core)            │
│  • Lazy evaluation                      │
│  • Unified memory management            │
│  • Device abstraction                   │
└─────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌──────────────┐       ┌──────────────┐
│ CPU Backend  │       │ GPU Backend  │
├──────────────┤       ├──────────────┤
│ • NEON       │       │ • Vulkan 1.3 │
│ • SVE/SVE2   │       │ • OpenCL 2.0 │
│ • OpenBLAS   │       │              │
└──────────────┘       └──────────────┘
```

## 🎮 Why Vulkan? The Steam/Proton Strategy

By choosing Vulkan as our primary GPU backend, we unlock unique opportunities:

- **Proton compatibility**: Valve's compatibility layer enables Steam integration
- **Gaming market access**: AI-powered NPCs, procedural generation, real-time ML
- **Steam Deck support**: Run ML models on portable gaming devices
- **Cross-platform**: Same code works on Android, Linux, Windows (via DXVK)

This positions MLX-ARM not just as a ML framework, but as a bridge to the **gaming ecosystem**.

## 📋 Supported Platforms (Planned)

| Platform | Priority | Status |
|----------|----------|--------|
| Android 8.0+ (ARM64) | P0 | 🚧 In Progress |
| ARM Linux (aarch64) | P0 | 📋 Planned |
| Snapdragon X Elite (Windows/Linux) | P1 | 📋 Planned |
| Steam Deck / Proton | P1 | 📋 Planned |

## 🔧 Requirements

### Development

- **Android NDK r25+** (for Android builds)
- **CMake 3.20+**
- **Vulkan SDK 1.3+** (LunarG or system packages)
- **C++17 compiler** (GCC 9+, Clang 10+)
- **glslangValidator** (for shader compilation)

### Runtime

- Android 8.0+ with Vulkan 1.1+ support
- ARM64 processor (ARMv8-A or later)
- GPU with Vulkan compute support (Adreno, Mali, PowerVR)

## 🚀 Quick Start (Coming Soon)

```kotlin
// Android Kotlin
val mlx = MLXContext()
mlx.setDefaultDevice(MLXDevice.GPU) // Vulkan

val A = mlx.random.normal(intArrayOf(1024, 1024))
val B = mlx.random.normal(intArrayOf(1024, 1024))
val C = mlx.matmul(A, B)

mlx.eval(C) // Executes on Adreno/Mali GPU via Vulkan
```

```python
# Python (Linux)
import mlx.core as mx

mx.set_default_device(mx.gpu)  # Vulkan backend

A = mx.random.normal((1024, 1024))
B = mx.random.normal((1024, 1024))
C = mx.matmul(A, B)

mx.eval(C)
```

## 📖 Documentation

- [Technical Specification](docs/technical_specification.md) - Complete architectural design
- [Build Instructions](docs/build.md) - Coming soon
- [API Reference](docs/api.md) - Coming soon
- [Contributing Guide](CONTRIBUTING.md) - Coming soon

## 🤝 Contributing

We're actively seeking contributors! Especially valuable:

- **Android developers** with JNI/NDK experience
- **Graphics programmers** familiar with Vulkan compute
- **ARM optimization experts** (NEON, SVE2)
- **ML researchers** for validation and benchmarking

### Getting Involved

1. Star ⭐ this repository to show support
2. Check out [open issues](../../issues) labeled `good first issue`
3. Join discussions in [GitHub Discussions](../../discussions)
4. Read the [Technical Specification](docs/technical_specification.md)

## 🗺️ Roadmap

### Phase 1: Android MVP (Q1 2026)
- ✅ Technical specification
- 🚧 Android build system
- 🚧 Vulkan compute backend
- 📋 Basic tensor operations

### Phase 2: LLM Support (Q2 2026)
- 📋 Quantization (Q4/Q8)
- 📋 Attention kernels
- 📋 Llama 3 inference

### Phase 3: Cross-Platform (Q3 2026)
- 📋 ARM Linux support
- 📋 Windows ARM support
- 📋 Python bindings

### Phase 4: Gaming Ecosystem (Q4 2026)
- 📋 Proton/Steam integration
- 📋 Unity/Unreal plugins
- 📋 Steam Workshop

## 🎯 Target Performance

**Llama 3 8B (Q4 quantized)**

| Device | Prompt (tok/s) | Generation (tok/s) |
|--------|----------------|-------------------|
| Snapdragon 8 Elite | >15 | >12 |
| Snapdragon X Elite | >25 | >20 |
| Raspberry Pi 5 (CPU) | >3 | >2 |

*Targets are preliminary estimates*

## � Contributors

This project is a **human-AI collaboration**:

- **[@jbarton43](https://github.com/jbarton43)** - Project lead, architecture decisions, device testing, strategic direction
- **AI Assistants (GitHub Copilot / Claude)** - Code implementation, documentation, technical research, pair programming

### Collaboration Model

MLX-ARM demonstrates modern AI-assisted software development:
- 🧠 **Human**: Vision, requirements, testing, quality control
- 🤖 **AI**: Implementation, documentation, optimization suggestions
- 🤝 **Together**: Iterative refinement, problem-solving, rapid prototyping

All code was written through human-AI pair programming sessions, with every architectural decision reviewed and approved by the human lead.

### Acknowledgments

- Apple's [MLX team](https://github.com/ml-explore/mlx) for the original framework
- Vulkan and OpenCL communities for cross-platform GPU computing
- Android development community for tools and guidance

## �📜 License

MIT License - same as upstream MLX

Copyright (c) 2026 MLX-ARM Contributors

## 🙏 Acknowledgments

- [Apple MLX Team](https://github.com/ml-explore/mlx) - Original framework
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - OpenCL inspiration
- [ncnn](https://github.com/Tencent/ncnn) - Vulkan mobile ML reference
- [Valve Proton](https://github.com/ValveSoftware/Proton) - Gaming ecosystem bridge

## 💬 Contact

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Twitter/X**: [@mlx_arm](#) - Coming soon

---

**Made with ❤️ for the open source and ARM community**

*"Bringing Apple's ML innovation to everyone, everywhere"*
