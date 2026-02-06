# MLX-ARM Project Roadmap

## Vision

Bring MLX to universal ARM platforms with Android-first approach and gaming ecosystem integration via Proton/Steam.

---

## Q1 2026 - Foundation

### Milestone 1: Android Infrastructure ✅ IN PROGRESS
**Target:** Late February 2026

- [x] Technical specification
- [x] Repository structure
- [x] CMake build system for Android NDK
- [x] Device abstraction layer (device.h, allocator.h, backend.h)
- [x] Vulkan backend scaffold
- [x] Vector add proof-of-concept shader
- [ ] Android JNI bindings
- [ ] Basic CPU backend (NEON elementwise ops)
- [ ] CI/CD with GitHub Actions
- [ ] First demo app

**Success Criteria:**
- Vector addition runs on Android device via Vulkan
- CPU fallback works for simple operations
- Build is reproducible on different machines

---

## Q1-Q2 2026 - GPU Acceleration

### Milestone 2: Vulkan GPU Backend
**Target:** April 2026

- [ ] Vulkan pipeline creation and caching
- [ ] GLSL → SPIR-V compilation (offline + runtime)
- [ ] Elementwise operations (add, mul, div, etc.)
- [ ] MatMul shader (FP32, naive)
- [ ] MatMul shader (FP32, tiled + subgroups)
- [ ] VkBuffer management and CPU↔GPU transfers
- [ ] Async execution with timeline semaphores
- [ ] VMA (Vulkan Memory Allocator) integration
- [ ] Benchmarking framework

**Success Criteria:**
- 1024×1024 matmul faster on GPU than CPU
- <10ms latency for kernel dispatch
- Memory usage within 10% of theoretical minimum

---

## Q2 2026 - LLM Support

### Milestone 3: Language Model Inference
**Target:** June 2026

- [ ] RMSNorm / LayerNorm kernels
- [ ] RoPE (rotary position embeddings)
- [ ] Softmax (numerically stable)
- [ ] Attention (basic implementation)
- [ ] Quantization support:
  - [ ] Q4_0 (4-bit symmetric)
  - [ ] Q4_1 (4-bit asymmetric)
  - [ ] Q8_0 (8-bit symmetric)
  - [ ] FP16 (half precision)
- [ ] Unified memory via Gralloc/AHardwareBuffer
- [ ] KV-cache operations
- [ ] Integration with mlx-lm (Python)

**Success Criteria:**
- Llama 3 8B (4-bit) runs on Snapdragon 8 Elite
- >15 tok/s prompt processing
- >12 tok/s generation
- Memory usage <6GB

---

## Q3 2026 - Optimization & Cross-Platform

### Milestone 4: Performance & Expansion
**Target:** August 2026

- [ ] Flash Attention v2 for mobile GPUs
- [ ] SVE2 CPU optimizations
- [ ] Kernel fusion (MLP, Attention)
- [ ] Advanced quantization:
  - [ ] Q4_K (k-quants)
  - [ ] Q5_K, Q6_K
  - [ ] MXFP8 (Microscaling FP8)
- [ ] Memory pooling and reuse
- [ ] Profile-guided optimizations
- [ ] ARM Linux support
- [ ] Windows ARM support (Snapdragon X Elite)
- [ ] Python bindings (pybind11)

**Success Criteria:**
- Llama 3 8B: >25 tok/s (prompt) + >20 tok/s (gen) on desktop ARM
- >20 tok/s on flagship smartphones
- Linux/Windows ARM working with same codebase

---

## Q4 2026 - Gaming Ecosystem

### Milestone 5: Proton/Steam Integration
**Target:** November 2026

- [ ] Proton compatibility testing
- [ ] Steam Deck validation
- [ ] Unity plugin (experimental)
- [ ] Unreal Engine plugin (experimental)
- [ ] Steam Workshop integration
- [ ] Documentation for game developers
- [ ] Example: AI NPC demo
- [ ] Example: Real-time style transfer
- [ ] Example: Procedural generation with ML

**Success Criteria:**
- MLX-ARM apps run via Proton on Steam
- Steam Deck runs Llama 3 7B at >10 tok/s
- At least 1 indie game using MLX-ARM for AI

---

## 2027 - Community & Ecosystem

### Phase 1: Maturity (Q1-Q2)
- [ ] v1.0.0 stable release
- [ ] Full API parity with MLX
- [ ] >80% test coverage
- [ ] Production-ready documentation
- [ ] Performance optimizations for specific devices
- [ ] Community showcase gallery

### Phase 2: Expansion (Q3-Q4)
- [ ] Vision models support (ConvNets, ViT)
- [ ] Audio processing (Whisper, TTS)
- [ ] Multi-GPU support
- [ ] Distributed inference
- [ ] Model quantization tools
- [ ] GUI tools for non-developers

---

## Future Considerations

### Potential Features (Post-2027)
- **NPU Integration** - Qualcomm Hexagon, Mali NPU
- **RISC-V Support** - Emerging platform
- **WebGPU Backend** - Browser support
- **Model Training** - Not just inference
- **Federated Learning** - Privacy-preserving training
- **ONNX Compatibility** - Import/export models

### Partnerships & Collaboration
- **Qualcomm** - NPU optimization, driver support
- **ARM** - SVE2 optimization guidance
- **Valve** - Proton integration, Steam Deck optimization
- **Unity/Epic** - Official plugin support
- **Game Studios** - Real-world use cases

---

## Success Metrics

### Technical Metrics
- **Performance:** Within 20% of llama.cpp on same hardware
- **Memory:** <10% overhead vs theoretical minimum
- **Compatibility:** Works on >90% of ARM devices (2023+)
- **Stability:** <0.1% crash rate in production

### Community Metrics
- **Contributors:** >50 active contributors by end of 2026
- **Stars:** >5,000 GitHub stars
- **Downloads:** >10,000 monthly PyPI downloads
- **Integrations:** Used in >10 production applications

### Ecosystem Metrics
- **Gaming:** >3 games using MLX-ARM for AI
- **Steam:** MLX-ARM apps available on Steam
- **Mobile:** >5 Android apps on Play Store
- **Education:** Used in >3 university courses

---

## Risk Management

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Vulkan driver bugs | High | CPU fallback, driver workarounds |
| Performance below target | High | Early profiling, expert consultation |
| Limited contributor base | Medium | Good first issues, mentorship |
| API divergence from MLX | Medium | Automated compatibility tests |
| Funding shortage | Low | Minimal dependencies, volunteer-driven |

---

## Get Involved

We need help! Priority areas:

1. **Android developers** - JNI, lifecycle, testing
2. **Graphics programmers** - Vulkan kernel optimization
3. **ML researchers** - Model validation, benchmarks
4. **Technical writers** - Documentation, tutorials
5. **Game developers** - Unity/Unreal integration

👉 Check [CONTRIBUTING.md](CONTRIBUTING.md) to get started!

---

**Last Updated:** February 6, 2026  
**Next Review:** End of Q1 2026
