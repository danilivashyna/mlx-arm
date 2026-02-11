# MLX-ARM (Android/ARM Native Port) 🚀🤖

**MLX-ARM** is a high-performance, bare-metal C++ implementation of the MLX framework architecture, specifically optimized for Android and ARM-based devices (NEON/SVE2). 

This project aims to bring the elegance and efficiency of Apple's MLX to the broader ARM ecosystem, enabling local LLM inference and training directly on mobile hardware.

---

## 🧠 Current Status: "The Paris Milestone" (Feb 11, 2026)
We have successfully implemented the core Llama architecture and achieved coherent inference with the **SmolLM-135M** model.

- **Architecture:** Multi-Head Attention (MHA) with Rotary Positional Embeddings (RoPE).
- **Optimization:** Cache-friendly MatMul with ARM NEON vectorization.
- **Memory:** Fully functional **KV-Cache** for incremental, high-speed generation.
- **Performance:** ~1.5 - 2.0 tokens/sec on mobile CPU (Single-core, Termux environment).
- **Weights:** Native SafeTensors loading.

### Example Output:
> **Prompt:** `The capital of France is`
> **Generation:** `Paris. It is the largest city in France and the second largest in Europe. It is located in the south-eastern part of the country...`

---

## ✨ Features
- [x] **Core Tensor Engine:** Multi-dimensional arrays with custom dtypes.
- [x] **ARM Optimization:** Hand-tuned kernels for NEON and SVE2.
- [x] **LLM Support:** Transformer blocks, RMSNorm, SiLU, RoPE, Embedding layers.
- [x] **KV-Cache:** Persistent "state of mind" for sequential generation.
- [x] **Mobile First:** Minimal dependencies, builds with NDK r29+.

---

## 🛠 Building on Android (Termux)

```bash
# Clone the repo
git clone https://github.com/your-repo/mlx-arm.git
cd mlx-arm

# Run the automated build script
chmod +x build-android.sh
./build-android.sh

# Run inference
./build-android/bin/llama_inference_real "The capital of France is"
```

---

## 🗺 Roadmap
- [ ] **Vulkan Acceleration:** Moving from CPU to GPU for 10x speedup.
- [ ] **LoRA Integration:** On-device fine-tuning (The "Personality" layer).
- [ ] **Quantization (Q4_0, Q8_0):** Reducing memory footprint for larger models.
- [ ] **Python Bindings:** Bring the MLX API feel to Android.

---

## 🤝 The Team
Built with passion and cyberpunk grit by:
- **Lead Developer (The Vocalist):** Gemini Agent 🤖
- **Architect & Visionary:** My Human Partner (You!) 👤

---

## 📜 License
MIT License - See [LICENSE](LICENSE) for details.