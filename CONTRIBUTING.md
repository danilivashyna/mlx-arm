# Contributing to MLX-ARM

Thank you for your interest in contributing to MLX-ARM! We welcome contributions from the community.

## 🎯 How Can I Contribute?

### Reporting Bugs

- Use GitHub Issues with the `bug` label
- Include device information (model, Android version, GPU)
- Provide minimal reproduction code
- Attach logs if available

### Suggesting Features

- Use GitHub Issues with the `enhancement` label
- Explain the use case and benefit
- Consider implementation complexity

### Code Contributions

We're especially looking for help with:

- **Vulkan shaders** - Optimized compute kernels
- **NEON optimizations** - CPU fast paths
- **Android integration** - JNI, lifecycle management
- **Testing** - Unit tests, device-specific tests
- **Documentation** - API docs, tutorials, examples

## 🚀 Development Setup

### Prerequisites

- Android NDK r25+
- CMake 3.20+
- Vulkan SDK
- C++17 compiler
- Git

### Building from source

See [docs/build.md](build.md) for detailed build instructions.

```bash
git clone https://github.com/yourorg/mlx-arm.git
cd mlx-arm
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DANDROID_PLATFORM=android-26 \
         -DMLX_BUILD_TESTS=ON
make -j$(nproc)
```

### Running tests

```bash
ctest --output-on-failure
```

## 📝 Code Style

### C++

- Follow C++17 standard
- Use clang-format (provided `.clang-format` in root)
- Naming conventions:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `member_name_`

### GLSL (Shaders)

- Use Vulkan-compatible GLSL (#version 450)
- Document compute dimensions and buffer layouts
- Optimize for mobile GPUs (limit local memory usage)

### Example

```cpp
// Good
class VulkanDevice {
public:
    void initialize();
    bool is_available() const;
    
private:
    VkDevice device_;
    bool initialized_ = false;
};

// Bad
class vulkan_device {
public:
    void Initialize();
    bool IsAvailable();
    
private:
    VkDevice m_device;
};
```

## 🔄 Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-amazing-feature
   ```
3. **Make your changes**
   - Write tests for new functionality
   - Update documentation
   - Run clang-format
4. **Commit with descriptive messages**
   ```bash
   git commit -m "feat: Add Flash Attention kernel for Adreno"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/my-amazing-feature
   ```
6. **Open a Pull Request**
   - Describe what and why
   - Link related issues
   - Add screenshots/benchmarks if relevant

### Commit Message Format

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `perf:` Performance improvement
- `test:` Adding tests
- `refactor:` Code restructuring
- `chore:` Maintenance tasks

Examples:
```
feat(vulkan): Add tiled matmul shader with subgroups
fix(android): Handle device lost during compute
docs: Update build instructions for NDK r26
perf(neon): Optimize fp16 conversion using vcvt
```

## 🧪 Testing Guidelines

### Unit Tests

Located in `tests/`:

```cpp
#include <gtest/gtest.h>
#include "mlx/core/array.h"

TEST(ArrayTest, BasicAddition) {
    auto a = mlx::core::array({1.0f, 2.0f, 3.0f});
    auto b = mlx::core::array({4.0f, 5.0f, 6.0f});
    auto c = a + b;
    
    EXPECT_FLOAT_EQ(c[0], 5.0f);
}
```

### Device Tests

Test on real hardware when possible:

```bash
adb push build/tests/mlx_tests /data/local/tmp/
adb shell /data/local/tmp/mlx_tests --gtest_filter=VulkanTest.*
```

### Benchmarks

Use Google Benchmark:

```cpp
static void BM_MatMul(benchmark::State& state) {
    auto A = mlx::random::normal({1024, 1024});
    auto B = mlx::random::normal({1024, 1024});
    
    for (auto _ : state) {
        auto C = mlx::matmul(A, B);
        mlx::eval(C);
    }
}
BENCHMARK(BM_MatMul);
```

## 📚 Documentation

- Update README.md for user-facing changes
- Add API documentation in header files (Doxygen style)
- Create examples for new features
- Update technical specification if architecture changes

### Documentation Style

```cpp
/**
 * Compute matrix multiplication C = A × B
 * 
 * @param A Input matrix (M × K)
 * @param B Input matrix (K × N)
 * @return Result matrix (M × N)
 * 
 * @throws std::invalid_argument if dimensions don't match
 * 
 * @note This operation is lazily evaluated
 * @see https://mlx-arm.readthedocs.io/matmul
 */
Array matmul(const Array& A, const Array& B);
```

## 🏷️ Issue Labels

- `good first issue` - Easy for newcomers
- `help wanted` - We need assistance
- `bug` - Something isn't working
- `enhancement` - New feature request
- `documentation` - Documentation improvements
- `performance` - Optimization opportunities
- `android` - Android-specific
- `vulkan` - Vulkan backend
- `cpu` - CPU backend

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Standards

- Be respectful and constructive
- Accept feedback gracefully
- Focus on what's best for the community
- Show empathy towards others

## 📞 Getting Help

- **GitHub Discussions**: General questions
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Real-time chat (coming soon)

## 🎓 Learning Resources

### Vulkan Compute
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [SaschaWillems Vulkan Samples](https://github.com/SaschaWillems/Vulkan)
- [Khronos Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples)

### ARM Optimization
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics/)
- [Optimizing for ARM](https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog)

### Android NDK
- [Android NDK Guide](https://developer.android.com/ndk/guides)
- [JNI Tips](https://developer.android.com/training/articles/perf-jni)

## 🙏 Attribution

MLX-ARM builds upon the excellent work of:
- Apple MLX team
- llama.cpp community
- ncnn (Tencent)
- Vulkan ecosystem

Thank you for contributing! 🚀
