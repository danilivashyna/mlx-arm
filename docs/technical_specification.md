# Техническое задание: MLX для универсальных ARM-платформ

**Версия:** 1.0  
**Дата:** 6 февраля 2026  
**Статус:** Проектирование

---

## 1. Цель проекта

Создать полнофункциональный порт фреймворка MLX для универсальных ARM-платформ (Linux/Android), обеспечивающий:
- Кроссплатформенность (не привязан к Apple Silicon или конкретным вендорам)
- Высокую производительность через GPU (OpenCL/Vulkan) и CPU (NEON/SVE2)
- Совместимость с существующим Python API MLX
- Возможность запуска больших языковых моделей (LLM) на ARM-устройствах

**Не входит в scope v1.0:**
- Поддержка проприетарных SDK (Qualcomm NPU, Mali NPU и т.д.)
- Специфичные для вендоров оптимизации
- Поддержка iOS/macOS (уже есть)

---

## 2. Архитектура решения

### 2.1. Целевые платформы

| Платформа | Приоритет | Примеры устройств |
|-----------|-----------|-------------------|
| Android NDK (aarch64) | P0 | Смартфоны, планшеты (первая целевая платформа) |
| ARM Linux (aarch64) | P0 | Raspberry Pi 5, Orange Pi, серверы ARM |
| Snapdragon X Elite (Linux/Windows) | P1 | Ноутбуки |
| Steam Deck / Proton (через Vulkan) | P1 | Gaming devices, возможность запуска через Steam |
| RISC-V (перспектива) | P2 | Будущие платформы |

### 2.2. Архитектура компонентов

```
┌─────────────────────────────────────────┐
│         Python API (mlx.core)           │
│  (без изменений, полная совместимость)  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│          C++ Core (mlx/core)            │
│  - Graph Builder (ленивые вычисления)   │
│  - Memory Manager (унифицированная)     │
│  - Device Abstraction Layer             │
└─────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌──────────────┐       ┌──────────────┐
│ CPU Backend  │       │ GPU Backend  │
├──────────────┤       ├──────────────┤
│ - NEON       │       │ - OpenCL     │
│ - SVE/SVE2   │       │ - Vulkan     │
│ - OpenBLAS   │       │ - WebGPU*    │
└──────────────┘       └──────────────┘
```

*WebGPU - опциональный универсальный слой

---

## 3. Технические требования

### 3.1. Вычислительные backends

#### 3.1.1. GPU Backend (приоритет P0)

**Vulkan Compute 1.3+**
- **Цель:** Основной GPU backend для универсальности и будущего
- **Ключевые преимущества:**
  - Нативная интеграция с Android (API 26+)
  - Портируемость через Proton → возможность запуска в Steam
  - Явный контроль над памятью и синхронизацией
  - Cross-vendor support (Mali, Adreno, PowerVR, desktop GPUs)
  - Перспектива для gaming/consumer adoption
- **Требования:**
  - VK_KHR_shader_float16_int8 (для FP16)
  - VK_KHR_8bit_storage, VK_KHR_16bit_storage
  - Subgroup operations (VK_VERSION_1_1+)
  - Shared memory минимум 32KB
- **Реализация:**
  - Instance, Physical Device, Logical Device management
  - Descriptor sets для buffer binding
  - Compute pipelines с кешированием
  - Command buffer recording и submission
  - Timeline semaphores для sync (VK_KHR_timeline_semaphore)

**OpenCL 2.0+ (приоритет P1)**
- **Цель:** Fallback для старых устройств и legacy support
- **Преимущества:**
  - Более зрелые драйверы на некоторых платформах
  - Проще для прототипирования (меньше boilerplate)
  - Хорошая база знаний от llama.cpp
- **Когда использовать:** 
  - Устройства без Vulkan 1.1+
  - Быстрое прототипирование новых kernel'ов

**WebGPU (приоритет P2)**
- **Цель:** Универсальный кроссплатформенный слой
- **Преимущества:**
  - Автоматическая трансляция в Metal/Vulkan/D3D12
  - Упрощает поддержку новых платформ
- **Недостатки:**
  - Дополнительный слой абстракции
  - Потенциальные потери производительности

#### 3.1.2. CPU Backend (приоритет P0)

**ARM NEON (базовый уровень)**
- 128-bit SIMD векторные инструкции
- Обязательная поддержка для всех ARMv8-A+ процессоров
- Реализация базовых операций:
  - Vectorized elementwise ops
  - Basic reductions
  - Fast memory copy/transpose

**ARM SVE/SVE2 (оптимизация)**
- Векторы переменной длины (128-2048 bits)
- Предикатные операции
- Расширенные матричные инструкции
- Runtime detection (через `getauxval(AT_HWCAP)`)

**Fallback: OpenBLAS/BLIS**
- Для BLAS операций (GEMM, GEMV)
- Используется если нет GPU или для малых матриц
- Multithreading через OpenMP

### 3.2. Управление памятью

#### 3.2.1. Унифицированная память (Unified Memory)

**Проблема:**  
На ARM-устройствах (кроме Apple Silicon) унифицированная память не поддерживается аппаратно на уровне кэш-когерентности между CPU и GPU.

**Решение для Linux:**
```
┌─────────────────────────────────────┐
│  MLX Unified Allocator              │
├─────────────────────────────────────┤
│  DMA-BUF (Linux kernel)             │
│  - dma_heap_alloc()                 │
│  - Zero-copy sharing                │
│  - Explicit cache management        │
└─────────────────────────────────────┘
```

**Решение для Android:**
```
┌─────────────────────────────────────┐
│  MLX Unified Allocator              │
├─────────────────────────────────────┤
│  Android Gralloc HAL                │
│  - Allocate shared buffers          │
│  - ION/DMA-BUF integration          │
│  - AHardwareBuffer API              │
└─────────────────────────────────────┘
```

**Ключевые функции:**
- `mlx_alloc_shared()` - выделение памяти, доступной CPU и GPU
- `mlx_cache_sync()` - ручная синхронизация кэша при необходимости
- Lazy allocation - реальное выделение только при материализации
- Reference counting для автоматического освобождения

#### 3.2.2. Требования к аллокатору

| Функция | Описание | Приоритет |
|---------|----------|-----------|
| Выделение выровненной памяти | Alignment 64B для NEON | P0 |
| Zero-copy buffers | Shared CPU/GPU без копирования | P0 |
| Memory pooling | Переиспользование буферов | P1 |
| Huge pages | Для больших моделей (2MB pages) | P2 |
| Memory pressure handling | Graceful degradation при нехватке | P1 |

### 3.3. Критичные операции (kernels)

Минимальный набор операций для запуска LLM:

#### Приоритет P0 (MVP)

| Операция | CPU (NEON) | GPU (OpenCL) | Сложность |
|----------|------------|--------------|-----------|
| MatMul (FP32/FP16) | OpenBLAS | Tiled + Subgroups | Высокая |
| MatMul (Quantized Q4/Q8) | Custom | Dequant on-the-fly | Очень высокая |
| Elementwise (add, mul, etc) | Векторизовано | Trivial kernels | Низкая |
| RMSNorm / LayerNorm | Fused kernel | Fused kernel | Средняя |
| RoPE (rotary embeddings) | Fused | Fused | Средняя |
| Softmax | Fused | Reduction + exp | Средняя |
| Копирование/транспонирование | NEON optimized | Coalesced access | Низкая |

#### Приоритет P1 (Оптимизация)

| Операция | Описание | Зависимости |
|----------|----------|-------------|
| SDPA (Scaled Dot-Product Attention) | Flash Attention v2 | Shared memory tiling |
| Fused MLP | GeLU + MatMul fusion | Kernel fusion framework |
| KV-cache operations | Efficient cache management | Memory optimization |
| Grouped Query Attention | GQA для Llama 3 | SDPA implementation |

#### Приоритет P2 (Расширенная функциональность)

- Conv2D (для vision models)
- BatchNorm
- Dropout
- Embeddings lookup

### 3.4. Квантование

**Поддерживаемые форматы (P0):**
- Q4_0: 4-bit symmetric, block size 32
- Q4_1: 4-bit asymmetric с bias
- Q8_0: 8-bit symmetric
- FP16: half precision

**Поддерживаемые форматы (P1):**
- Q4_K: k-quants от llama.cpp (лучшее качество)
- Q5_K, Q6_K: варианты с большей точностью
- MXFP8: Microscaling FP8

**Реализация:**
- Dequantization в GPU kernel'ах (on-the-fly)
- NEON оптимизированная деквантизация для CPU
- Zero memory overhead (не храним развернутые веса)

---

## 4. Компоненты системы

### 4.1. Core Components

#### 4.1.1. Device Abstraction (`mlx/core/device.h`)

```cpp
enum class DeviceType {
    CPU,
    GPU,
    NPU  // Reserved for future
};

class Device {
public:
    DeviceType type() const;
    std::string name() const;
    size_t memory_capacity() const;
    
    // Capabilities
    bool supports_fp16() const;
    bool supports_bf16() const;
    bool supports_int8() const;
};

class DeviceManager {
public:
    static Device default_device();
    static std::vector<Device> available_devices();
    static void set_default_device(Device device);
};
```

#### 4.1.2. Memory Manager (`mlx/core/allocator.h`)

```cpp
class SharedAllocator {
public:
    virtual void* allocate(size_t size, size_t alignment) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual bool is_shared() const = 0;  // CPU/GPU shared?
    virtual void sync_to_device(void* ptr, size_t size) = 0;
    virtual void sync_to_host(void* ptr, size_t size) = 0;
};

// Platform-specific implementations
class DMABufAllocator : public SharedAllocator { /*...*/ };
class GrallocAllocator : public SharedAllocator { /*...*/ };
```

#### 4.1.3. Backend Interface (`mlx/core/backend.h`)

```cpp
class Backend {
public:
    virtual void execute(
        const std::vector<Primitive*>& primitives,
        const StreamContext& stream
    ) = 0;
    
    virtual void compile_kernel(const Kernel& kernel) = 0;
    virtual std::shared_ptr<Buffer> allocate_buffer(size_t size) = 0;
};

class OpenCLBackend : public Backend { /*...*/ };
class VulkanBackend : public Backend { /*...*/ };
class CPUBackend : public Backend { /*...*/ };
```

### 4.2. Vulkan Backend Structure

```
mlx/backend/vulkan/
├── vulkan_context.h/cpp       # Instance, device, queue management
├── vulkan_device.h/cpp        # Physical/logical device
├── vulkan_buffer.h/cpp        # VkBuffer management
├── vulkan_allocator.h/cpp     # VMA (Vulkan Memory Allocator) integration
├── vulkan_pipeline.h/cpp      # Compute pipeline & descriptor sets
├── vulkan_command.h/cpp       # Command buffer recording/submission
└── shaders/                   # GLSL → SPIR-V
    ├── matmul.comp            # Matrix multiplication
    ├── elementwise.comp       # Unary/binary ops
    ├── reduction.comp         # Sum, max, etc
    ├── normalization.comp     # RMSNorm, LayerNorm
    ├── attention.comp         # SDPA kernels
    └── quantized.comp         # Quantized operations
```

### 4.3. OpenCL Backend Structure (Fallback)

```
mlx/backend/opencl/
├── opencl_context.h/cpp       # OpenCL context management
├── opencl_device.h/cpp        # Device enumeration
├── opencl_buffer.h/cpp        # Buffer management
├── opencl_kernel.h/cpp        # Kernel compilation & caching
├── opencl_stream.h/cpp        # Command queue wrapper
└── kernels/
    ├── matmul.cl              # Matrix multiplication
    ├── elementwise.cl         # Unary/binary ops
    ├── reduction.cl           # Sum, max, etc
    ├── normalization.cl       # RMSNorm, LayerNorm
    ├── attention.cl           # SDPA kernels
    └── quantized.cl           # Quantized operations
```

### 4.3. CPU Backend Structure

```
mlx/backend/cpu/
├── cpu_backend.h/cpp          # Main CPU backend
├── cpu_primitives.h/cpp       # Operation implementations
├── neon/
│   ├── neon_matmul.cpp       # NEON optimized GEMM
│   ├── neon_elementwise.cpp  # Vectorized ops
│   └── neon_utils.h          # NEON intrinsics helpers
├── sve/
│   ├── sve_matmul.cpp        # SVE2 optimized (if available)
│   └── sve_detection.cpp     # Runtime CPU feature detection
└── blas/
    └── blas_wrapper.cpp      # OpenBLAS integration
```

---

## 5. Система сборки и зависимости

### 5.1. Зависимости

**Обязательные (P0):**
- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+)
- Python 3.9+ (для биндингов)
- pybind11 2.10+
- OpenCL headers & ICD loader
- OpenBLAS или BLIS

**Опциональные:**
- Vulkan SDK (для Vulkan backend)
- Android NDK r25+ (для Android)
- LLVM/Clang (для JIT компиляции kernel'ов)

### 5.2. CMake структура

```cmake
# CMakeLists.txt (root)
project(mlx-arm VERSION 0.1.0)

option(MLX_BUILD_OPENCL "Build OpenCL backend" ON)
option(MLX_BUILD_VULKAN "Build Vulkan backend" OFF)
option(MLX_BUILD_PYTHON "Build Python bindings" ON)
option(MLX_ENABLE_NEON "Enable NEON optimizations" ON)
option(MLX_ENABLE_SVE "Enable SVE/SVE2 if available" ON)

# Auto-detect CPU features
include(CheckCPUFeatures)

# Backends
add_subdirectory(mlx/backend/cpu)
if(MLX_BUILD_OPENCL)
    add_subdirectory(mlx/backend/opencl)
endif()
```

### 5.3. Кросс-компиляция для Android

```bash
# Toolchain file
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DMLX_BUILD_OPENCL=ON \
  -DMLX_BUILD_PYTHON=OFF \
  ..
```

---

## 6. API и совместимость

### 6.1. Принцип совместимости

**100% API compatibility** с оригинальным MLX для Python:

```python
import mlx.core as mx

# Все существующие скрипты работают без изменений
a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])
c = a + b
mx.eval(c)

# Device management
mx.set_default_device(mx.gpu)  # Использует OpenCL/Vulkan
mx.set_default_device(mx.cpu)  # Использует NEON/SVE2
```

### 6.2. Расширения для ARM

**Новый модуль:** `mlx.arm` (опциональный)

```python
import mlx.arm

# Информация о платформе
info = mlx.arm.platform_info()
print(f"CPU: {info.cpu_model}")
print(f"NEON: {info.has_neon}")
print(f"SVE2: {info.has_sve2}")
print(f"GPU: {info.gpu_name}")
print(f"OpenCL: {info.opencl_version}")

# Специфичные настройки
mlx.arm.set_opencl_platform(0)  # Выбор OpenCL platform
mlx.arm.enable_kernel_cache(True)  # Кеширование скомпилированных kernel'ов
mlx.arm.set_shared_memory_mode("dma-buf")  # Linux
mlx.arm.set_shared_memory_mode("gralloc")  # Android
```

---

## 7. План разработки

### 7.1. Milestone 1: Инфраструктура (2-3 месяца)

**Цели:**
- [ ] CMake build система с кросс-компиляцией
- [ ] Device abstraction layer
- [ ] Basic CPU backend (NEON elementwise ops)
- [ ] OpenCL context management
- [ ] Memory allocator (basic, без zero-copy)
- [ ] Python bindings (pybind11)
- [ ] CI/CD на ARM Linux

**Критерий успеха:**  
Запускается простой пример:
```python
import mlx.core as mx
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
c = mx.add(a, b)  # На CPU через NEON
print(mx.eval(c))
```

### 7.2. Milestone 2: GPU Backend (2-3 месяца)

**Цели:**
- [ ] OpenCL kernel компиляция и кеширование
- [ ] Базовые elementwise операции на GPU
- [ ] MatMul kernel (FP32, naive реализация)
- [ ] MatMul kernel (FP32, tiled + subgroups)
- [ ] Memory transfers CPU↔GPU
- [ ] Async execution с event tracking
- [ ] Benchmarking framework

**Критерий успеха:**  
Матричное умножение на GPU работает быстрее CPU:
```python
import mlx.core as mx
mx.set_default_device(mx.gpu)
A = mx.random.normal((1024, 1024))
B = mx.random.normal((1024, 1024))
C = mx.matmul(A, B)
mx.eval(C)  # Выполняется на GPU через OpenCL
```

### 7.3. Milestone 3: LLM Support (3-4 месяца)

**Цели:**
- [ ] RMSNorm / LayerNorm kernels
- [ ] RoPE (rotary embeddings)
- [ ] Softmax
- [ ] Attention (базовая реализация)
- [ ] Квантование Q4_0, Q4_1, Q8_0
- [ ] Unified memory через DMA-BUF (Linux)
- [ ] KV-cache операции
- [ ] Интеграция с mlx-lm

**Критерий успеха:**  
Запускается Llama 3 8B (quantized) для генерации:
```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
response = generate(model, tokenizer, prompt="Hello, world!", max_tokens=100)
print(response)
```

Производительность: >15 tokens/sec на Snapdragon X Elite

### 7.4. Milestone 4: Оптимизация (2-3 месяца)

**Цели:**
- [ ] Flash Attention для мобильных GPU
- [ ] SVE2 CPU оптимизации
- [ ] Kernel fusion (MLP, Attention)
- [ ] Улучшенное квантование (Q4_K, MXFP8)
- [ ] Memory pooling и reuse
- [ ] Profile-guided optimizations
- [ ] Android NDK поддержка
- [ ] Gralloc allocator для Android

**Критерий успеха:**
- Llama 3 8B: >25 tokens/sec (prompt) + >20 tokens/sec (generation)
- Memory overhead <10% относительно размера модели
- Работает на Android-смартфонах

### 7.5. Milestone 5: Polishing (1-2 месяца)

**Цели:**
- [ ] Документация (Getting Started, API Reference)
- [ ] Примеры и туториалы
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests на реальных устройствах
- [ ] Performance regression tests
- [ ] Contribution guidelines
- [ ] Docker images для разработки

**Критерий успеха:**
- Публичный релиз v0.1.0
- Документация на уровне оригинального MLX
- Работает на >5 различных ARM-устройствах

---

## 8. Метрики успеха

### 8.1. Функциональные метрики

| Метрика | Target | Критичность |
|---------|--------|-------------|
| Совместимость Python API | 100% | P0 |
| Поддерживаемые операции | >50 | P0 |
| Поддержка квантования | Q4/Q8/FP16 | P0 |
| Размер моделей | До 70B (quantized) | P1 |

### 8.2. Производительные метрики

**Benchmarks: Llama 3 8B (Q4_K)**

| Устройство | Prompt (tok/s) | Generation (tok/s) |
|------------|----------------|-------------------|
| Snapdragon X Elite | >25 | >20 |
| Snapdragon 8 Elite (mobile) | >15 | >12 |
| Raspberry Pi 5 (CPU only) | >3 | >2 |

**Сравнение с конкурентами:**
- llama.cpp (OpenCL): сопоставимо ±10%
- Ollama (CPU): быстрее в 2-3x на GPU
- PyTorch Mobile: быстрее в 3-5x

### 8.3. Качественные метрики

- Latency первого токена: <500ms
- Memory overhead: <10%
- Power efficiency: сопоставимо с llama.cpp
- Crash-free rate: >99.9%

---

## 9. Риски и митигация

| Риск | Вероятность | Влияние | Митигация |
|------|-------------|---------|-----------|
| Производительность OpenCL хуже ожидаемой | Средняя | Высокое | Vulkan fallback, профилирование на ранних этапах |
| Проблемы с драйверами GPU на разных устройствах | Высокая | Среднее | CPU fallback, тестирование на широком спектре устройств |
| Сложность unified memory | Средняя | Высокое | Поэтапная реализация: сначала copy, потом zero-copy |
| Недостаточная поддержка SVE2 | Низкая | Низкое | NEON как fallback |
| Размер сообщества разработчиков | Высокая | Среднее | Открытый код, good first issues, подробная документация |
| Изменения в upstream MLX ломают совместимость | Средняя | Высокое | CI с регрессионными тестами, tracking upstream |

---

## 10. Команда и ресурсы

### 10.1. Необходимые компетенции

**Core team (минимум):**
- 1x Lead Engineer (C++, GPU programming)
- 1x ML Engineer (знание LLM, трансформеры)
- 1x Systems Engineer (Linux kernel, memory management)
- 1x DevOps (CI/CD, кросс-компиляция)

**Part-time:**
- Android Engineer (для Milestone 4)
- Technical Writer (документация)

### 10.2. Оборудование для тестирования

**Критично (P0):**
- Snapdragon X Elite notebook (Linux)
- ARM workstation / cloud instance (AWS Graviton)
- Raspberry Pi 5 (8GB)

**Желательно (P1):**
- Snapdragon 8 Elite smartphone (Android)
- Orange Pi / Rock Pi с Mali GPU
- Jetson Nano/Orin (для сравнения с CUDA)

---

## 11. Лицензирование

**Предлагаемая лицензия:** MIT (как у оригинального MLX)

**Обоснование:**
- Максимальная открытость и доступность
- Совместимость с upstream MLX
- Позволяет коммерческое использование
- Упрощает contribution от сообщества

**Зависимости:**
- OpenCL ICD: Apache 2.0 ✓
- OpenBLAS: BSD 3-Clause ✓
- pybind11: BSD 3-Clause ✓
- LLVM (опционально): Apache 2.0 ✓

---

## 12. Следующие шаги

### 12.1. Немедленные действия (неделя 1-2)

1. **Setup репозитория:**
   - GitHub repo с базовой структурой
   - CMake skeleton
   - CI/CD pipeline (GitHub Actions)
   - CONTRIBUTING.md, CODE_OF_CONDUCT.md

2. **Research spike:**
   - Изучить OpenCL backend из llama.cpp
   - Проанализировать CUDA backend MLX
   - Benchmark OpenBLAS vs собственные NEON kernel'ы

3. **Proof of concept:**
   - Простейший OpenCL kernel (vector add)
   - NEON optimized matmul (малые размеры)
   - Замер производительности

4. **Community:**
   - Пост на Reddit r/LocalLLaMA
   - Обсуждение в MLX GitHub Discussions
   - Связаться с maintainers llama.cpp (поделиться опытом)

### 12.2. Документация (неделя 3-4)

- Architecture Decision Records (ADR)
- Development setup guide
- Contribution workflow
- API design document

---

## 13. Приложения

### Приложение A: Сравнение с альтернативами

| Фреймворк | Платформы | GPU | Квантование | Ленивые вычисления | Динамические графы |
|-----------|-----------|-----|-------------|-------------------|-------------------|
| **MLX (Apple)** | macOS, iOS | Metal | Q4/Q8/FP16 | ✓ | ✓ |
| **MLX-ARM (этот проект)** | Linux, Android | OpenCL/Vulkan | Q4/Q8/FP16 | ✓ | ✓ |
| **llama.cpp** | Все | OpenCL, Vulkan | Множество | ✗ | ✗ |
| **PyTorch Mobile** | Android, iOS | Metal, Vulkan | Limited | ✗ | ✗ |
| **ONNX Runtime** | Все | Multiple | INT8 | ✗ | ✗ |

**Уникальное преимущество MLX-ARM:**  
Единственный фреймворк с ленивыми вычислениями и динамическими графами для ARM Linux/Android

### Приложение B: Список референсных kernel'ов

Kernel'ы для анализа и портирования:

**Из llama.cpp:**
- `ggml-opencl.cpp` - OpenCL backend для Adreno
- `ggml-quants.c` - Квантование Q4/Q8

**Из MLX CUDA:**
- `mlx/backend/cuda/matmul.cu` - Tiled matmul
- `mlx/backend/cuda/primitives.cu` - Базовые операции

**Из других проектов:**
- CLBlast - оптимизированный GEMM для OpenCL
- TVM - compiler для генерации kernel'ов
- Halide - DSL для image processing (применимо для tensor ops)

### Приложение C: Профилирование и отладка

**Инструменты:**
- `perfetto` - системный профайлер для Android/Linux
- `gdb` с поддержкой OpenCL (для отладки kernel'ов)
- `CodeXL` (AMD) - анализ OpenCL производительности
- `Arm Mobile Studio` - профилирование Mali GPU
- `Snapdragon Profiler` - для Adreno GPU

**Метрики для отслеживания:**
- Kernel execution time
- Memory bandwidth utilization
- Cache hit rate
- Power consumption (через `powertop` / Android Battery Historian)

---

## 14. Глоссарий

- **AMX** - Advanced Matrix Extensions (Apple Silicon)
- **NEON** - ARM SIMD расширения (128-bit)
- **SVE/SVE2** - Scalable Vector Extensions (128-2048 bit)
- **DMA-BUF** - Linux framework для zero-copy буферов
- **Gralloc** - Android Graphics Allocator HAL
- **Subgroups** - OpenCL/Vulkan механизм для синхронизации внутри workgroup
- **Flash Attention** - Memory-efficient attention algorithm
- **KV-cache** - Key-Value cache для autoregressive generation
- **Q4_0/Q4_1** - 4-bit квантование (symmetric/asymmetric)

---

## Заключение

Проект MLX-ARM является амбициозным, но технически реализуемым. Ключевыми факторами успеха будут:

1. **Фокус на MVP** - сначала работающая версия, потом оптимизация
2. **Открытость** - активное вовлечение сообщества
3. **Тестирование** - на реальных устройствах с первых дней
4. **Модульность** - возможность добавлять новые backends

Ожидаемый timeline: **10-14 месяцев** до production-ready v1.0

**Вопросы для обсуждения:**
- Начать с OpenCL или Vulkan?
- Нужен ли JIT компилятор kernel'ов или достаточно предкомпиляции?
- Android поддержка в MVP или отложить?
- Стратегия monetization (если планируется)?

---

**Контакты и координация:**  
GitHub: [будет создан]  
Discord: [опционально]  
Email: [будет указан]
