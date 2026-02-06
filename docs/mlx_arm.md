Написать библиотеку MLX под ARM Snapdragon 

Вот что стоит учесть:
Основные сложности:
1. Архитектурные различины
MLX оптимизирована под Apple Silicon (ARM с расширениями Apple), а Snapdragon использует ARM с другим набором расширений (Qualcomm Hexagon DSP, Adreno GPU). Придётся адаптировать низкоуровневые операции под специфику Snapdragon.
2. Вычислительные backends

MLX использует Metal для GPU на Apple устройствах
На Snapdragon нужно будет использовать OpenCL, Vulkan Compute или проприетарный Qualcomm Neural Processing SDK
Hexagon DSP требует отдельной работы для максимальной производительности

3. Объём работы

Нужно переписать все SIMD-оптимизации под ARM NEON (если ещё не универсальны)
Адаптировать граф вычислений и memory management
Реализовать эффективные kernel'ы для основных операций (matmul, conv, attention и т.д.)

Что нужно сделать:

Backend для Adreno GPU (OpenCL/Vulkan)

Портирование всех kernel'ов из Metal
Оптимизация subgroups для matmul
Реализация Flash Attention для мобильных GPU


Unified Memory через dma-buf 

Кастомный аллокатор
Интеграция с Gralloc HAL
Управление когерентностью кэша


Hexagon NPU интеграция 

Либо через Qualcomm SDK (статические графы — противоречит философии MLX)
Либо через TVM (сложно, но даст динамичность)


CPU оптимизации (NEON/SVE2) 

Переписать critical path под новые инструкции