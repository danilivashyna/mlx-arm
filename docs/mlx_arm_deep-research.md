Архитектурный синтез и техническая стратегия портирования фреймворка MLX на платформу Qualcomm Snapdragon ARM: Системный анализ и дорожная карта реализации
Переход к эре повсеместных вычислений в области искусственного интеллекта на периферийных устройствах обусловил острую необходимость в высокопроизводительных тензорных библиотеках, способных максимально эффективно использовать гетерогенные архитектуры мобильных систем на кристалле (SoC). Фреймворк MLX, разработанный исследовательским подразделением Apple, продемонстрировал впечатляющие результаты в области оптимизации вычислений для архитектур с унифицированной памятью, предложив исследователям и разработчикам гибкую модель ленивых вычислений и динамических графов. Однако жесткая привязка MLX к экосистеме Apple Silicon и API Metal оставляет значительный сегмент рынка, представленный устройствами на базе процессоров Qualcomm Snapdragon под управлением Android, без аналогичного по функциональности и производительности инструментария. Создание полноценного низкоуровневого порта MLX для платформы Snapdragon требует не только глубокого понимания архитектурных различий между проприетарными расширениями Apple и стандартными компонентами ARM, но и фундаментальной переработки механизмов управления памятью и диспетчеризации ядер.   

Архитектурная дивергенция и фундаментальные вызовы переноса
Фундаментальная сложность портирования MLX на платформу Snapdragon заключается в глубокой интеграции исходного фреймворка с аппаратными особенностями чипов серии M и A от Apple. MLX не просто использует ARM-ядра; он опирается на специфические расширения и сопроцессоры, которые отсутствуют в стандартных или модифицированных решениях Qualcomm.

Сравнение механизмов матричного ускорения: Apple AMX против Qualcomm Hexagon
Ключевым элементом производительности MLX на чипах Apple является использование расширений Advanced Matrix Extensions (AMX). AMX представляет собой специализированный ускоритель матричных операций, который работает в том же адресном пространстве, что и основные ядра центрального процессора, и управляется через недокументированные инструкции. Архитектура AMX построена вокруг идеи внешнего произведения (outer product), где одна инструкция может выполнять умножение векторов из пулов регистров X и Y с последующей аккумуляцией результата в пуле регистров Z. Это позволяет достигать производительности свыше 1 TFLOPS на одно ядро в задачах умножения матриц (SGEMM), что в пятнадцать раз превышает возможности стандартных инструкций NEON.   

В экосистеме Snapdragon аналогичные задачи возложены на Hexagon DSP и его тензорные ускорители (HMX). В отличие от AMX, который практически прозрачен для программиста в рамках стандартного потока инструкций CPU, Hexagon представляет собой отдельный вычислительный домен со своей собственной архитектурой команд (VLIW), специализированной памятью и сложным конвейером обработки. Интеграция Hexagon в ленивую модель вычислений MLX требует создания сложного слоя абстракции, который мог бы динамически формировать задачи для DSP, минимизируя задержки на переключение контекста между CPU и NPU.   

Сравнительные характеристики аппаратных блоков ускорения
Для понимания масштаба задачи портирования необходимо рассмотреть технические параметры целевых аппаратных блоков. В таблице ниже представлены ключевые характеристики, которые определяют стратегию оптимизации низкоуровневых операций.

Характеристика	Apple Silicon (M4/A18)	Qualcomm Snapdragon (8 Elite)
Матричный ускоритель	AMX (интегрирован в CPU)	Hexagon NPU (выделенный блок)
Основной векторный блок	ARM NEON / AMX	ARM NEON / SVE2
Графический ускоритель	Apple GPU (Metal)	Adreno GPU (OpenCL/Vulkan)
Пропускная способность памяти	
100 - 400 ГБ/с 

50 - 90 ГБ/с 

Тип памяти	Унифицированная (LPDDR5x)	Гетерогенная (LPDDR5x)
Поддерживаемые форматы	
FP16, BF16, INT8, FP32 

INT2-INT16, FP8, FP16 

  
Стратегия вычислительных бэкендов: OpenCL против Vulkan и проприетарных SDK
Выбор основного вычислительного бэкенда для графического процессора Adreno является критическим этапом. MLX в своей основе полагается на Metal, который обеспечивает низкоуровневый доступ к ресурсам GPU на устройствах Apple. Для Snapdragon существует три основных пути: OpenCL, Vulkan и Qualcomm Neural Processing SDK.   

Преимущества и зрелость OpenCL для Adreno
На текущий момент OpenCL является наиболее предпочтительным кандидатом для низкоуровневого порта. Qualcomm на протяжении многих лет оптимизировала драйверы OpenCL для архитектуры Adreno, обеспечивая поддержку стандарта OpenCL 3.0 и критически важных расширений, таких как cl_khr_subgroups. Именно поддержка подгрупп (subgroups) позволяет эффективно реализовывать ядра для умножения матриц и векторов (GEMV), что является узким местом при генерации токенов в больших языковых моделях.   

Опыт сообщества в проекте llama.cpp подтверждает жизнеспособность этого пути. Недавно внедренный бэкенд OpenCL для Adreno позволил достичь значительного прироста производительности, перенеся все слои модели и кэш KV на GPU. Использование OpenCL также упрощает перенос ядер из Metal, так как оба языка шейдеров основаны на C++ и имеют схожие концепции управления памятью и потоками.   

Vulkan как перспективная альтернатива
Vulkan Compute рассматривается как более современная альтернатива, обеспечивающая лучшую интеграцию с экосистемой Android и предоставляющая более явный контроль над синхронизацией и управлением памятью. Однако сложность написания и отладки Vulkan-шейдеров существенно выше, а драйверы на некоторых мобильных устройствах могут демонстрировать нестабильность. Инициативы по созданию бэкенда WebGPU для MLX могут стать мостом к поддержке Vulkan, предоставляя кроссплатформенный слой абстракции, который автоматически транслирует вычисления в нативные API целевой платформы.   

Роль Qualcomm Neural Processing SDK
Использование проприетарного SDK от Qualcomm позволяет извлечь максимальную производительность из Hexagon NPU, достигая 45-60 TOPS на последних моделях чипов. Тем не менее, этот подход имеет существенные недостатки для порта MLX. SDK ориентирован на выполнение статических графов, скомпилированных заранее в специализированные бинарные файлы (DLC). Это в корне противоречит философии MLX, которая подразумевает динамическое построение графа вычислений и ленивую материализацию массивов. Таким образом, интеграция NPU через проприетарные SDK может быть реализована только для фиксированных операций или через сложные механизмы динамической генерации кода, что значительно увеличивает объем работы.   

Управление памятью и модель унифицированного доступа на Android
Одной из самых инновационных черт MLX является модель унифицированной памяти, в которой массивы могут обрабатываться на CPU или GPU без явного копирования данных. На Apple Silicon это реализуется аппаратно через единый контроллер памяти. На платформе Snapdragon, несмотря на наличие общей физической памяти LPDDR5x, программная модель доступа более фрагментирована.   

Механизмы DMA-BUF и ION
Для реализации "нулевого копирования" (zero-copy) на Android необходимо использовать фреймворк dma-buf и подсистему ION (или современные DMA-BUF heaps). Эти механизмы позволяют выделить буфер в памяти, который будет одновременно доступен для отображения в виртуальное адресное пространство CPU и доступен для аппаратных ускорителей через их собственные MMU.   

Реализация порта потребует глубокой модификации аллокатора MLX. В текущей версии MLX для Apple Silicon используется системный аллокатор Metal. Для Snapdragon необходимо разработать новый аллокатор, который взаимодействует с Gralloc HAL Android для выделения памяти, пригодной для совместного использования. Это позволит сохранить ключевое преимущество MLX — возможность выполнять ленивые вычисления над одними и теми же данными на разных устройствах без накладных расходов на передачу по шине.   

Проблемы когерентности кэша
В отличие от Apple Silicon, где когерентность между CPU и GPU поддерживается аппаратно на высоком уровне, на многих SoC Snapdragon разработчику приходится вручную управлять операциями очистки и инвалидации кэша при переходе владения буфером между вычислительными блоками. Неправильное управление этими операциями может привести либо к повреждению данных, либо к существенным потерям производительности из-за избыточных циклов записи в основную память.   

Оптимизация низкоуровневых ядер и SIMD-инструкций
Переписывание SIMD-оптимизаций под ARM NEON является одной из наиболее трудоемких задач. Несмотря на то, что MLX уже содержит базовую поддержку NEON для работы на CPU, многие высокопроизводительные ядра в mlx-lm жестко завязаны на специфику Metal Performance Shaders (MPS) или AMX.   

Адаптация под NEON и SVE2
Современные процессоры Snapdragon, начиная с ядер Oryon в Snapdragon 8 Elite, поддерживают расширения SVE2 (Scalable Vector Extensions), которые позволяют работать с векторами переменной длины и предоставляют более мощные инструкции для матричных вычислений. Портирование должно ориентироваться не только на базовый NEON, но и на SVE2 для обеспечения конкурентоспособности с Apple Silicon.   

Реализация специализированных ядер
Для достижения целевой производительности необходимо реализовать набор ядер, оптимизированных под архитектуру Adreno:

Матричное умножение (MatMul): Реализация на базе OpenCL subgroups для минимизации обращений к глобальной памяти.   

Attention: Оптимизация механизма Flash Attention для мобильных GPU, учитывая ограничения объема локальной памяти (LDS/Shared memory).   

Квантование: Реализация быстрых ядер для деквантования "на лету" (on-the-fly) для форматов Q4_0, Q4_1 и новых MXFP8.   

В таблице ниже приведена оценка сложности реализации различных примитивов для Snapdragon.

Операция	Сложность портирования	Ключевая технология
Поэлементные (Unary/Binary)	Низкая	OpenCL Kernels / NEON
Умножение матриц (GEMM)	Высокая	OpenCL Subgroups / HMX
Слои нормализации (RMSNorm)	Средняя	Fused Kernels
Attention (SDPA)	Очень высокая	Tiled OpenCL / Flash Attention
Квантование (Q4_0)	Средняя	Bit-manipulation NEON / GPU
Текущее состояние сообщества и существующие наработки
Вопрос о портировании MLX на другие платформы активно обсуждается в сообществе. Важным прецедентом стало появление бэкенда CUDA для MLX, что доказало возможность отделения ядра фреймворка от Metal.   

Проект MLX CUDA и его уроки
Бэкенд CUDA для MLX был реализован как отдельный слой в C++ ядре фреймворка. Он позволил запускать MLX на Linux-системах с видеокартами NVIDIA, сохранив при этом привычный Python API. Этот опыт показывает, что архитектура MLX достаточно модульна, чтобы поддерживать новые типы устройств через реализацию соответствующих примитивов. Основными сложностями при создании CUDA-порта стали отсутствие некоторых квантованных операций и функций LAPACK, которые пришлось реализовывать с нуля или адаптировать из сторонних библиотек.   

Движение в сторону Android и Snapdragon X Elite
С выходом ноутбуков на базе Snapdragon X Elite интерес к запуску тяжелых ML-моделей на ARM-процессорах Qualcomm резко возрос. Хотя официальная поддержка Linux на этих чипах все еще находится в стадии стабилизации, сообщество уже добилось запуска llama.cpp с использованием OpenCL на Adreno GPU, что дает готовую базу оптимизированных ядер, которые могут быть заимствованы для порта MLX.   

Кроме того, существуют экспериментальные проекты по запуску MLX через виртуализацию. В Android 15 терминальное приложение может использовать фреймворк AVF для запуска контейнера Debian с поддержкой pKVM, что теоретически позволяет запустить текущую версию MLX (в режиме CPU-only) прямо на смартфоне. Однако для "полнофункционального низкоуровневого порта" этого недостаточно, так как требуется прямой доступ к GPU и NPU.   

Технологическая дорожная карта реализации порта
Создание порта такого уровня требует поэтапного подхода, начиная от базовой инфраструктуры и заканчивая глубокой аппаратной оптимизацией.

Этап 1: Инфраструктура и сборка под Android NDK
Первоочередной задачей является адаптация системы сборки CMake для кросс-компиляции под Android. Необходимо заменить специфичные для macOS зависимости (такие как Accelerate) на открытые альтернативы (OpenBLAS) или специализированные библиотеки Qualcomm. Также требуется реализация JNI-моста для вызова C++ ядра MLX из Android-приложений на Java или Kotlin.   

Этап 2: Реализация бэкенда OpenCL для Adreno GPU
На этом этапе создается структура mlx/core/backends/opencl. Основная работа сосредоточена на:

Создании менеджера контекста и очередей команд OpenCL.

Трансляции базовых примитивов из Metal в OpenCL C.

Реализации системы кэширования бинарных файлов ядер для ускорения запуска приложений.   

Этап 3: Оптимизация управления памятью
Замена стандартного аллокатора на систему, основанную на dma-buf. Это позволит реализовать настоящую унифицированную память, где данные, загруженные с диска (например, веса модели через mmap), могут быть напрямую использованы GPU без промежуточного копирования. Это критически важно для мобильных устройств, где пропускная способность памяти (50-90 ГБ/с) существенно ниже, чем у Apple Silicon (100-400 ГБ/с).   

Этап 4: Интеграция с Hexagon NPU через TVM или нативные ядра
Наиболее сложный этап, который может включать использование компилятора TVM для генерации кода под Hexagon. Это позволит MLX использовать тензорные блоки HMX для матричных вычислений, что даст значительный выигрыш в энергоэффективности и производительности по сравнению с GPU.   

Анализ производительности и ожидаемые результаты
При успешной реализации порта производительность на современных чипах Snapdragon (серии 8 Gen 2/3 и Elite) может быть сопоставима с чипами Apple серии M1/M2.

Пропускная способность памяти как ограничивающий фактор
В задачах инференса LLM основным ограничителем является скорость чтения весов из памяти. При пропускной способности 80 ГБ/с на Snapdragon 8 Elite, теоретический предел для модели объемом 7 млрд параметров (в 4-битном квантовании это ~3.5 ГБ) составляет около 20-22 токенов в секунду. Это сопоставимо с результатами базовых моделей M1.   

Сравнение эффективности квантования
Qualcomm продемонстрировала, что её архитектура наиболее эффективна при использовании квантования Q4_0. Внедрение "чистого" Q4_0 (pure Q4_0) в порт MLX позволит использовать специализированные аппаратные пути Adreno, что может дать преимущество в 2-3 раза по сравнению с неоптимизированными реализациями на CPU.   

Проблемы и риски реализации
Портирование проекта такого масштаба сопряжено с рядом рисков:

Фрагментация драйверов: Качество драйверов OpenCL и Vulkan сильно варьируется между производителями смартфонов (Samsung, Xiaomi, Google), что может привести к нестабильной работе порта на некоторых устройствах.   

Тепловой троттлинг: Мобильные устройства не имеют активного охлаждения, и длительные вычисления на GPU/NPU быстро приведут к снижению частот.   

Закрытость документации: Отсутствие детальной документации на низкоуровневые инструкции Hexagon HMX затрудняет создание максимально эффективных ядер без использования закрытых SDK Qualcomm.   

Заключение и стратегические рекомендации
Создание полнофункционального порта MLX для Android Snapdragon — это амбициозная инженерная задача, которая в случае успеха может радикально изменить ландшафт мобильного ИИ. Основной упор должен быть сделан на использовании OpenCL 3.0 как наиболее зрелого API для Adreno GPU и на реализации модели унифицированной памяти через dma-buf.

На текущий момент подвижки сообщества сосредоточены вокруг llama.cpp и MLC-LLM, которые уже подготовили почву в виде оптимизированных ядер. Однако портирование именно MLX принесет в экосистему Android уникальные возможности: динамическую сборку графов, ленивые вычисления и удобный интерфейс для исследований, максимально близкий к NumPy и PyTorch. Это позволит сократить разрыв в инструментарии между разработчиками под iOS и Android, создав единую среду для прототипирования и развертывания современных нейросетевых моделей на мобильных устройствах.   

Стратегически целесообразно начинать с поддержки Snapdragon X Elite на Linux, где доступ к аппаратным ресурсам более прозрачен, и затем переносить наработки в среду Android NDK. Использование опыта создания CUDA-бэкенда послужит надежным архитектурным шаблоном, гарантируя, что новый порт сохранит все функциональные преимущества оригинального фреймворка Apple.


github.com
ml-explore/mlx: MLX: An array framework for Apple silicon - GitHub
Відкриється в новому вікні

machinelearning.apple.com
Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU
Відкриється в новому вікні

youtube.com
Demystifying Apple's AMX AI accelerator: when and why it outperforms the GPU - YouTube
Відкриється в новому вікні

zhen8838.github.io
Explore AMX instructions: Unlock the performance of Apple Silicon - Zheng's Notes
Відкриється в новому вікні

qualcomm.com
Harnessing Qualcomm Adreno GPU for Generative AI: Open-Source Approach
Відкриється в новому вікні

arxiv.org
Scaling LLM Test-Time Compute with Mobile NPU on Smartphones - arXiv
Відкриється в новому вікні

arxiv.org
Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency - arXiv
Відкриється в новому вікні

v-chandra.github.io
On-Device LLMs: State of the Union, 2026 - Vikas Chandra
Відкриється в новому вікні

reddit.com
LLM is twice as fast on Snapdragon vs. Intel! SP11 v. SP10 Comparisons. - Reddit
Відкриється в новому вікні

qualcomm.com
NexaSDK for Android: a simple way to bring on-device AI to smartphones with Snapdragon
Відкриється в новому вікні

mlx-framework.org
MLX
Відкриється в новому вікні

qualcomm.com
Introducing the new OpenCL™ GPU backend in llama.cpp for Qualcomm Adreno GPUs
Відкриється в новому вікні

reddit.com
How To Cmake Llama.cpp Build For Adreno 750 GPU Snapdragon 8 Gen 3? : r/termux
Відкриється в новому вікні

app.daily.dev
Introducing the new OpenCL™ GPU Backend in llama.cpp for Qualcomm Adreno GPUs
Відкриється в новому вікні

github.com
LLama.cpp GPU Support on Android Device #16606 - GitHub
Відкриється в новому вікні

droidcon.com
Introducing the new OpenCL™ GPU Backend in llama.cpp for Qualcomm Adreno GPUs
Відкриється в новому вікні

github.com
OpenCL-Guide/chapters/programming_opencl_kernels.md at main - GitHub
Відкриється в новому вікні

learn.arm.com
Run neural graphics workloads with ML Extensions for Vulkan - Arm Learning Paths
Відкриється в новому вікні

docs.qualcomm.com
Adreno GPU on Mobile: Best Practices - Game Developer Guide
Відкриється в новому вікні

reddit.com
Can an expert chime in and explain what is holding Vulkan back from becoming the standard API for ML? : r/LocalLLaMA - Reddit
Відкриється в новому вікні

reddit.com
I'm sick and tired of emulation on Android. : r/EmulationOnAndroid - Reddit
Відкриється в новому вікні

github.com
[Feature] WebGPU backend · Issue #1790 · ml-explore/mlx - GitHub
Відкриється в новому вікні

github.com
mlx/examples/cpp/tutorial.cpp at main · ml-explore/mlx - GitHub
Відкриється в новому вікні

lpc.events
2021 LPC Allocator Attribution for DMA-BUFs in Android - Linux Plumbers Conference
Відкриється в новому вікні

docs.nvidia.com
4.1. Unified Memory — CUDA Programming Guide - NVIDIA Documentation
Відкриється в новому вікні

docs.silabs.com
Hardware Abstraction Layer (HAL) - Developer Docs - Silicon Labs
Відкриється в новому вікні

aman.ai
Aman's AI Journal • Primers • ML Runtimes
Відкриється в новому вікні

github.com
llama.cpp/docs/backend/OPENCL.md at master · ggml-org/llama.cpp · GitHub
Відкриється в новому вікні

reddit.com
MLX model NOT downloading on mobile/cellular data : r/LocalLLaMA - Reddit
Відкриється в новому вікні

github.com
MLX on CUDA · ml-explore mlx · Discussion #2422 - GitHub
Відкриється в новому вікні

github.com
NVIDIA Brev GPU Launchable to test CUDA Backend · ml-explore mlx · Discussion #2742
Відкриється в новому вікні

github.com
Support an MLX CUDA backend to improve Exo inference on NVIDIA GPUs #861 - GitHub
Відкриється в новому вікні

docs.qualcomm.com
Run LLaMA_CPP - Inference workflow for Llama3 - Windows on Snapdragon
Відкриється в новому вікні

phoronix.com
Snapdragon X Elite Laptop Performance On Linux Ends 2025 Disappointing - Phoronix
Відкриється в новому вікні

proandroiddev.com
Unleashing Linux on Android: A Developer's Playground | by Cedric Ferry | ProAndroidDev
Відкриється в новому вікні

scribd.com
Developer Documentation - MLX 0.0.4 Documentation | PDF | Input/Output - Scribd
Відкриється в новому вікні

developer.arm.com
Part 2: Sing this song in another language, translating Machine Learning Pipelines to Android - Arm Developer
Відкриється в новому вікні

callstack.com
Profiling MLC-LLM's OpenCL Backend on Android: Performance Insights - Callstack
Відкриється в новому вікні

github.com
Performance of llama.cpp on Snapdragon X Elite/Plus #8273 - GitHub
Відкриється в новому вікні

github.com
ml-explore/mlx-swift: Swift API for MLX - GitHub
Відкриється в новому вікні

github.com
ml-explore - GitHub
Відкриється в новому вікні

reddit.com
Research: vllm-mlx on Apple Silicon achieves 21% to 87% higher throughput than llama.cpp : r/LocalLLaMA - Reddit
Відкриється в новому вікні

opensource.apple.com
MLX - Apple Open Source
Відкриється в новому вікні

arxiv.org
Produc'on-Grade Local LLM Inference on Apple Silicon: A Compara've Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS - arXiv
Відкриється в новому вікні

ml-explore.github.io
MLX 0.30.5 documentation
Відкриється в новому вікні

github.com
Labels · ml-explore/mlx - GitHub
Відкриється в новому вікні

raw.githubusercontent.com
Відкриється в новому вікні

github.com
MLX Community Projects · ml-explore mlx · Discussion #654 - GitHub
Відкриється в новому вікні

github.com
stevelaskaridis/awesome-mobile-llm: Awesome Mobile LLMs - GitHub
Відкриється в новому вікні

haptic-data.com
Android Experiments/ project / haptic-data.com
Відкриється в новому вікні

arxiv.org
Puzzle: Scheduling Multiple Deep Learning Models on Mobile Device with Heterogeneous Processors - arXiv
Відкриється в новому вікні

wholetomato.com
Building High-Performance AI/ML Pipelines with C++ and CUDA - Tomato Soup
Відкриється в новому вікні

news.ycombinator.com
Apple's MLX adding CUDA support - Hacker News
Відкриється в новому вікні

kubeflow.org
MLX Guide | Kubeflow
Відкриється в новому вікні

cgmb-rocm-docs.readthedocs.io
OpenCL Programming Guide — ROCm 4.5.0 documentation
Відкриється в новому вікні

docs.qualcomm.com
Qualcomm ® Adreno™ GPU - Game Developer Guide
Відкриється в новому вікні

hgpu.org
Qualcomm Snapdragon Mobile Platform OpenCL General Programming and Optimization
Відкриється в новому вікні

github.com
hughjonesd/Rmlx: R interface to Apple's mlx library - GitHub
Відкриється в новому вікні

reddit.com
Snapdragon X Elite performance regression on Linux - Reddit
Відкриється в новому вікні

tomshardware.com
Snapdragon X1 Elite Linux laptop cancelled due to performance concerns — Linux PC maker says Qualcomm CPU is 'less suitable for Linux than expected' | Tom's Hardware
Відкриється в новому вікні

news.ycombinator.com
Discontinuation of ARM Notebook with Snapdragon X Elite SoC | Hacker News
Відкриється в новому вікні

discourse.ubuntu.com
FAQ: Ubuntu 25.04/25.10 on Snapdragon X Elite - Ubuntu Architectures - Ubuntu Community Hub
Відкриється в новому вікні

infineon.com
Getting started with Hardware Abstraction Layer (HAL) on PSOC™ Edge MCU - Infineon Technologies
Відкриється в новому вікні

ml-explore.github.io
Build and Install — MLX 0.30.5 documentation
Відкриється в новому вікні

reddit.com
MLX port of BDH (Baby Dragon Hatchling) is up : r/LocalLLaMA - Reddit
Відкриється в новому вікні

reddit.com
why does this subreddit ignore/underestimate every non snapdragon powered device? : r/EmulationOnAndroid
Відкриється в новому вікні

ml-explore.github.io
Custom Extensions in MLX — MLX 0.30.5 documentation
Відкриється в новому вікні
Оценка технического ландшафта
Я приступаю к детальному анализу архитектурных различий между экосистемами Apple Silicon и Qualcomm Snapdragon. В центре внимания — сопоставление проприетарных расширений Apple с возможностями ядер ARM NEON/SVE и специализированных вычислительных блоков Hexagon. Синтезируя текущие требования к низкоуровневой оптимизации, я планирую изучить, как механизмы унифицированной памяти MLX могут быть адаптированы к подсистемам управления памятью Android, таким как ION и DMA-BUF.
Поиск решений для графических бэкендов
Важным этапом станет исследование альтернатив Metal для работы на устройствах Snapdragon. Я анализирую возможность интеграции Vulkan Compute и OpenCL в качестве вычислительных бэкендов, а также изучаю потенциал Qualcomm Neural Processing SDK (QNN) для ускорения нейросетевых операций. Это позволит выявить критические точки при портировании ядер для матричного умножения и механизмов внимания.
Мониторинг инициатив сообщества
Для формирования полной картины я перехожу к изучению существующих форков и дискуссий в репозиториях MLX. Моя задача — найти активные попытки создания кроссплатформенных слоев абстракции оборудования (HAL) и сравнить текущий прогресс MLX с успехами таких проектов, как MLC LLM и ExecuTorch, которые уже решают схожие задачи на Android. Далее я сосредоточусь на поиске конкретных реализаций SIMD-оптимизаций под архитектуру Snapdragon.

Researching websites...

github.com
ml-explore/mlx-swift: Swift API for MLX - GitHub

github.com
ml-explore/mlx: MLX: An array framework for Apple silicon - GitHub

github.com
ml-explore - GitHub

github.com
MLX Community Projects · ml-explore mlx · Discussion #654 - GitHub

ml-explore.github.io
MLX 0.30.5 documentation

machinelearning.apple.com
Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU

learn.arm.com
Run neural graphics workloads with ML Extensions for Vulkan - Arm Learning Paths

reddit.com
Can an expert chime in and explain what is holding Vulkan back from becoming the standard API for ML? : r/LocalLLaMA - Reddit

github.com
[Feature] WebGPU backend · Issue #1790 · ml-explore/mlx - GitHub

reddit.com
Research: vllm-mlx on Apple Silicon achieves 21% to 87% higher throughput than llama.cpp : r/LocalLLaMA - Reddit

callstack.com
Profiling MLC-LLM's OpenCL Backend on Android: Performance Insights - Callstack

app.daily.dev
Introducing the new OpenCL™ GPU Backend in llama.cpp for Qualcomm Adreno GPUs

qualcomm.com
Harnessing Qualcomm Adreno GPU for Generative AI: Open-Source Approach

qualcomm.com
Introducing the new OpenCL™ GPU backend in llama.cpp for Qualcomm Adreno GPUs

droidcon.com
Introducing the new OpenCL™ GPU Backend in llama.cpp for Qualcomm Adreno GPUs

aman.ai
Aman's AI Journal • Primers • ML Runtimes

youtube.com
Demystifying Apple's AMX AI accelerator: when and why it outperforms the GPU - YouTube

arxiv.org
Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency - arXiv

zhen8838.github.io
Explore AMX instructions: Unlock the performance of Apple Silicon - Zheng's Notes

github.com
Performance of llama.cpp on Snapdragon X Elite/Plus #8273 - GitHub

v-chandra.github.io
On-Device LLMs: State of the Union, 2026 - Vikas Chandra

reddit.com
LLM is twice as fast on Snapdragon vs. Intel! SP11 v. SP10 Comparisons. - Reddit
Архитектурная гибкость и новые бэкенды
В ходе анализа текущего состояния проекта я обнаружил важные признаки расширения экосистемы за пределы Apple Silicon. В частности, появление упоминаний о поддержке CUDA для Linux и активные обсуждения по внедрению WebGPU указывают на то, что архитектура постепенно становится более модульной. Это открывает потенциальный путь для интеграции универсальных графических API, которые могут быть эффективно использованы на платформе Android.
Прогресс в оптимизации под Snapdragon
Я зафиксировал значительный прорыв в области поддержки оборудования Qualcomm: компания официально интегрировала бэкенд на базе OpenCL в проект llama.cpp, специально оптимизированный для графических процессоров Adreno. Синтезируя эти данные, я вижу возможность адаптации этих наработок для низкоуровневых операций матричного умножения в рамках порта. Также подтверждена эффективность работы подобных решений в проекте MLC LLM, что снимает часть неопределенности относительно производительности на чипах Snapdragon 8-й серии.
План углубленного исследования механизмов
Далее я намерен детально изучить исходный код на предмет жесткой привязки к Metal и оценить сложность создания уровня абстракции оборудования (HAL). В моих планах — разобраться в механизмах ленивых вычислений и унифицированной памяти, чтобы понять, насколько эффективно они транслируются на подсистемы памяти Android. Я сосредоточусь на анализе возможности портирования специфических инструкций ускорения матричных вычислений с Apple AMX на доступные альтернативы в Snapdragon.

Researching websites...

opensource.apple.com
MLX - Apple Open Source

arxiv.org
Produc'on-Grade Local LLM Inference on Apple Silicon: A Compara've Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS - arXiv

ml-explore.github.io
MLX 0.30.5 documentation

scribd.com
Developer Documentation - MLX 0.0.4 Documentation | PDF | Input/Output - Scribd

github.com
MLX on CUDA · ml-explore mlx · Discussion #2422 - GitHub

github.com
NVIDIA Brev GPU Launchable to test CUDA Backend · ml-explore mlx · Discussion #2742

github.com
Labels · ml-explore/mlx - GitHub

github.com
Support an MLX CUDA backend to improve Exo inference on NVIDIA GPUs #861 - GitHub

github.com
ml-explore/mlx: MLX: An array framework for Apple silicon - GitHub

github.com
LLama.cpp GPU Support on Android Device #16606 - GitHub

docs.qualcomm.com
Run LLaMA_CPP - Inference workflow for Llama3 - Windows on Snapdragon

github.com
llama.cpp/docs/backend/OPENCL.md at master · ggml-org/llama.cpp · GitHub

raw.githubusercontent.com

app.daily.dev
Introducing the new OpenCL™ GPU Backend in llama.cpp for Qualcomm Adreno GPUs

github.com
MLX Community Projects · ml-explore mlx · Discussion #654 - GitHub

mlx-framework.org
MLX

github.com
stevelaskaridis/awesome-mobile-llm: Awesome Mobile LLMs - GitHub

haptic-data.com
Android Experiments/ project / haptic-data.com

proandroiddev.com
Unleashing Linux on Android: A Developer's Playground | by Cedric Ferry | ProAndroidDev

docs.nvidia.com
4.1. Unified Memory — CUDA Programming Guide - NVIDIA Documentation

qualcomm.com
NexaSDK for Android: a simple way to bring on-device AI to smartphones with Snapdragon

lpc.events
2021 LPC Allocator Attribution for DMA-BUFs in Android - Linux Plumbers Conference

developer.arm.com
Part 2: Sing this song in another language, translating Machine Learning Pipelines to Android - Arm Developer

arxiv.org
Puzzle: Scheduling Multiple Deep Learning Models on Mobile Device with Heterogeneous Processors - arXiv

github.com
[Feature] WebGPU backend · Issue #1790 · ml-explore/mlx - GitHub
Модульность и универсальность бэкендов
Анализируя внутреннюю структуру фреймворка, я обнаружил, что он движется в сторону мультиплатформенности. Появление реализации для CUDA и активная разработка слоя для WebGPU показывают, что жесткая привязка к Metal постепенно ослабевает. Это критически важное наблюдение, так как оно предоставляет готовый шаблон для внедрения новых графических API, позволяя абстрагироваться от специфики оборудования Apple и подготовить почву для интеграции решений под Snapdragon.
Технологические мосты для Adreno
Я сопоставил текущие успехи в оптимизации ядер для графических процессоров Adreno, реализованные в смежных проектах, с требованиями к портированию. Особое внимание заслуживают наработки Qualcomm по использованию подгрупп в OpenCL, которые значительно повышают эффективность матричных вычислений. Синтезируя эти данные, я прихожу к выводу, что использование OpenCL может стать наиболее быстрым и производительным способом реализации низкоуровневых операций на чипах Snapdragon, минуя ограничения стандартного ARM NEON.
Изучение механизмов управления памятью
Мои дальнейшие действия будут сосредоточены на детальном изучении того, как разделение вычислений между CPU и GPU реализовано в новых бэкендах. Я планирую проанализировать структуру управления памятью, чтобы понять, как адаптировать концепцию унифицированной памяти к Android-специфичным механизмам, таким как DMA-BUF. Также я намерен исследовать технические детали реализации последних графических расширений для Adreno, чтобы определить оптимальный способ трансляции вычислительных графов.

Researching websites...

wholetomato.com
Building High-Performance AI/ML Pipelines with C++ and CUDA - Tomato Soup

news.ycombinator.com
Apple's MLX adding CUDA support - Hacker News

github.com
ml-explore/mlx: MLX: An array framework for Apple silicon - GitHub

kubeflow.org
MLX Guide | Kubeflow

github.com
mlx/examples/cpp/tutorial.cpp at main · ml-explore/mlx - GitHub

cgmb-rocm-docs.readthedocs.io
OpenCL Programming Guide — ROCm 4.5.0 documentation

github.com
OpenCL-Guide/chapters/programming_opencl_kernels.md at main - GitHub

docs.qualcomm.com
Qualcomm ® Adreno™ GPU - Game Developer Guide

docs.qualcomm.com
Adreno GPU on Mobile: Best Practices - Game Developer Guide

hgpu.org
Qualcomm Snapdragon Mobile Platform OpenCL General Programming and Optimization

github.com
MLX on CUDA · ml-explore mlx · Discussion #2422 - GitHub

github.com
hughjonesd/Rmlx: R interface to Apple's mlx library - GitHub

github.com
Support an MLX CUDA backend to improve Exo inference on NVIDIA GPUs #861 - GitHub

ml-explore.github.io
Build and Install — MLX 0.30.5 documentation

phoronix.com
Snapdragon X Elite Laptop Performance On Linux Ends 2025 Disappointing - Phoronix

reddit.com
Snapdragon X Elite performance regression on Linux - Reddit

tomshardware.com
Snapdragon X1 Elite Linux laptop cancelled due to performance concerns — Linux PC maker says Qualcomm CPU is 'less suitable for Linux than expected' | Tom's Hardware

news.ycombinator.com
Discontinuation of ARM Notebook with Snapdragon X Elite SoC | Hacker News

discourse.ubuntu.com
FAQ: Ubuntu 25.04/25.10 on Snapdragon X Elite - Ubuntu Architectures - Ubuntu Community Hub
Трансформация архитектуры в мультиплатформенную
Я проанализировал последние изменения в кодовой базе и зафиксировал важный сдвиг: появление бэкенда для CUDA официально подтверждает переход фреймворка к модульной архитектуре. Это означает, что внутренняя логика теперь отделена от специфики Apple Silicon, предоставляя четкие интерфейсы для добавления новых вычислительных устройств. Синтезируя этот опыт, я вижу возможность внедрения специализированного бэкенда для Snapdragon через абстракции потоков и примитивов, которые ранее были жестко связаны с Metal.
Оптимизация под графику Adreno
Изучая наработки Qualcomm для открытых проектов, я выделил критически важные расширения OpenCL, такие как работа с подгруппами и специфические инструкции для матричных операций (matmul). Эти инструменты позволяют обходить ограничения стандартных инструкций ARM NEON, приближая производительность Snapdragon к показателям Apple AMX. Я прихожу к выводу, что использование этих низкоуровневых расширений в сочетании с квантованием Q4_0 станет фундаментом для обеспечения высокой скорости генерации токенов на Android-устройствах.
Проектирование системного моста
Далее я сосредоточусь на детальном сопоставлении механизмов управления памятью, чтобы связать унифицированную архитектуру массивов с Android-специфичным интерфейсом DMA-BUF. В моих планах — исследовать структуру реализации CUDA-бэкенда как наиболее актуального примера для создания нового слоя абстракции. Я намерен определить оптимальный способ трансляции вычислительных графов в шейдеры, подходящие для мобильных графических процессоров, и оценить возможность использования Hexagon DSP для разгрузки центрального процессора.

Researching websites...

docs.silabs.com
Hardware Abstraction Layer (HAL) - Developer Docs - Silicon Labs

infineon.com
Getting started with Hardware Abstraction Layer (HAL) on PSOC™ Edge MCU - Infineon Technologies

github.com
ml-explore/mlx: MLX: An array framework for Apple silicon - GitHub

scribd.com
Developer Documentation - MLX 0.0.4 Documentation | PDF | Input/Output - Scribd

ml-explore.github.io
Build and Install — MLX 0.30.5 documentation

qualcomm.com
Harnessing Qualcomm Adreno GPU for Generative AI: Open-Source Approach

arxiv.org
Scaling LLM Test-Time Compute with Mobile NPU on Smartphones - arXiv

qualcomm.com
Introducing the new OpenCL™ GPU backend in llama.cpp for Qualcomm Adreno GPUs

reddit.com
How To Cmake Llama.cpp Build For Adreno 750 GPU Snapdragon 8 Gen 3? : r/termux

github.com
Performance of llama.cpp on Snapdragon X Elite/Plus #8273 - GitHub

reddit.com
MLX port of BDH (Baby Dragon Hatchling) is up : r/LocalLLaMA - Reddit

reddit.com
why does this subreddit ignore/underestimate every non snapdragon powered device? : r/EmulationOnAndroid

reddit.com
MLX model NOT downloading on mobile/cellular data : r/LocalLLaMA - Reddit

reddit.com
I'm sick and tired of emulation on Android. : r/EmulationOnAndroid - Reddit
Унификация вычислительных бэкендов
Синтезируя данные о последних обновлениях, я зафиксировал официальное появление бэкенда CUDA и активные работы над WebGPU. Это фундаментальное изменение подтверждает, что архитектура фреймворка полностью переработана для поддержки сторонних графических процессоров. Теперь я вижу четкую структуру, в которой вычислительные примитивы отделены от аппаратной логики Apple, что открывает прямой путь для реализации специфических ядер под Adreno через стандартные интерфейсы расширений.
Адаптация низкоуровневых инструкций
Я проанализировал роль проприетарных расширений Apple AMX и сопоставил их с возможностями Snapdragon. Выявлено, что для достижения сопоставимой производительности на Android необходимо использовать расширения OpenCL для работы с подгруппами и кооперативными векторами, которые уже успешно внедрены Qualcomm в другие открытые проекты. Это позволяет транслировать сложные операции матричного умножения и механизмы внимания напрямую в аппаратные возможности графических ядер Adreno, минимизируя задержки при обработке больших языковых моделей.
Финальная интеграция системных уровней
В завершение я планирую детально проработать механизм управления памятью, чтобы обеспечить бесшовную работу унифицированных массивов с подсистемой DMA-BUF в Android. Мои действия будут сосредоточены на изучении структуры реализации расширений для создания новых типов ядер, что позволит интегрировать оптимизированные инструкции ARM NEON и специфические вызовы для мобильных GPU. Я намерен подготовить технический обзор того, как модульная система позволит перенести высокоуровневые API на новую мобильную платформу без потери эффективности.

Researching websites...

ml-explore.github.io
Custom Extensions in MLX — MLX 0.30.5 documentation
Модульность и выход за пределы Apple Silicon
Я завершил анализ архитектурных изменений и подтверждаю, что переход фреймворка на мульти-бэкендную систему — это свершившийся факт. Появление официальной поддержки CUDA для Linux стало решающим доказательством того, что ядро больше не имеет жесткой привязки к Metal. Синтезируя структуру новых интерфейсов, я вижу, что теперь существует четкий путь для интеграции Snapdragon: создание нового вычислительного слоя на базе OpenCL или Vulkan может быть выполнено через стандартизированную систему примитивов, которую разработчики подготовили для сторонних производителей оборудования.
Стратегия оптимизации для Snapdragon
В ходе исследования я сопоставил возможности Apple AMX с современными решениями для чипов Qualcomm. Мой анализ показывает, что ключевым ресурсом для порта станут недавно опубликованные наработки Qualcomm по OpenCL-бэкенду для графических процессоров Adreno. Использование расширений для работы с подгруппами и кооперативными векторами позволяет эффективно заменить специфические инструкции Apple, обеспечивая высокую скорость матричных вычислений. Это устраняет главный информационный разрыв: мы теперь точно знаем, какие именно низкоуровневые механизмы обеспечат производительность, сопоставимую с оригиналом.
Интеграция с системными ресурсами Android
Я структурирую финальный план действий, который свяжет высокоуровневые возможности фреймворка с системной спецификой Android. Основное внимание будет уделено адаптации модели унифицированной памяти через механизм DMA-BUF для исключения лишнего копирования данных между CPU и GPU. Я приступаю к формированию итоговых выводов о том, как наиболее эффективно перенести существующие API, используя модульную архитектуру для реализации специализированных ядер под архитектуру Snapdragon и интеграции с мобильными графическими ускорителями.
