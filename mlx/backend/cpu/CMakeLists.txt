set(MLX_CPU_SOURCES
    cpu_backend.cpp
)

add_library(mlx-backend-cpu STATIC ${MLX_CPU_SOURCES})

target_include_directories(mlx-backend-cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

target_compile_definitions(mlx-backend-cpu PUBLIC
    MLX_HAS_CPU
)
