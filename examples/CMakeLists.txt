add_executable(vector_add_example
    vector_add.cpp
)

target_link_libraries(vector_add_example PRIVATE
    mlx-core
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(vector_add_example PRIVATE mlx-backend-vulkan)
    target_compile_definitions(vector_add_example PRIVATE MLX_BUILD_VULKAN)
endif()

# Matrix multiplication benchmark
add_executable(matmul_benchmark
    matmul_benchmark.cpp
)

target_link_libraries(matmul_benchmark PRIVATE
    mlx-core
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(matmul_benchmark PRIVATE mlx-backend-vulkan)
    target_compile_definitions(matmul_benchmark PRIVATE MLX_BUILD_VULKAN)
endif()

# RMSNorm benchmark
add_executable(rmsnorm_benchmark
    rmsnorm_benchmark.cpp
)

target_link_libraries(rmsnorm_benchmark PRIVATE
    mlx-core
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(rmsnorm_benchmark PRIVATE mlx-backend-vulkan)
    target_compile_definitions(rmsnorm_benchmark PRIVATE MLX_BUILD_VULKAN)
endif()

# SiLU activation benchmark
add_executable(silu_benchmark
    silu_benchmark.cpp
)

target_link_libraries(silu_benchmark PRIVATE
    mlx-core
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(silu_benchmark PRIVATE mlx-backend-vulkan)
    target_compile_definitions(silu_benchmark PRIVATE MLX_BUILD_VULKAN)
endif()

# Install examples
install(TARGETS vector_add_example matmul_benchmark rmsnorm_benchmark silu_benchmark
    RUNTIME DESTINATION bin/examples
)
