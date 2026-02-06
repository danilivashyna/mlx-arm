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

# Element-wise multiply benchmark
add_executable(elemwise_multiply_benchmark
    elemwise_multiply_benchmark.cpp
)

target_link_libraries(elemwise_multiply_benchmark PRIVATE
    mlx-core
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(elemwise_multiply_benchmark PRIVATE mlx-backend-vulkan)
    target_compile_definitions(elemwise_multiply_benchmark PRIVATE MLX_BUILD_VULKAN)
endif()

# RoPE benchmark
add_executable(rope_benchmark
    rope_benchmark.cpp
)

target_link_libraries(rope_benchmark PRIVATE
    mlx-core
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(rope_benchmark PRIVATE mlx-backend-vulkan)
    target_compile_definitions(rope_benchmark PRIVATE MLX_BUILD_VULKAN)
endif()

# Softmax benchmark
add_executable(softmax_benchmark
    softmax_benchmark.cpp
)

target_link_libraries(softmax_benchmark PRIVATE
    mlx-core
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(softmax_benchmark PRIVATE mlx-backend-vulkan)
    target_compile_definitions(softmax_benchmark PRIVATE MLX_BUILD_VULKAN)
endif()

# Attention pipeline test
add_executable(attention_test
    attention_test.cpp
)

target_link_libraries(attention_test PRIVATE
    mlx-core
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(attention_test PRIVATE mlx-backend-vulkan)
    target_compile_definitions(attention_test PRIVATE MLX_BUILD_VULKAN)
endif()

# Install examples
install(TARGETS vector_add_example matmul_benchmark rmsnorm_benchmark silu_benchmark elemwise_multiply_benchmark rope_benchmark softmax_benchmark attention_test
    RUNTIME DESTINATION bin/examples
)
