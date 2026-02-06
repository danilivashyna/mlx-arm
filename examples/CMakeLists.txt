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

# Install examples
install(TARGETS vector_add_example matmul_benchmark
    RUNTIME DESTINATION bin/examples
)
