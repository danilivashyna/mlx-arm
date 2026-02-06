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

# Install example
install(TARGETS vector_add_example
    RUNTIME DESTINATION bin/examples
)
