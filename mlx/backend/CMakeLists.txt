# CPU backend (always built)
add_subdirectory(cpu)

# Vulkan backend
if(MLX_BUILD_VULKAN)
    add_subdirectory(vulkan)
endif()

# OpenCL backend
if(MLX_BUILD_OPENCL)
    add_subdirectory(opencl)
endif()

# Backend aggregation library
add_library(mlx-backend INTERFACE)
target_link_libraries(mlx-backend INTERFACE
    mlx-backend-cpu
)

if(MLX_BUILD_VULKAN)
    target_link_libraries(mlx-backend INTERFACE
        mlx-backend-vulkan
    )
endif()

if(MLX_BUILD_OPENCL)
    target_link_libraries(mlx-backend INTERFACE
        mlx-backend-opencl
    )
endif()
