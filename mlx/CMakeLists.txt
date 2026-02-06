# Core library
add_subdirectory(core)

# Backends
add_subdirectory(backend)

# Main library target
add_library(mlx-core INTERFACE)
target_link_libraries(mlx-core INTERFACE
    mlx-core-impl
    mlx-backend
)
