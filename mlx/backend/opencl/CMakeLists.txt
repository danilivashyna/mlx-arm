# Copyright © 2026 MLX-ARM Contributors
# SPDX-License-Identifier: MIT

add_library(mlx-backend-opencl STATIC
    cl_context.cpp
)

target_include_directories(mlx-backend-opencl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
)

# OpenCL will be loaded dynamically via dlopen, so no need to link directly
# but we need dl library for dlopen/dlsym
target_link_libraries(mlx-backend-opencl PRIVATE dl)
