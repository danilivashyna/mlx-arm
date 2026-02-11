cmake_minimum_required(VERSION 3.22)
project(mlx-jni)

# MLX-ARM is built separately, just link to it
# add_subdirectory(../../ mlx-arm-build)

# JNI library
add_library(mlx-jni SHARED
    jni/mlx_jni.cpp
)

target_link_libraries(mlx-jni PRIVATE
    mlx-core
    mlx-backend-vulkan
    log
    android
)

target_include_directories(mlx-jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
