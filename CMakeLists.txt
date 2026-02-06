cmake_minimum_required(VERSION 3.20)
project(mlx-arm VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(MLX_BUILD_VULKAN "Build Vulkan backend" ON)
option(MLX_BUILD_OPENCL "Build OpenCL fallback backend" OFF)
option(MLX_BUILD_PYTHON "Build Python bindings" ON)
option(MLX_BUILD_TESTS "Build unit tests" ON)
option(MLX_ENABLE_NEON "Enable NEON optimizations" ON)
option(MLX_ENABLE_SVE "Enable SVE/SVE2 if available" ON)
option(MLX_ANDROID "Building for Android" OFF)

# Platform detection
if(ANDROID)
    set(MLX_ANDROID ON)
    set(MLX_BUILD_PYTHON OFF)  # No Python on Android
    message(STATUS "Building for Android (API ${ANDROID_PLATFORM})")
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find required packages
if(MLX_BUILD_VULKAN)
    if(ANDROID)
        # Android provides Vulkan through NDK
        find_library(VULKAN_LIB vulkan REQUIRED)
        set(Vulkan_LIBRARIES ${VULKAN_LIB})
        message(STATUS "Found Android Vulkan: ${VULKAN_LIB}")
    else()
        find_package(Vulkan REQUIRED)
        message(STATUS "Found Vulkan: ${Vulkan_VERSION}")
    endif()
endif()

if(MLX_BUILD_OPENCL)
    find_package(OpenCL REQUIRED)
    message(STATUS "Found OpenCL: ${OpenCL_VERSION}")
endif()

# CPU features detection
include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)

# Check for NEON support (should be present on all ARMv8-A+)
if(MLX_ENABLE_NEON)
    check_cxx_compiler_flag("-march=armv8-a+simd" COMPILER_SUPPORTS_NEON)
    if(COMPILER_SUPPORTS_NEON)
        add_compile_options(-march=armv8-a+simd)
        add_compile_definitions(MLX_HAS_NEON)
        message(STATUS "NEON support: Enabled")
    endif()
endif()

# Check for SVE/SVE2 support
if(MLX_ENABLE_SVE)
    check_cxx_compiler_flag("-march=armv8-a+sve2" COMPILER_SUPPORTS_SVE2)
    if(COMPILER_SUPPORTS_SVE2)
        # SVE2 will be enabled per-file for specific kernels
        add_compile_definitions(MLX_HAS_SVE2)
        message(STATUS "SVE2 support: Available (will be used selectively)")
    endif()
endif()

# Compiler warnings
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# Core library
add_subdirectory(mlx)

# Tests
if(MLX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Android-specific
if(ANDROID)
    add_subdirectory(android)
endif()

# Python bindings
if(MLX_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Examples
add_subdirectory(examples)

# Installation
install(TARGETS mlx-core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY mlx/core/
    DESTINATION include/mlx/core
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "")
message(STATUS "MLX-ARM Configuration Summary")
message(STATUS "==============================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "")
message(STATUS "Features:")
message(STATUS "  Vulkan backend: ${MLX_BUILD_VULKAN}")
message(STATUS "  OpenCL backend: ${MLX_BUILD_OPENCL}")
message(STATUS "  Python bindings: ${MLX_BUILD_PYTHON}")
message(STATUS "  NEON optimizations: ${MLX_ENABLE_NEON}")
message(STATUS "  SVE2 support: ${MLX_ENABLE_SVE}")
message(STATUS "  Android build: ${MLX_ANDROID}")
message(STATUS "")
