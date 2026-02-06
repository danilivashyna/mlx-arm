# MLX-ARM Shader Compilation
# Compiles GLSL compute shaders to SPIR-V for Vulkan

cmake_minimum_required(VERSION 3.28)

# Find glslangValidator (from Vulkan SDK)
find_program(GLSLANG_VALIDATOR
    NAMES glslangValidator
    HINTS
        "$ENV{VULKAN_SDK}/bin"
        "$ENV{VULKAN_SDK}/macOS/bin"
        "${ANDROID_NDK}/sources/third_party/vulkan/src/build-android/external/glslang/StandAlone"
    REQUIRED
)

if(NOT GLSLANG_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found. Install Vulkan SDK or Android NDK.")
endif()

message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR}")

# Output directory for compiled SPIR-V
set(SPIRV_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})

# List of compute shaders
set(COMPUTE_SHADERS
    vector_add.comp
    matmul_naive.comp
    matmul_tiled.comp
    matmul_fp16.comp
    matmul_vectorized.comp
    matmul_subgroup.comp
    matmul_q4_0.comp
    rms_norm.comp
    silu.comp
    elemwise_multiply.comp
    rope.comp
)

# Compile each shader
foreach(SHADER ${COMPUTE_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}")
    set(SHADER_SPIRV "${SPIRV_OUTPUT_DIR}/${SHADER_NAME}.spv")
    
    add_custom_command(
        OUTPUT ${SHADER_SPIRV}
        COMMAND ${GLSLANG_VALIDATOR}
            -V                           # Vulkan mode
            -o ${SHADER_SPIRV}          # Output file
            ${SHADER_SOURCE}            # Input file
            --target-env vulkan1.3      # Target Vulkan 1.3
            -g                          # Debug info
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling shader: ${SHADER} -> ${SHADER_NAME}.spv"
        VERBATIM
    )
    
    list(APPEND SPIRV_BINARIES ${SHADER_SPIRV})
endforeach()

# Custom target for all shaders
add_custom_target(mlx_shaders ALL
    DEPENDS ${SPIRV_BINARIES}
    COMMENT "Building all MLX compute shaders"
)

# Install shaders to binary directory
install(
    FILES ${SPIRV_BINARIES}
    DESTINATION shaders
)

# For Android APK, copy to assets
if(ANDROID)
    foreach(SPIRV ${SPIRV_BINARIES})
        get_filename_component(SPIRV_NAME ${SPIRV} NAME)
        # This will be used by Android build system
        add_custom_command(
            TARGET mlx_shaders POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${SPIRV}
                "${CMAKE_BINARY_DIR}/android-assets/${SPIRV_NAME}"
            COMMENT "Copying ${SPIRV_NAME} to Android assets"
        )
    endforeach()
endif()
