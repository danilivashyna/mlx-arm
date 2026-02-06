set(MLX_VULKAN_SOURCES
    vulkan_backend.cpp
    vulkan_context.cpp
    vulkan_device.cpp
    vulkan_buffer.cpp
    vulkan_pipeline.cpp
    vulkan_command.cpp
)

set(MLX_VULKAN_HEADERS
    vulkan_backend.h
    vulkan_context.h
    vulkan_device.h
    vulkan_buffer.h
    vulkan_pipeline.h
    vulkan_command.h
)

add_library(mlx-backend-vulkan STATIC ${MLX_VULKAN_SOURCES})

target_include_directories(mlx-backend-vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Link Vulkan
if(ANDROID)
    target_link_libraries(mlx-backend-vulkan PUBLIC
        ${Vulkan_LIBRARIES}
        log
    )
else()
    target_link_libraries(mlx-backend-vulkan PUBLIC
        Vulkan::Vulkan
    )
endif()

target_compile_definitions(mlx-backend-vulkan PUBLIC
    MLX_HAS_VULKAN
)

# Shader compilation
# Find glslangValidator
find_program(GLSLANG_VALIDATOR glslangValidator)
if(NOT GLSLANG_VALIDATOR)
    message(WARNING "glslangValidator not found. Shaders will not be compiled.")
else()
    message(STATUS "Found glslangValidator: ${GLSLANG_VALIDATOR}")
    
    # Compile shaders
    set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(SPIRV_DIR ${CMAKE_CURRENT_BINARY_DIR}/spirv)
    file(MAKE_DIRECTORY ${SPIRV_DIR})
    
    file(GLOB SHADER_SOURCES "${SHADER_DIR}/*.comp")
    
    foreach(SHADER_SOURCE ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
        set(SPIRV_OUTPUT ${SPIRV_DIR}/${SHADER_NAME}.spv)
        
        add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_SOURCE} -o ${SPIRV_OUTPUT}
            DEPENDS ${SHADER_SOURCE}
            COMMENT "Compiling shader ${SHADER_NAME}"
        )
        
        list(APPEND SPIRV_BINARIES ${SPIRV_OUTPUT})
    endforeach()
    
    # Add custom target for all shaders
    add_custom_target(mlx-vulkan-shaders ALL DEPENDS ${SPIRV_BINARIES})
    add_dependencies(mlx-backend-vulkan mlx-vulkan-shaders)
    
    # Install SPIR-V binaries
    install(FILES ${SPIRV_BINARIES}
        DESTINATION share/mlx/shaders
    )
endif()

install(FILES ${MLX_VULKAN_HEADERS}
    DESTINATION include/mlx/backend/vulkan
)
