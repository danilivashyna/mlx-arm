set(MLX_CORE_SOURCES
    device.cpp
    allocator.cpp
    array.cpp
    ops.cpp
    graph.cpp
    utils.cpp
)

set(MLX_CORE_HEADERS
    device.h
    allocator.h
    array.h
    ops.h
    graph.h
    backend.h
    utils.h
)

add_library(mlx-core-impl STATIC ${MLX_CORE_SOURCES})

target_include_directories(mlx-core-impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_compile_features(mlx-core-impl PUBLIC cxx_std_17)

# Platform-specific linking
if(ANDROID)
    target_link_libraries(mlx-core-impl PUBLIC
        log
        android
    )
endif()

install(FILES ${MLX_CORE_HEADERS}
    DESTINATION include/mlx/core
)
