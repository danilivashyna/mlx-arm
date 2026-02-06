cmake_minimum_required(VERSION 3.20)

add_executable(vector_add_example
    vector_add_example.cpp
)

target_link_libraries(vector_add_example
    mlx-core
    mlx-backend-vulkan
)

# For Android, also link log
if(ANDROID)
    target_link_libraries(vector_add_example
        log
        android
    )
endif()

# Install example
install(TARGETS vector_add_example
    RUNTIME DESTINATION examples
)
