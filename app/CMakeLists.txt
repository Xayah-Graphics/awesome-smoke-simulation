cmake_minimum_required(VERSION 4.1.2)

find_package(CUDAToolkit REQUIRED)

find_program(
        SLANGC_EXECUTABLE
        NAMES slangc
        HINTS ENV VULKAN_SDK
        PATH_SUFFIXES Bin bin
)

if (NOT SLANGC_EXECUTABLE)
    message(FATAL_ERROR "slangc was not found. Install the Vulkan SDK and make sure slangc is available.")
endif ()

set(SMOKE_VIS_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${SMOKE_VIS_SHADER_DIR}")

set(SMOKE_SIM_SHADER_SOURCES
        shaders/smoke_volume.slang
)

set(SMOKE_SIM_SHADER_OUTPUTS "")
foreach (shader IN LISTS SMOKE_SIM_SHADER_SOURCES)
    get_filename_component(shader_name "${shader}" NAME_WE)
    set(shader_output "${SMOKE_VIS_SHADER_DIR}/${shader_name}.spv")
    add_custom_command(
            OUTPUT "${shader_output}"
            COMMAND "${SLANGC_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${shader}" -profile sm_6_5 -target spirv -entry vs_main -entry fs_main -o "${shader_output}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${shader}"
            COMMENT "Compiling ${shader_name}.slang"
            VERBATIM
    )
    list(APPEND SMOKE_SIM_SHADER_OUTPUTS "${shader_output}")
endforeach ()

add_custom_target(smoke-visualizer-shaders DEPENDS ${SMOKE_SIM_SHADER_OUTPUTS})

add_executable(
        smoke-visualizer
        app.cpp
        main.cpp
)

target_sources(smoke-visualizer
        PUBLIC FILE_SET cxx_modules TYPE CXX_MODULES FILES
        app.ixx
)

add_dependencies(smoke-visualizer smoke-visualizer-shaders)

target_compile_features(smoke-visualizer PRIVATE cxx_std_23)
target_include_directories(smoke-visualizer PRIVATE ${CMAKE_SOURCE_DIR}/001-stable-fluids)
target_compile_definitions(smoke-visualizer PRIVATE SMOKE_SIM_SHADER_DIR="${SMOKE_VIS_SHADER_DIR}")
target_link_libraries(smoke-visualizer PRIVATE vk-core::vk-core stable-fluids CUDA::cudart)

set_property(TARGET smoke-visualizer PROPERTY CXX_MODULE_STD ON)

set_target_properties(
        smoke-visualizer
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
