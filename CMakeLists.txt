cmake_minimum_required(VERSION 4.1.2)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(awesome-smoke-simulation VERSION 1.0.0 LANGUAGES CXX CUDA)


# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)


# ============================================================================
# Vulkan App
# ============================================================================
add_subdirectory(app/vulkan-visualizer)
add_subdirectory(001-stable-fluids)
add_subdirectory(002-visual-simulation-of-smoke)
add_subdirectory(app)
