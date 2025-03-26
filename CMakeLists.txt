# CMakeLists.txt (项目根目录)

# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

# Generate the compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project name and C++ standard
project(cuda_image_lab LANGUAGES C CXX CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

if(WIN32)
  set(CUDAToolkit_ROOT $ENV{CUDAToolkit_ROOT})
  message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
endif()

# Find packages
if(NOT WIN32)
  find_package(OpenCV REQUIRED)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# Custom clean target
add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_clean.cmake
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
)

add_subdirectory(src)

set(CMAKE_CXX_CLANG_TIDY "${CLANGTIDY};-extra-arg=-Wno-unknown-warning-option;-header-filter=${CMAKE_SOURCE_DIR}/src/*")
