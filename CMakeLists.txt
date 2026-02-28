# Minimum CMake version required
cmake_minimum_required(VERSION 3.18)

if(UNIX)
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc" CACHE FILEPATH "CUDA compiler")

  # set(CUDA_HOME $ENV{CUDA_HOME})
  # set(CUDA_ROOT $ENV{CUDA_ROOT})
endif()

if(WIN32)
  set(CUDAToolkit_ROOT $ENV{CUDAToolkit_ROOT})
  message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
endif()

project(cuda_lab VERSION 1.0 LANGUAGES CXX CUDA)

# Make ./include available to all subprojects
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_custom_target(clean-all
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_clean.cmake
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

add_definitions(-DREPO_ROOT_PATH=\"${CMAKE_SOURCE_DIR}\")

add_subdirectory(src)

# enable_testing()
# add_subdirectory(test)

# set(CMAKE_CXX_CLANG_TIDY "${CLANGTIDY};-extra-arg=-Wno-unknown-warning-option;-header-filter=${CMAKE_SOURCE_DIR}/src/*")
