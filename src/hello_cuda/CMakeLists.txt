# Project name and C++ standard
project(hello_cuda LANGUAGES CXX CUDA)

# Set C++ standard for CUDA files
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify the CUDA architecture for RTX 3080
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Define source files and headers
file(GLOB SOURCES
  "*.cpp"
  "*.cu"
)

file(GLOB HEADERS
  "*.h"
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Enable profiling with -pg
target_compile_options(${PROJECT_NAME} PRIVATE -pg)
target_link_options(${PROJECT_NAME} PRIVATE -pg)
