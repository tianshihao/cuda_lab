project(matrix_multiply LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
endif()

file(GLOB SOURCES
  "*.cpp"
  "*.cu"
)

file(GLOB HEADERS
  "*.h"
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)