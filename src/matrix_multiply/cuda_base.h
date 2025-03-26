#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

namespace cuda_lab::matrix_multiply {

enum class MemoryType : uint8_t {
  kHostMemory = 0,
  kPinnedMemory = 1,
  kDeviceMemory = 2
};

template <MemoryType memory_type>
struct CudaMemDeleter;

template <>
struct CudaMemDeleter<MemoryType::kDeviceMemory> {
  template <typename T>
  void operator()(T* ptr) const noexcept {
    if (ptr) {
      cudaFree(ptr);
      std::cout << "Free device memory" << std::endl;
    }
  }
};

template <>
struct CudaMemDeleter<MemoryType::kHostMemory> {
  template <typename T>
  void operator()(T* ptr) const noexcept {
    if (ptr) {
      free(ptr);
      std::cout << "Free host memory" << std::endl;
      // cudaFreeHost(ptr);
    }
  }
};

// template <>
// struct CudaMemDeleter<MemoryType::kRegisteredHostMemory> {
//   template <typename T>
//   void operator()(T* ptr) const noexcept {
//     if (ptr) {
//       cudaHostUnregister(ptr);
//       std::cout << "Unregister host memory" << std::endl;
//       free(ptr);
//     }
//   }
// };

}  // namespace cuda_lab::matrix_multiply