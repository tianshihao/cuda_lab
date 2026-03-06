#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "cuda_check.hpp"
#include "memory_type.hpp"

namespace cuda_lab {

template <typename T, MemoryType Type>
struct MemoryInitializer;

template <typename T>
struct MemoryInitializer<T, MemoryType::kHost> {
  static void fill(T* ptr, std::size_t n, T const& value) {
    std::uninitialized_fill_n(ptr, n, value);
  }
};

template <typename T>
struct MemoryInitializer<T, MemoryType::kPinned> {
  static void fill(T* ptr, std::size_t n, T const& value) {
    std::uninitialized_fill_n(ptr, n, value);
  }
};

template <typename T>
struct MemoryInitializer<T, MemoryType::kMappedPinned> {
  static void fill(T* ptr, std::size_t n, T const& value) {
    std::uninitialized_fill_n(ptr, n, value);
  }
};

namespace operators {
// Declaration of device fill kernel launcher (defined in device_fill.cu)
template <typename T>
void device_fill_kernel_launcher(T* ptr, std::size_t n, T value);
}  // namespace operators

template <typename T>
struct MemoryInitializer<T, MemoryType::kDevice> {
  static void fill(T* ptr, std::size_t n, T const& value) {
    if constexpr (std::is_integral_v<T> || std::is_floating_point_v<T>) {
      if (value == T{}) {
        // Use cudaMemset for zero-initialization
        CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(T)));
        return;
      }
    }
    // General value: launch kernel (implementation in device_fill.cu)
    operators::device_fill_kernel_launcher(ptr, n, value);
  }
};

}  // namespace cuda_lab
