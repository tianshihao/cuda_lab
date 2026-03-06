#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

#include "cuda_check.hpp"
#include "memory_type.hpp"

namespace cuda_lab {

template <typename T, MemoryType Type>
struct MemoryAllocator;

// Host memory allocator
template <typename T>
struct MemoryAllocator<T, MemoryType::kHost> {
  static T* allocate(std::size_t n) {
    return static_cast<T*>(::operator new[](n * sizeof(T)));
  }
  static void deallocate(T* ptr) { ::operator delete[](ptr); }
};

// Pinned memory allocator
template <typename T>
struct MemoryAllocator<T, MemoryType::kPinned> {
  static T* allocate(std::size_t n) {
    void* tmp{nullptr};
    CUDA_CHECK(cudaHostAlloc(&tmp, n * sizeof(T), cudaHostAllocDefault));
    return static_cast<T*>(tmp);
  }
  static void deallocate(T* ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }
};

// Device memory allocator
template <typename T>
struct MemoryAllocator<T, MemoryType::kDevice> {
  static T* allocate(std::size_t n) {
    void* tmp{nullptr};
    CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(T)));
    return static_cast<T*>(tmp);
  }
  static void deallocate(T* ptr) { CUDA_CHECK(cudaFree(ptr)); }
};

// Mapped pinned memory allocator
template <typename T>
struct MemoryAllocator<T, MemoryType::kMappedPinned> {
  static T* allocate(std::size_t n) {
    void* tmp{nullptr};
    CUDA_CHECK(cudaHostAlloc(&tmp, n * sizeof(T), cudaHostAllocMapped));
    return static_cast<T*>(tmp);
  }
  static void deallocate(T* ptr) { CUDA_CHECK(cudaFreeHost(ptr)); }
};

}  // namespace cuda_lab
