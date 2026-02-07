#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>  // for std::byte
#include <iostream>
#include <stdexcept>

namespace cuda_lab::async_overlap {

enum class MemoryType { kHost = 0, kPinned, kDevice };

template <typename T, MemoryType Type>
class Memory {
 public:
  explicit Memory(std::size_t n, T value = T{})
      : n_{n}, raw_{nullptr}, bytes_{n * sizeof(T)} {
    if constexpr (Type == MemoryType::kPinned) {
      void* tmp{nullptr};
      if (cudaMallocHost(&tmp, bytes_) != cudaSuccess) {
        throw std::runtime_error("cudaMallocHost failed");
      }
      raw_ = static_cast<std::byte*>(tmp);
    } else if constexpr (Type == MemoryType::kDevice) {
      void* tmp{nullptr};
      if (cudaMalloc(&tmp, bytes_) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc (device) failed");
      }
      raw_ = static_cast<std::byte*>(tmp);
    } else {
      raw_ = static_cast<std::byte*>(operator new[](bytes_));
    }

    fill(value);
  }

  ~Memory() {
    if constexpr (Type == MemoryType::kPinned) {
      if (raw_) {
        cudaFreeHost(raw_);
      }
    } else if constexpr (Type == MemoryType::kDevice) {
      if (raw_) {
        cudaFree(raw_);
      }
    } else {
      operator delete[](raw_);
    }
  }

  Memory(const Memory&) = delete;
  Memory& operator=(const Memory&) = delete;

  Memory(Memory&& other) noexcept
      : raw_{other.raw_}, n_{other.n_}, bytes_{other.bytes_} {
    other.raw_ = nullptr;
    other.n_ = 0;
    other.bytes_ = 0;
  }
  Memory& operator=(Memory&& other) noexcept {
    if (this != &other) {
      if constexpr (Type == MemoryType::kPinned) {
        if (raw_) {
          cudaFreeHost(raw_);
        }
      } else {
        operator delete[](raw_);
      }
      raw_ = other.raw_;
      n_ = other.n_;
      bytes_ = other.bytes_;
      other.raw_ = nullptr;
      other.n_ = 0;
      other.bytes_ = 0;
    }
    return *this;
  }

  T* data() { return reinterpret_cast<T*>(raw_); }
  T const* data() const { return reinterpret_cast<const T*>(raw_); }
  std::size_t size() const { return n_; }
  std::size_t bytes() const { return bytes_; }
  void fill(T value) {
    if constexpr (Type == MemoryType::kDevice) {
      if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, char> ||
                    std::is_same_v<T, std::byte>) {
        cudaMemset(raw_, static_cast<int>(value), bytes_);
      } else {
        auto* tmp{new T[n_]};
        std::fill(tmp, tmp + n_, value);
        cudaMemcpy(raw_, tmp, bytes_, cudaMemcpyHostToDevice);
        delete[] tmp;
      }
    } else {
      std::fill(data(), data() + n_, value);
    }
  }
  T* slice(std::size_t offset) { return data() + offset; }

 private:
  std::byte* raw_{nullptr};
  std::size_t n_{0};
  std::size_t bytes_{0};
};

}  // namespace cuda_lab::async_overlap
