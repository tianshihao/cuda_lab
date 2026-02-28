#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>  // for std::byte
#include <iostream>
#include <stdexcept>

namespace cuda_lab {

enum class MemoryType { kHost = 0, kPinned, kDevice, kMappedPinned };

template <typename T, MemoryType Type>
class Memory {
 public:
  explicit Memory(std::size_t n, T value = T{})
      : n_{n}, raw_{nullptr}, bytes_{n * sizeof(T)} {
    if constexpr (Type == MemoryType::kPinned) {
      void* tmp{nullptr};
      if (cudaHostAlloc(&tmp, bytes_, cudaHostAllocDefault) != cudaSuccess) {
        throw std::runtime_error("cudaHostAlloc failed");
      }
      raw_ = static_cast<std::byte*>(tmp);
    } else if constexpr (Type == MemoryType::kDevice) {
      void* tmp{nullptr};
      if (cudaMalloc(&tmp, bytes_) != cudaSuccess) {
        throw std::runtime_error("cudaMalloc (device) failed");
      }
      raw_ = static_cast<std::byte*>(tmp);
    } else if constexpr (Type == MemoryType::kMappedPinned) {
      void* tmp{nullptr};
      if (cudaHostAlloc(&tmp, bytes_, cudaHostAllocMapped) != cudaSuccess) {
        throw std::runtime_error("cudaHostAlloc (mapped) failed");
      }
      raw_ = static_cast<std::byte*>(tmp);
    } else {
      raw_ = static_cast<std::byte*>(operator new[](bytes_));
    }

    fill(value);
  }

  ~Memory() {
    if constexpr (Type == MemoryType::kPinned ||
                  Type == MemoryType::kMappedPinned) {
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
      if constexpr (Type == MemoryType::kPinned ||
                    Type == MemoryType::kMappedPinned) {
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

  T* host_ptr() {
    if constexpr (Type == MemoryType::kHost || Type == MemoryType::kPinned ||
                  Type == MemoryType::kMappedPinned) {
      return reinterpret_cast<T*>(raw_);
    } else {
      return nullptr;
    }
  }
  T const* host_ptr() const {
    if constexpr (Type == MemoryType::kHost || Type == MemoryType::kPinned ||
                  Type == MemoryType::kMappedPinned) {
      return reinterpret_cast<const T*>(raw_);
    } else {
      return nullptr;
    }
  }
  T* data() {
    if constexpr (Type == MemoryType::kDevice ||
                  Type == MemoryType::kMappedPinned ||
                  Type == MemoryType::kPinned) {
      return device_ptr();
    } else {
      return host_ptr();
    }
  }
  T const* data() const {
    if constexpr (Type == MemoryType::kDevice ||
                  Type == MemoryType::kMappedPinned ||
                  Type == MemoryType::kPinned) {
      return device_ptr();
    } else {
      return host_ptr();
    }
  }
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
  T* device_ptr() {
    if constexpr (Type == MemoryType::kMappedPinned) {
      T* dev_ptr{nullptr};
      if (cudaHostGetDevicePointer(&dev_ptr, raw_, 0) != cudaSuccess) {
        throw std::runtime_error("cudaHostGetDevicePointer failed");
      }
      return dev_ptr;
    } else if constexpr (Type == MemoryType::kDevice) {
      return data();
    } else if constexpr (Type == MemoryType::kPinned) {
      return data();
    } else if constexpr (Type == MemoryType::kHost) {
      return data();
    } else {
      static_assert(
          Type == MemoryType::kMappedPinned || Type == MemoryType::kDevice ||
              Type == MemoryType::kPinned || Type == MemoryType::kHost,
          "device_ptr() is only valid for mapped pinned, device, pinned, or "
          "host memory");
      return nullptr;
    }
  }
  const T* device_ptr() const {
    if constexpr (Type == MemoryType::kMappedPinned) {
      T* dev_ptr{nullptr};
      if (cudaHostGetDevicePointer(&dev_ptr, raw_, 0) != cudaSuccess) {
        throw std::runtime_error("cudaHostGetDevicePointer failed");
      }
      return dev_ptr;
    } else if constexpr (Type == MemoryType::kDevice) {
      return data();
    } else if constexpr (Type == MemoryType::kPinned) {
      return data();
    } else if constexpr (Type == MemoryType::kHost) {
      return data();
    } else {
      static_assert(
          Type == MemoryType::kMappedPinned || Type == MemoryType::kDevice ||
              Type == MemoryType::kPinned || Type == MemoryType::kHost,
          "device_ptr() is only valid for mapped pinned, device, pinned, or "
          "host memory");
      return nullptr;
    }
  }

 private:
  std::byte* raw_{nullptr};
  std::size_t n_{0};
  std::size_t bytes_{0};
};

}  // namespace cuda_lab
