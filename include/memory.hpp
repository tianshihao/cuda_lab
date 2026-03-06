#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "allocator.hpp"
#include "cuda_check.hpp"
#include "initializer.hpp"
#include "memory_type.hpp"

namespace cuda_lab {

template <typename T, MemoryType Type>
class Memory {
 public:
  explicit Memory(std::size_t const& n, T const& initial_value = T{})
      : n_{n},
        raw_{reinterpret_cast<std::byte*>(
            MemoryAllocator<T, Type>::allocate(n))},
        bytes_{n * sizeof(T)} {
    fill(initial_value);
  }

  ~Memory() {
    MemoryAllocator<T, Type>::deallocate(reinterpret_cast<T*>(raw_));
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
      MemoryAllocator<T, Type>::deallocate(reinterpret_cast<T*>(raw_));
      raw_ = other.raw_;
      n_ = other.n_;
      bytes_ = other.bytes_;
      other.raw_ = nullptr;
      other.n_ = 0;
      other.bytes_ = 0;
    }
    return *this;
  }

  void fill(T const& value) {
    MemoryInitializer<T, Type>::fill(data(), n_, value);
  }

  T* data() { return reinterpret_cast<T*>(raw_); }
  T const* data() const { return reinterpret_cast<const T*>(raw_); }
  std::size_t size() const { return n_; }
  std::size_t bytes() const { return bytes_; }

  template <MemoryType OtherType>
  void copy_to(Memory<T, OtherType>& dst) const {
    if constexpr (Type == MemoryType::kHost &&
                  OtherType == MemoryType::kDevice) {
      CUDA_CHECK(cudaMemcpy(dst.data(), this->data(), this->size() * sizeof(T),
                            cudaMemcpyHostToDevice));
    } else if constexpr (Type == MemoryType::kDevice &&
                         OtherType == MemoryType::kHost) {
      CUDA_CHECK(cudaMemcpy(dst.data(), this->data(), this->size() * sizeof(T),
                            cudaMemcpyDeviceToHost));
    } else {
      // Add more cases as needed
    }
  }

 private:
  std::byte* raw_{nullptr};
  std::size_t n_{0};
  std::size_t bytes_{0};
};

}  // namespace cuda_lab
