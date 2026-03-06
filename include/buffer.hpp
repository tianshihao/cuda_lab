#pragma once

#include <memory>
#include <optional>
#include <type_traits>

#include "memory.hpp"
#include "memory_type.hpp"

namespace cuda_lab {

template <typename T, MemoryType InitialType>
class Buffer {
 public:
  explicit Buffer(std::size_t n, T const& initial_value = T{})
      : n_{n}, host_dirty_{false}, device_dirty_{false} {
    set_memory<InitialType>(n_, initial_value);
    current_type_ = InitialType;
    if constexpr (InitialType == MemoryType::kHost ||
                  InitialType == MemoryType::kPinned ||
                  InitialType == MemoryType::kMappedPinned) {
      mark_host_dirty();
      mark_device_clean();
    } else if constexpr (InitialType == MemoryType::kDevice) {
      mark_host_clean();
      mark_device_dirty();
    }
  }

  void set_host(size_t idx, const T& value) { host_ptr()[idx] = value; }
  T get_host(size_t idx) const { return host_ptr()[idx]; }
  // In real code, launch a kernel to set or get value on device
  // void set_device(size_t idx, const T& value);
  // T get_device(size_t idx) const;

  T* host_ptr() {
    if constexpr (InitialType == MemoryType::kPinned ||
                  InitialType == MemoryType::kMappedPinned) {
      mark_host_dirty();
      return get_pinned_or_mapped_ptr();
    } else {
      sync_to_host();
      mark_host_dirty();
      return get_memory<MemoryType::kHost>()->data();
    }
  }
  T const* host_ptr() const { return const_cast<Buffer*>(this)->host_ptr(); }
  T* device_ptr() {
    if constexpr (InitialType == MemoryType::kPinned ||
                  InitialType == MemoryType::kMappedPinned) {
      mark_device_dirty();
      return get_pinned_or_mapped_ptr();
    } else {
      sync_to_device();
      mark_device_dirty();
      return device_mem_->data();
    }
  }
  T const* device_ptr() const {
    return const_cast<Buffer*>(this)->device_ptr();
  }

  void sync_to_host() {
    if (is_device_dirty()) {
      sync<MemoryType::kHost>();
      mark_device_clean();
    }
    mark_host_clean();
  }
  void sync_to_device() {
    if (is_host_dirty()) {
      sync<MemoryType::kDevice>();
      mark_host_clean();
    }
    mark_device_clean();
  }

  void mark_host_clean() { host_dirty_ = false; }
  void mark_device_clean() { device_dirty_ = false; }
  void mark_host_dirty() { host_dirty_ = true; }
  void mark_device_dirty() { device_dirty_ = true; }
  bool is_host_dirty() const { return host_dirty_; }
  bool is_device_dirty() const { return device_dirty_; }
  bool is_host_clean() const { return !host_dirty_; }
  bool is_device_clean() const { return !device_dirty_; }

  std::size_t size() const { return n_; }

  template <MemoryType Type>
  Memory<T, Type>* get_memory();
  template <MemoryType Type>
  void set_memory(std::size_t n, T const& initial_value = T{});
  template <MemoryType Type>
  void sync();
  T* get_pinned_or_mapped_ptr();

 private:
  std::size_t n_{};
  MemoryType current_type_{InitialType};
  bool host_dirty_{false};
  bool device_dirty_{false};
  std::unique_ptr<Memory<T, MemoryType::kHost>> host_mem_{nullptr};
  std::unique_ptr<Memory<T, MemoryType::kPinned>> pinned_mem_{nullptr};
  std::unique_ptr<Memory<T, MemoryType::kDevice>> device_mem_{nullptr};
  std::unique_ptr<Memory<T, MemoryType::kMappedPinned>> mapped_mem_{nullptr};
};

// Implementation of Buffer member templates (move outside class, fix shadowing)
template <typename T, MemoryType InitialType>
template <MemoryType Type>
Memory<T, Type>* Buffer<T, InitialType>::get_memory() {
  if constexpr (Type == MemoryType::kHost) {
    return host_mem_.get();
  } else if constexpr (Type == MemoryType::kPinned) {
    return pinned_mem_.get();
  } else if constexpr (Type == MemoryType::kDevice) {
    return device_mem_.get();
  } else if constexpr (Type == MemoryType::kMappedPinned) {
    return mapped_mem_.get();
  } else {
    return nullptr;
  }
}

template <typename T, MemoryType InitialType>
template <MemoryType Type>
void Buffer<T, InitialType>::set_memory(std::size_t n, T const& initial_value) {
  if constexpr (Type == MemoryType::kHost) {
    host_mem_ =
        std::make_unique<Memory<T, MemoryType::kHost>>(n, initial_value);
  } else if constexpr (Type == MemoryType::kPinned) {
    pinned_mem_ =
        std::make_unique<Memory<T, MemoryType::kPinned>>(n, initial_value);
  } else if constexpr (Type == MemoryType::kDevice) {
    device_mem_ =
        std::make_unique<Memory<T, MemoryType::kDevice>>(n, initial_value);
  } else if constexpr (Type == MemoryType::kMappedPinned) {
    mapped_mem_ = std::make_unique<Memory<T, MemoryType::kMappedPinned>>(
        n, initial_value);
  }
}

template <typename T, MemoryType InitialType>
template <MemoryType Type>
void Buffer<T, InitialType>::sync() {
  if constexpr (Type == MemoryType::kHost) {
    // The first sync of kDevice Buffer.
    if (!host_mem_) {
      set_memory<MemoryType::kHost>(n_);
    }
    if (device_mem_ && is_device_dirty()) {
      device_mem_->copy_to(*host_mem_);
    } else if (pinned_mem_ && is_device_dirty()) {
      pinned_mem_->copy_to(*host_mem_);
    } else if (mapped_mem_ && is_device_dirty()) {
      mapped_mem_->copy_to(*host_mem_);
    }
  } else if constexpr (Type == MemoryType::kDevice) {
    // The first sync of kHost Buffer.
    if (!device_mem_) {
      set_memory<MemoryType::kDevice>(n_);
    }
    if (host_mem_ && is_host_dirty()) {
      host_mem_->copy_to(*device_mem_);
    } else if (pinned_mem_ && is_host_dirty()) {
      pinned_mem_->copy_to(*device_mem_);
    } else if (mapped_mem_ && is_host_dirty()) {
      mapped_mem_->copy_to(*device_mem_);
    }
  }
}

template <typename T, MemoryType InitialType>
T* Buffer<T, InitialType>::get_pinned_or_mapped_ptr() {
  if constexpr (InitialType == MemoryType::kPinned) {
    if (!pinned_mem_) {
      set_memory<MemoryType::kPinned>(n_);
    }
    return pinned_mem_->data();
  } else if constexpr (InitialType == MemoryType::kMappedPinned) {
    if (!mapped_mem_) {
      set_memory<MemoryType::kMappedPinned>(n_);
    }
    return mapped_mem_->data();
  } else {
    return nullptr;
  }
}

}  // namespace cuda_lab
