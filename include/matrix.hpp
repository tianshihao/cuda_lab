#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "buffer.hpp"
#include "memory.hpp"

namespace cuda_lab {

// Device-side POD matrix struct (template)
template <typename T>
struct MatrixDevice {
  T* data;
  std::size_t rows;
  std::size_t cols;

  __host__ __device__ T get(std::size_t row, std::size_t col) const {
    return data[row * cols + col];
  }
  __host__ __device__ void set(std::size_t row, std::size_t col, T value) {
    data[row * cols + col] = value;
  }
};

// Host-side matrix class (template)

template <typename T>

class MatrixHost {
 public:
  MatrixHost(std::size_t rows, std::size_t cols, T value = T{})
      : rows_{rows},
        cols_{cols},
        buffer_{rows * cols, value},
        device_{nullptr} {}

  T* data() { return buffer_.host_ptr(); }
  T const* data() const { return buffer_.host_ptr(); }
  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  T get(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("MatrixHost::get: index out of range");
    }
    return data()[row * cols_ + col];
  }

  void set(std::size_t row, std::size_t col, T value) {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("MatrixHost::set: index out of range");
    }
    data()[row * cols_ + col] = value;
    buffer_.mark_dirty_host();
  }

  // Get or create managed device matrix
  MatrixDevice<T>* device() const {
    if (!device_) {
      to_device();
    }
    return device_;
  }

  // Explicitly sync host to device and update device_ pointer
  void to_device() const {
    buffer_.sync_to_device();
    if (!device_) {
      device_ = new MatrixDevice<T>{buffer_.device_ptr(), rows_, cols_};
    } else {
      device_->data = buffer_.device_ptr();
      device_->rows = rows_;
      device_->cols = cols_;
    }
  }

  // Explicitly sync device to host
  void from_device() const { buffer_.sync_to_host(); }

  void print(std::size_t max_count = 10, int width = 6,
             std::ostream& os = std::cout) const {
    auto r{rows()};
    auto c{cols()};
    std::size_t hc{max_count / 2};

    auto is_row_skipped = [&](std::size_t i) {
      return (r > max_count && i >= hc && i < r - hc);
    };
    auto is_col_skipped = [&](std::size_t j) {
      return (c > max_count && j >= hc && j < c - hc);
    };

    for (std::size_t i = 0; i < r; ++i) {
      if (is_row_skipped(i)) {
        if (i == hc) {
          os << std::setw(width) << "..." << std::endl;
        }
        continue;
      }
      for (std::size_t j = 0; j < c; ++j) {
        if (is_col_skipped(j)) {
          if (j == hc) os << std::setw(width) << "...";
          continue;
        }
        os << std::setw(width) << get(i, j);
      }
      os << "\n";
    }
  }

 private:
  std::size_t rows_;
  std::size_t cols_;
  Buffer<T, MemoryType::kHost> buffer_;
  mutable MatrixDevice<T>* device_ = nullptr;
};

}  // namespace cuda_lab
