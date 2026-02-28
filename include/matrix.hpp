#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

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
      : rows_{rows}, cols_{cols}, mem_{rows * cols, value} {}

  T* data() { return mem_.host_ptr(); }
  T const* data() const { return mem_.host_ptr(); }
  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  T get(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("MatrixHost::get: index out of range");
    }
    return mem_.host_ptr()[row * cols_ + col];
  }

  void set(std::size_t row, std::size_t col, T value) {
    if (row >= rows_ || col >= cols_) {
      throw std::out_of_range("MatrixHost::set: index out of range");
    }
    mem_.host_ptr()[row * cols_ + col] = value;
  }

  // Copy host matrix to device
  MatrixDevice<T> to_device() const {
    T* dev_ptr{nullptr};
    auto size{rows_ * cols_ * sizeof(T)};
    auto err{cudaMalloc(reinterpret_cast<void**>(&dev_ptr), size)};
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed in MatrixHost::to_device");
    }
    err = cudaMemcpy(dev_ptr, mem_.host_ptr(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(dev_ptr);
      throw std::runtime_error("cudaMemcpy failed in MatrixHost::to_device");
    }
    return MatrixDevice<T>{dev_ptr, rows_, cols_};
  }

  // Copy device matrix back to host
  void from_device(MatrixDevice<T> const& dev_mat) {
    auto size{rows_ * cols_ * sizeof(T)};
    auto err{cudaMemcpy(mem_.host_ptr(), dev_mat.data, size,
                        cudaMemcpyDeviceToHost)};
    if (err != cudaSuccess) {
      throw std::runtime_error("cudaMemcpy failed in MatrixHost::from_device");
    }
  }

  void print(std::size_t max_count = 10, int width = 6,
             std::ostream& os = std::cout) const {
    auto r{rows()};
    auto c{cols()};
    std::size_t hc{max_count / 2};  // half count

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
  Memory<T, MemoryType::kHost> mem_;
};

}  // namespace cuda_lab
