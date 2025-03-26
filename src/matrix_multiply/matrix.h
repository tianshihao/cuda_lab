#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_lab::matrix_multiply {

struct Matrix {
  std::size_t width;
  std::size_t height;
  std::size_t stride;
  float* elements;
};

}  // namespace cuda_lab::matrix_multiply