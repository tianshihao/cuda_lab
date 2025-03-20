#include "matrix_multiply.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

namespace cuda_lab::matrix_multiply {

void CheckCudaError(cudaError_t const err, char const* msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void PrintMatrix(const float* const matrix, std::size_t const rows,
                 std::size_t const cols, std::size_t const tile_size,
                 const char* const name) {
  std::cout << "Matrix " << name << " (partial):" << std::endl;
  for (std::size_t i{0}; i < tile_size && i < rows; ++i) {
    for (std::size_t j{0}; j < tile_size && j < cols; ++j) {
      std::cout << matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Shape of Matrix " << name << ": " << rows << " x " << cols
            << std::endl
            << std::endl;
}

}  // namespace cuda_lab::matrix_multiply