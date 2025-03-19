#include "matrix_multiply.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace cuda_lab::matrix_multiply {

void CheckCudaError(cudaError_t const err, char const* msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void PrintMatrix(const float* matrix, int rows, int cols, int tile_size,
                 const char* name) {
  std::cout << "Matrix " << name << " (partial):" << std::endl;
  for (int i = 0; i < tile_size && i < rows; ++i) {
    for (int j = 0; j < tile_size && j < cols; ++j) {
      std::cout << matrix[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Shape of Matrix " << name << ": " << rows << " x " << cols
            << std::endl
            << std::endl;
}

}  // namespace cuda_lab::matrix_multiply