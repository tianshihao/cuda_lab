#include "matrix_multiply.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>

#include "matrix.h"

namespace cuda_lab::matrix_multiply {

void CheckCudaError(cudaError_t const err, char const* msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void PrintMatrix(Matrix const& m, std::string const& n) {
  std::cout << "Matrix " << n << " (partial):" << std::endl;
  for (std::size_t i{0}; i < kBlockSize; ++i) {
    for (std::size_t j{0}; j < kBlockSize; ++j) {
  // for (std::size_t i{0}; i < m.height; ++i) {
  //   for (std::size_t j{0}; j < m.width; ++j) {
      std::cout << m.elements[i * m.width + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Shape of Matrix " << n << ": " << m.height << " x " << m.width
            << std::endl
            << std::endl;
}

}  // namespace cuda_lab::matrix_multiply