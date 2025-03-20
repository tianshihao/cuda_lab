#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cuda_lab::matrix_multiply {

// Give me an enum class to indicate the type of matrix multiplication
// that is being performed. The options are: Simple, Tiled, and Shared Memory.
// The default is Simple.
enum class MatrixMultiplyType : std::size_t {
  kSimple = 0,
  kCoalesced = 1,
  kSharedAB = 2
};

constexpr std::size_t TILE_DIM{32};

void CheckCudaError(cudaError_t const err, char const* msg);
void MatrixMultiply(float const* const h_a, float const* const h_b,
                    float* const h_c, std::size_t const a_rows,
                    std::size_t const a_cols, std::size_t const b_cols,
                    MatrixMultiplyType const type);
void PrintMatrix(const float* const matrix, std::size_t const rows,
                 std::size_t const cols, std::size_t const tile_size,
                 const char* const name);
inline std::size_t DivUp(std::size_t const a, std::size_t const b) {
  return (a + b - 1) / b;
}
// More, matrix multiply using cpu serial and cpu parallel.
}  // namespace cuda_lab::matrix_multiply