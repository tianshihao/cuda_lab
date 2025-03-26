#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <string>

#include "matrix.h"

namespace cuda_lab::matrix_multiply {

constexpr std::size_t kBlockSize{16};
constexpr std::size_t kTileSize{16};

enum class MatrixMultiplyType : std::size_t {
  kSimple = 0,
  kCoalesced = 1,
  kSharedAB = 2
};

void CheckCudaError(cudaError_t const err, char const* msg);
void MatrixMultiply(Matrix const& a, Matrix const& b, Matrix& c,
                    MatrixMultiplyType const type);
void PrintMatrix(Matrix const& m, std::string const& n);
/// @brief Compute ceiling division(rounds up when there's a remainder)
/// @param a Numerator - the total quantity to be divided (e.g., total elements,
/// bytes)
/// @param b Denominator - the capacity per unit (e.g., elements per block,
/// bytes per page)
/// @return The minimal number of units required to contain the total quantity
/// @example
///   DivUp(7, 3) = 3;  // 7/3=2.333 → ceil to 3
///   DivUp(8, 4) = 2;  // 8/4=2 → exact division
inline std::size_t DivUp(std::size_t const a, std::size_t const b) {
  return (a + b - 1) / b;
}
// More, matrix multiply using cpu serial and cpu parallel.
}  // namespace cuda_lab::matrix_multiply