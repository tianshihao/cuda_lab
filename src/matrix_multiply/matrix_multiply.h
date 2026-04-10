#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>

#include "matrix.hpp"

namespace cuda_lab::matrix_multiply {
enum class MatrixMultiplyType { kSimple = 0, kCoalesced = 1, kSharedAB = 2 };

constexpr std::size_t kBlockSize{32};

template <typename T>
void MatrixMultiplyWrapper(cuda_lab::MatrixHost<T> const& a,
                           cuda_lab::MatrixHost<T> const& b,
                           cuda_lab::MatrixHost<T>& c, MatrixMultiplyType type);

template <typename T>
inline void MatrixMultiply(
    cuda_lab::MatrixHost<T> const& a, cuda_lab::MatrixHost<T> const& b,
    cuda_lab::MatrixHost<T>& c,
    MatrixMultiplyType type = MatrixMultiplyType::kSimple) {
  assert(a.cols() == b.rows() && c.rows() == a.rows() && c.cols() == b.cols() &&
         "Inner dimensions must match for multiplication");

  MatrixMultiplyWrapper(a, b, c, type);
}
/// @param a Numerator - the total quantity to be divided (e.g., total
/// elements, bytes)
/// @param b Denominator - the capacity per unit (e.g., elements per block,
/// bytes per page)
/// @return The minimal number of units required to contain the total quantity
/// @example
///   DivUp(7, 3) = 3;  // 7/3=2.333 → ceil to 3
///   DivUp(8, 4) = 2;  // 8/4=2 → exact division
__host__ __device__ inline std::size_t DivUp(std::size_t const a,
                                             std::size_t const b) {
  return (a + b - 1) / b;
}
}  // namespace cuda_lab::matrix_multiply
