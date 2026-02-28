#pragma once

#include <cassert>
#include <cstddef>
#include <stdexcept>

#include "matrix.hpp"

namespace cuda_lab::matrix_multiply {
enum class MatrixMultiplyType { kSimple = 0, kCoalesced = 1, kSharedAB = 2 };

constexpr std::size_t kBlockSize{32};
constexpr std::size_t kTileSize{32};

template <typename T>
void SimpleMatrixMultiplyKernel(cuda_lab::MatrixDevice<T> const& a,
                                cuda_lab::MatrixDevice<T> const& b,
                                cuda_lab::MatrixDevice<T>& c);

template <typename T>
inline void MatrixMultiply(
    cuda_lab::MatrixHost<T> const& a, cuda_lab::MatrixHost<T> const& b,
    cuda_lab::MatrixHost<T>& c,
    MatrixMultiplyType type = MatrixMultiplyType::kSimple) {
  assert(a.cols() == b.rows() && c.rows() == a.rows() && c.cols() == b.cols() &&
         "Inner dimensions must match for multiplication");

  cuda_lab::MatrixDevice<T> dev_a{a.to_device()};
  cuda_lab::MatrixDevice<T> dev_b{b.to_device()};
  cuda_lab::MatrixDevice<T> dev_c{c.to_device()};

  switch (type) {
    case MatrixMultiplyType::kSimple:
      SimpleMatrixMultiplyKernel(dev_a, dev_b, dev_c);
      break;
    case MatrixMultiplyType::kCoalesced:
      // CoalescedMatrixMultiplyKernel<T>(dev_a, dev_b, dev_c);
      break;
    case MatrixMultiplyType::kSharedAB:
      // SharedABMatrixMultiplyKernel<T>(dev_a, dev_b, dev_c);
      break;
    default:
      throw std::invalid_argument("Invalid MatrixMultiplyType");
  }

  c.from_device(dev_c);

  cudaFree(dev_a.data);
  cudaFree(dev_b.data);
  cudaFree(dev_c.data);
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
