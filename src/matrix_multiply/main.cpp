#include <cstddef>
#include <iostream>
#include <vector>

#include "matrix_multiply.h"

int main() {
//   constexpr std::size_t a_rows{160};  // Number of rows in matrix A
//   constexpr std::size_t a_cols{
//       cuda_lab::matrix_multiply::TILE_DIM};  // Number of columns in matrix A
//   constexpr std::size_t b_cols{192};         // Number of columns in matrix B
//   std::vector<float> h_a(a_rows * a_cols,
//                          1.0f);  // Initialize matrix A with 1.0
//   std::vector<float> h_b(a_cols * b_cols,
//                          2.0f);  // Initialize matrix B with 2.0
//   std::vector<float> h_c(a_rows * b_cols,
//                          0.0f);  // Initialize matrix C with 0.0

  constexpr std::size_t a_rows{2};  // Number of rows in matrix A
  constexpr std::size_t a_cols{3};  // Number of columns in matrix A
  constexpr std::size_t b_cols{4};  // Number of columns in matrix B
  std::vector<float> h_a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> h_b{1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                         7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  std::vector<float> h_c(a_rows * b_cols,
                         0.0f);  // Initialize matrix C with 0.0

  cuda_lab::matrix_multiply::PrintMatrix(
      h_a.data(), a_rows, a_cols, cuda_lab::matrix_multiply::TILE_DIM, "A");
  cuda_lab::matrix_multiply::PrintMatrix(
      h_b.data(), a_cols, b_cols, cuda_lab::matrix_multiply::TILE_DIM, "B");

  cuda_lab::matrix_multiply::MatrixMultiply(
      h_a.data(), h_b.data(), h_c.data(), a_rows, a_cols, b_cols,
      cuda_lab::matrix_multiply::MatrixMultiplyType::kSharedAB);

  cuda_lab::matrix_multiply::PrintMatrix(
      h_c.data(), a_rows, b_cols, cuda_lab::matrix_multiply::TILE_DIM, "C");

  return 0;
}