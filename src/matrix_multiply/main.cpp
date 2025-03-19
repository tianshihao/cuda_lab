#include <iostream>
#include <vector>

#include "matrix_multiply.h"

int main() {
  constexpr int A_rows{160};  // Number of rows in matrix A
  constexpr int A_cols{
      cuda_lab::matrix_multiply::TILE_DIM};  // Number of columns in matrix A
  constexpr int B_cols{192};                 // Number of columns in matrix B
  std::vector<float> h_a(A_rows * A_cols,
                         1.0f);  // Initialize matrix A with 1.0
  std::vector<float> h_b(A_cols * B_cols,
                         2.0f);  // Initialize matrix B with 2.0
  std::vector<float> h_c(A_rows * B_cols,
                         0.0f);  // Initialize matrix C with 0.0

  cuda_lab::matrix_multiply::PrintMatrix(
      h_a.data(), A_rows, A_cols, cuda_lab::matrix_multiply::TILE_DIM, "A");
  cuda_lab::matrix_multiply::PrintMatrix(
      h_b.data(), A_cols, B_cols, cuda_lab::matrix_multiply::TILE_DIM, "B");

  cuda_lab::matrix_multiply::MatrixMultiply(h_a.data(), h_b.data(), h_c.data(),
                                            A_rows, A_cols, B_cols);

  cuda_lab::matrix_multiply::PrintMatrix(
      h_c.data(), A_rows, B_cols, cuda_lab::matrix_multiply::TILE_DIM, "C");

  return 0;
}