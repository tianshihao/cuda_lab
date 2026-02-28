#include <iostream>

#include "cuda_scoped_timer.hpp"
#include "matrix.hpp"
#include "matrix_multiply.h"

int main() {
  using namespace cuda_lab;
  using namespace cuda_lab::matrix_multiply;

  std::size_t const rows_a{200};
  std::size_t const cols_a{kTileSize};
  std::size_t const rows_b{kTileSize};
  std::size_t const cols_b{200};

  // Int test
  MatrixHost<int> mat_a_int{rows_a, cols_a, 2};
  MatrixHost<int> mat_b_int{rows_b, cols_b, 3};
  MatrixHost<int> mat_c_int{rows_a, cols_b, 0};

  {
    CudaScopedTimer timer{"MatrixMultiply (int)"};
    MatrixMultiply<int>(mat_a_int, mat_b_int, mat_c_int,
                        MatrixMultiplyType::kSimple);
  }

  std::cout << "Result matrix C (int):" << std::endl;
  mat_c_int.print();

  return 0;
}
