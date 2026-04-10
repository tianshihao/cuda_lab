#include <iostream>

#include "cuda_scoped_timer.hpp"
#include "matrix.hpp"
#include "matrix_multiply.h"
#include "size_literals.hpp"

void PrintResult(float elapsed, std::size_t bytes,
                 cuda_lab::MatrixHost<int>& mat_c) {
  using namespace cuda_lab;
  std::cout << "Elapsed time: " << elapsed << " ms"
            << ", Effective bandwidth: " << bytes / (elapsed / 1_k) / 1_GB
            << " GB/s" << std::endl;
  std::cout << "Result matrix C (int):" << std::endl;
  mat_c.print();
}

int main() {
  using namespace cuda_lab;
  using namespace cuda_lab::matrix_multiply;

  std::size_t const remainder_outer{0};
  std::size_t const remainder_inner{1};

  std::size_t const rows_a{kBlockSize + remainder_outer};
  std::size_t const cols_a{kBlockSize + remainder_inner};
  std::size_t const rows_b{kBlockSize + remainder_inner};
  std::size_t const cols_b{kBlockSize + remainder_outer};

  // Int test
  MatrixHost<int> mat_a_int{rows_a, cols_a, 2};
  MatrixHost<int> mat_b_int{rows_b, cols_b, 3};
  MatrixHost<int> mat_c_int{rows_a, cols_b, 0};

  auto bytes{mat_a_int.bytes() + mat_b_int.bytes() + mat_c_int.bytes()};

  {
    CudaScopedTimer timer{"MatrixMultiply (int) Simple"};
    MatrixMultiply<int>(mat_a_int, mat_b_int, mat_c_int,
                        MatrixMultiplyType::kSimple);
    PrintResult(timer.finish(), bytes, mat_c_int);
  }

  mat_c_int.fill(0);

  {
    CudaScopedTimer timer{"MatrixMultiply (int) Coalesced"};
    MatrixMultiply<int>(mat_a_int, mat_b_int, mat_c_int,
                        MatrixMultiplyType::kCoalesced);
    PrintResult(timer.finish(), bytes, mat_c_int);
  }

  mat_c_int.fill(0);

  {
    CudaScopedTimer timer{"MatrixMultiply (int) SharedAB"};
    MatrixMultiply<int>(mat_a_int, mat_b_int, mat_c_int,
                        MatrixMultiplyType::kSharedAB);
    PrintResult(timer.finish(), bytes, mat_c_int);
  }

  return 0;
}
