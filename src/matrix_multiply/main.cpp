#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "matrix.h"
#include "matrix_multiply.h"

constexpr cuda_lab::matrix_multiply::MatrixMultiplyType kMatrixMultiplyType{
    cuda_lab::matrix_multiply::MatrixMultiplyType::kSharedAB};

void Test1() {
  cuda_lab::matrix_multiply::Matrix a;
  a.height = 2;
  a.width = 3;
  a.stride = 3;
  a.elements = new float[a.height * a.stride];
  std::vector<float> h_data_a{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::copy(h_data_a.begin(), h_data_a.end(), a.elements);

  cuda_lab::matrix_multiply::Matrix b;
  b.height = 3;
  b.width = 4;
  b.stride = 4;
  b.elements = new float[b.height * b.stride];
  std::vector<float> h_data_b{1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                              7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::copy(h_data_b.begin(), h_data_b.end(), b.elements);

  cuda_lab::matrix_multiply::PrintMatrix(a, "A");
  cuda_lab::matrix_multiply::PrintMatrix(b, "B");

  cuda_lab::matrix_multiply::Matrix c;
  c.height = a.height;
  c.width = b.width;
  c.stride = b.width;
  c.elements = new float[c.height * c.stride];
  std::fill_n(c.elements, c.height * c.stride, 0.0f);

  cuda_lab::matrix_multiply::MatrixMultiply(a, b, c, kMatrixMultiplyType);

  cuda_lab::matrix_multiply::PrintMatrix(c, "C");
}

void Test2() {
  cuda_lab::matrix_multiply::Matrix a;
  // a.height = 1600;
  a.height = 16;
  a.width = cuda_lab::matrix_multiply::kBlockSize;
  a.stride = cuda_lab::matrix_multiply::kBlockSize;
  a.elements = new float[a.height * a.stride];
  std::vector<float> h_data_a(a.height * a.stride, 1.0f);
  std::copy(h_data_a.begin(), h_data_a.end(), a.elements);

  cuda_lab::matrix_multiply::Matrix b;
  b.height = cuda_lab::matrix_multiply::kBlockSize;
  b.width = 192;
  b.stride = 192;
  // b.width = 192000;
  // b.stride = 192000;
  b.elements = new float[b.height * b.stride];
  std::vector<float> h_data_b(b.height * b.stride, 2.0f);
  std::copy(h_data_b.begin(), h_data_b.end(), b.elements);

  cuda_lab::matrix_multiply::PrintMatrix(a, "A");
  cuda_lab::matrix_multiply::PrintMatrix(b, "B");

  cuda_lab::matrix_multiply::Matrix c;
  c.height = a.height;
  c.width = b.width;
  c.stride = b.width;
  c.elements = new float[c.height * c.stride];
  std::fill_n(c.elements, c.height * c.stride, 0.0f);

  cuda_lab::matrix_multiply::MatrixMultiply(a, b, c, kMatrixMultiplyType);

  cuda_lab::matrix_multiply::PrintMatrix(c, "C");
}

int main() {
  Test1();
  // Test2();

  return 0;
}