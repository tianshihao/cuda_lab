#include <cuda_runtime.h>

#include "matrix.hpp"
#include "matrix_multiply.h"

namespace cuda_lab::matrix_multiply {

template <typename T>
MatrixDevice<T>* CreateMatrixDevice(const MatrixHost<T>& host_matrix) {
  MatrixDevice<T>* device_matrix{nullptr};
  cudaMalloc(&device_matrix, sizeof(MatrixDevice<T>));
  cudaMemcpy(device_matrix, host_matrix.device(), sizeof(MatrixDevice<T>),
             cudaMemcpyHostToDevice);
  return device_matrix;
}

template <typename T>
__global__ void simple_matrix_multipy_kernel(
    cuda_lab::MatrixDevice<T> const* const a,
    cuda_lab::MatrixDevice<T> const* const b,
    cuda_lab::MatrixDevice<T>* const c) {
  auto const row{blockIdx.y * blockDim.y + threadIdx.y};
  auto const col{blockIdx.x * blockDim.x + threadIdx.x};

  if (row >= c->rows || col >= c->cols) {
    return;
  }

  T sum{0};
  for (std::size_t i{0}; i < a->cols; ++i) {
    sum += a->get(row, i) * b->get(i, col);
  }
  c->set(row, col, sum);
}

// Template wrapper for kernel launch
template <typename T>
void SimpleMatrixMultiplyKernel(cuda_lab::MatrixHost<T> const& a,
                                cuda_lab::MatrixHost<T> const& b,
                                cuda_lab::MatrixHost<T>& c) {
  // Allocate device memory for MatrixDevice<T> structs
  auto* d_a{CreateMatrixDevice(a)};
  auto* d_b{CreateMatrixDevice(b)};
  auto* d_c{CreateMatrixDevice(c)};

  dim3 block_size(kBlockSize, kBlockSize);
  dim3 grid_size(static_cast<unsigned int>(DivUp(c.cols(), block_size.x)),
                 static_cast<unsigned int>(DivUp(c.rows(), block_size.y)));

  cudaDeviceSynchronize();
  simple_matrix_multipy_kernel<T><<<grid_size, block_size>>>(d_a, d_b, d_c);
  cudaDeviceSynchronize();

  // No need to copy struct back unless fields are changed
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

// Explicit instantiation for float and int (pointer version)
template __global__ void simple_matrix_multipy_kernel<float>(
    cuda_lab::MatrixDevice<float> const* const,
    cuda_lab::MatrixDevice<float> const* const,
    cuda_lab::MatrixDevice<float>* const);
template void SimpleMatrixMultiplyKernel<float>(
    cuda_lab::MatrixHost<float> const&, cuda_lab::MatrixHost<float> const&,
    cuda_lab::MatrixHost<float>&);
template __global__ void simple_matrix_multipy_kernel<int>(
    cuda_lab::MatrixDevice<int> const* const,
    cuda_lab::MatrixDevice<int> const* const,
    cuda_lab::MatrixDevice<int>* const);
template void SimpleMatrixMultiplyKernel<int>(cuda_lab::MatrixHost<int> const&,
                                              cuda_lab::MatrixHost<int> const&,
                                              cuda_lab::MatrixHost<int>&);

}  // namespace cuda_lab::matrix_multiply
