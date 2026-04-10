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

template <typename T>
__global__ void coalesced_matrix_multipy_kernel(
    cuda_lab::MatrixDevice<T> const* const a,
    cuda_lab::MatrixDevice<T> const* const b,
    cuda_lab::MatrixDevice<T>* const c) {
  auto const row{blockIdx.y * blockDim.y + threadIdx.y};
  auto const col{blockIdx.x * blockDim.x + threadIdx.x};

  __shared__ T tile_a[kBlockSize][kBlockSize];

  std::size_t const K{a->cols};
  std::size_t const num_tiles{(K + kBlockSize - 1) / kBlockSize};

  T sum{0};

  for (std::size_t tile{0}; tile < num_tiles; ++tile) {
    std::size_t const k_idx{tile * kBlockSize + threadIdx.x};
    if (row < a->rows && k_idx < K) {
      tile_a[threadIdx.y][threadIdx.x] = a->get(row, k_idx);
    } else {
      tile_a[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if (row < c->rows && col < c->cols) {
      std::size_t const k_in_tile{
          (tile + 1) * kBlockSize <= K ? kBlockSize : K - tile * kBlockSize};
      for (std::size_t i{0}; i < k_in_tile; ++i) {
        sum += tile_a[threadIdx.y][i] * b->get(tile * kBlockSize + i, col);
      }
    }
    __syncthreads();
  }

  if (row < c->rows && col < c->cols) {
    c->set(row, col, sum);
  }
}

template <typename T>
__global__ void shared_ab_matrix_multipy_kernel(
    cuda_lab::MatrixDevice<T> const* const a,
    cuda_lab::MatrixDevice<T> const* const b,
    cuda_lab::MatrixDevice<T>* const c) {
  auto const row{blockIdx.y * blockDim.y + threadIdx.y};
  auto const col{blockIdx.x * blockDim.x + threadIdx.x};

  __shared__ T tile_a[kBlockSize][kBlockSize], tile_b[kBlockSize][kBlockSize];

  std::size_t const K{a->cols};
  std::size_t const num_tiles{(K + kBlockSize - 1) / kBlockSize};

  T sum{0};

  for (std::size_t tile{0}; tile < num_tiles; ++tile) {
    std::size_t const k_col{tile * kBlockSize + threadIdx.x};
    if (row < a->rows && k_col < K) {
      tile_a[threadIdx.y][threadIdx.x] = a->get(row, k_col);
    } else {
      tile_a[threadIdx.y][threadIdx.x] = 0;
    }

    std::size_t const k_row{tile * kBlockSize + threadIdx.y};
    if (k_row < K && col < b->cols) {
      tile_b[threadIdx.y][threadIdx.x] = b->get(k_row, col);
    } else {
      tile_b[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    if (row < c->rows && col < c->cols) {
      std::size_t const k_in_tile{
          (tile + 1) * kBlockSize <= K ? kBlockSize : K - tile * kBlockSize};
      for (std::size_t i{0}; i < k_in_tile; ++i) {
        sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
      }
    }
    __syncthreads();
  }

  if (row < c->rows && col < c->cols) {
    c->set(row, col, sum);
  }
}

template <typename T>
void MatrixMultiplyWrapper(cuda_lab::MatrixHost<T> const& a,
                           cuda_lab::MatrixHost<T> const& b,
                           cuda_lab::MatrixHost<T>& c,
                           MatrixMultiplyType type) {
  // Allocate device memory for MatrixDevice<T> structs
  auto* d_a{CreateMatrixDevice(a)};
  auto* d_b{CreateMatrixDevice(b)};
  auto* d_c{CreateMatrixDevice(c)};

  dim3 block_size(kBlockSize, kBlockSize);
  dim3 grid_size(static_cast<unsigned int>(DivUp(c.cols(), block_size.x)),
                 static_cast<unsigned int>(DivUp(c.rows(), block_size.y)));

  switch (type) {
    case MatrixMultiplyType::kSimple:
      simple_matrix_multipy_kernel<T><<<grid_size, block_size>>>(d_a, d_b, d_c);
      break;
    case MatrixMultiplyType::kCoalesced:
      coalesced_matrix_multipy_kernel<T>
          <<<grid_size, block_size>>>(d_a, d_b, d_c);
      break;
    case MatrixMultiplyType::kSharedAB:
      shared_ab_matrix_multipy_kernel<T>
          <<<grid_size, block_size>>>(d_a, d_b, d_c);
      break;
  }
  cudaDeviceSynchronize();

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template void MatrixMultiplyWrapper<int>(cuda_lab::MatrixHost<int> const&,
                                         cuda_lab::MatrixHost<int> const&,
                                         cuda_lab::MatrixHost<int>&,
                                         MatrixMultiplyType);
template void MatrixMultiplyWrapper<float>(cuda_lab::MatrixHost<float> const&,
                                           cuda_lab::MatrixHost<float> const&,
                                           cuda_lab::MatrixHost<float>&,
                                           MatrixMultiplyType);

}  // namespace cuda_lab::matrix_multiply
