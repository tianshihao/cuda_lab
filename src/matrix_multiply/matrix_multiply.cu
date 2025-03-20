#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#include "matrix_multiply.h"

namespace cuda_lab::matrix_multiply {

__global__ void CoalescedMultiply(float const* const a, float const* const b,
                                  float* const c, std::size_t const a_rows,
                                  std::size_t const a_cols,
                                  std::size_t const b_cols) {
  // Share memory on chip
  __shared__ float a_tile[TILE_DIM][TILE_DIM];

  auto row{blockIdx.y * blockDim.y + threadIdx.y};
  auto col{blockIdx.x * blockDim.x + threadIdx.x};

  if (row >= a_rows || col >= b_cols) {
    return;
  }

  // Cache all rows in the block.
  a_tile[threadIdx.y][threadIdx.x] = a[row * a_cols + threadIdx.x];

  __syncwarp();

  auto sum{0.0f};

  // i to iterate over the columns of A and rows of B.
  for (std::size_t i{0}; i < a_cols; ++i) {
    sum += a_tile[threadIdx.y][i] * b[i * b_cols + col];
  }
  c[row * b_cols + col] = sum;
}

__global__ void SimpleMultiply(float* a, float* b, float* c, int a_rows,
                               int a_cols, int b_cols) {
  auto row{blockIdx.y * blockDim.y + threadIdx.y};
  auto col{blockIdx.x * blockDim.x + threadIdx.x};

  if (row >= a_rows || col >= b_cols) {
    return;
  }

  // Output sample kernel info
  if (row == a_rows / 2 && col == b_cols / 2) {
    printf("Sample kernel info:\n");
    printf(
        "Block (%d, %d), Thread (%d, %d)\nRow = blockIdx.y * blockDim.y + "
        "threadIdx.y = %d * %d + %d = %d\nCol = blockIdx.x * blockDim.x + "
        "threadIdx.x = %d * %d + %d = %d\n\n",
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockIdx.y,
        blockDim.y, threadIdx.y, row, blockIdx.x, blockIdx.y, threadIdx.x, col);
  }

  auto sum{0.0f};
  for (std::size_t i{0}; i < a_cols; ++i) {
    sum += a[row * a_cols + i] * b[i * b_cols + col];
  }
  c[row * b_cols + col] = sum;
}

void MatrixMultiply(float const* const h_a, float const* const h_b,
                    float* const h_c, std::size_t const a_rows,
                    std::size_t const a_cols, std::size_t const b_cols,
                    MatrixMultiplyType const type) {
  float *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  // Allocate device memory
  auto err{cudaMalloc((void**)&d_a, a_rows * a_cols * sizeof(float))};
  CheckCudaError(err, "Failed to allocate device memory for d_a");

  err = cudaMalloc((void**)&d_b, a_cols * b_cols * sizeof(float));
  CheckCudaError(err, "Failed to allocate device memory for d_b");

  err = cudaMalloc((void**)&d_c, a_rows * b_cols * sizeof(float));
  CheckCudaError(err, "Failed to allocate device memory for d_c");

  // Transfer data from host to device
  err = cudaMemcpy(d_a, h_a, a_rows * a_cols * sizeof(float),
                   cudaMemcpyHostToDevice);
  CheckCudaError(err, "Failed to copy h_a to d_a");

  err = cudaMemcpy(d_b, h_b, a_cols * b_cols * sizeof(float),
                   cudaMemcpyHostToDevice);
  CheckCudaError(err, "Failed to copy h_b to d_b");

  dim3 block_size(TILE_DIM, TILE_DIM);
  dim3 grid_size(DivUp(b_cols, TILE_DIM), DivUp(a_rows, TILE_DIM));

  // Print kernel info
  std::cout << "Kernel info: " << std::endl;
  std::cout << "Data size: " << "row: " << a_rows << " x col: " << b_cols
            << std::endl;
  std::cout << "Element number: " << a_rows * b_cols << std::endl;
  std::cout << "Block size: " << block_size.x << " x " << block_size.y
            << std::endl;
  std::cout << "Grid size: " << grid_size.x << " x " << grid_size.y << std::endl
            << std::endl;

  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start, 0);

  // Calling the kernel
  switch (type) {
    case MatrixMultiplyType::kSimple: {
      SimpleMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c, a_rows, a_cols,
                                                b_cols);
      break;
    }
    case MatrixMultiplyType::kCoalesced: {
      CoalescedMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c, a_rows,
                                                   a_cols, b_cols);
      break;
    }
    default:
      break;
    default: {
      break;
    }
  }

  // Record the stop event
  cudaEventRecord(stop, 0);

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  switch (type) {
    case MatrixMultiplyType::kSimple:
      std::cout << "SimpleMultiply executed." << std::endl;
      break;

    case MatrixMultiplyType::kCoalesced:
      std::cout << "CoalescedMultiply executed." << std::endl;
      break;

    default:
      break;
  }

  // Calculate the elapsed time
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "Kernel execution time: " << elapsed_time << " ms" << std::endl;

  // Calculate bandwidth
  // A and B matrices
  float data_size_gb{2.0f * a_rows * a_cols * sizeof(float) /
                     (1024.0f * 1024.0f * 1024.0f)};
  data_size_gb += a_rows * b_cols * sizeof(float) /
                  (1024.0f * 1024.0f * 1024.0f);             // C matrix
  float bandwidth{data_size_gb / (elapsed_time / 1000.0f)};  // Convert ms to s
  std::cout << "Kernel bandwidth: " << bandwidth << " GB/s" << std::endl
            << std::endl;

  // Destroy CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaStreamSynchronize(0);  // Wait for the kernel to finish
  // Check for kernel errors
  err = cudaGetLastError();
  CheckCudaError(err, "Kernel execution failed");

  // Transfer data from device to host
  err = cudaMemcpy(h_c, d_c, a_rows * b_cols * sizeof(float),
                   cudaMemcpyDeviceToHost);
  CheckCudaError(err, "Failed to copy d_c to h_c");

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

}  // namespace cuda_lab::matrix_multiply