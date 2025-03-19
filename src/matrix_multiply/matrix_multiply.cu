#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#include "matrix_multiply.h"

namespace cuda_lab::matrix_multiply {

__global__ void SimpleMultiply(float* a, float* b, float* c, int A_rows,
                               int A_cols, int B_cols) {
  auto row{blockIdx.y * blockDim.y + threadIdx.y};
  auto col{blockIdx.x * blockDim.x + threadIdx.x};

  // Output sample kernel info
  if (row == A_rows / 2 && col == B_cols / 2) {
    printf("Sample kernel info:\n");
    printf(
        "Block (%d, %d), Thread (%d, %d)\nRow = blockIdx.y * blockDim.y + "
        "threadIdx.y = %d * %d + %d = %d\nCol = blockIdx.x * blockDim.x + "
        "threadIdx.x = %d * %d + %d = %d\n\n",
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockIdx.y,
        blockDim.y, threadIdx.y, row, blockIdx.x, blockIdx.y, threadIdx.x, col);
  }

  if (row < A_rows && col < B_cols) {
    auto sum{0.0f};
    for (int i{0}; i < A_cols; i++) {
      sum += a[row * A_cols + i] * b[i * B_cols + col];
    }
    c[row * B_cols + col] = sum;
  }
}

void MatrixMultiply(float* h_a, float* h_b, float* h_c, int A_rows, int A_cols,
                    int B_cols) {
  float *d_a{nullptr}, *d_b{nullptr}, *d_c{nullptr};

  // Allocate device memory
  auto err{cudaMalloc((void**)&d_a, A_rows * A_cols * sizeof(float))};
  CheckCudaError(err, "Failed to allocate device memory for d_a");

  err = cudaMalloc((void**)&d_b, A_cols * B_cols * sizeof(float));
  CheckCudaError(err, "Failed to allocate device memory for d_b");

  err = cudaMalloc((void**)&d_c, A_rows * B_cols * sizeof(float));
  CheckCudaError(err, "Failed to allocate device memory for d_c");

  // Transfer data from host to device
  err = cudaMemcpy(d_a, h_a, A_rows * A_cols * sizeof(float),
                   cudaMemcpyHostToDevice);
  CheckCudaError(err, "Failed to copy h_a to d_a");

  err = cudaMemcpy(d_b, h_b, A_cols * B_cols * sizeof(float),
                   cudaMemcpyHostToDevice);
  CheckCudaError(err, "Failed to copy h_b to d_b");

  dim3 block_size(TILE_DIM, TILE_DIM);
  dim3 grid_size(DivUp(B_cols, TILE_DIM), DivUp(A_rows, TILE_DIM));

  // Print kernel info
  std::cout << "Kernel info: " << std::endl;
  std::cout << "Data size: " << "row: " << A_rows << " x col: " << B_cols
            << std::endl;
  std::cout << "Element number: " << A_rows * B_cols << std::endl;
  std::cout << "Block size: " << block_size.x << " x " << block_size.y
            << std::endl;
  std::cout << "Grid size: " << grid_size.x << " x " << grid_size.y << std::endl
            << std::endl;

  // Calling the kernel
  SimpleMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c, A_rows, A_cols,
                                            B_cols);

  cudaStreamSynchronize(0);  // Wait for the kernel to finish
  // Check for kernel errors
  err = cudaGetLastError();
  CheckCudaError(err, "Kernel execution failed");

  // Transfer data from device to host
  err = cudaMemcpy(h_c, d_c, A_rows * B_cols * sizeof(float),
                   cudaMemcpyDeviceToHost);
  CheckCudaError(err, "Failed to copy d_c to h_c");

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

}  // namespace cuda_lab::matrix_multiply