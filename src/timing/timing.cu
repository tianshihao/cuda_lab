#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#include "timing.h"

namespace cuda_lab::timing {

void CheckCudaError(cudaError_t const err, char const* msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

__global__ void VectorAddKernel(int const* d_a, int const* d_b, int* d_c,
                                unsigned int N) {
  auto idx{blockIdx.x * blockDim.x + threadIdx.x};
  if (idx < N) {
    d_c[idx] = d_a[idx] + d_b[idx];
  }
}

void VectorAdd(int const* h_a, int const* h_b, int* h_c, unsigned int N) {
  int *d_a, *d_b, *d_c;

  // Assign device memory
  cudaError_t err{cudaMalloc((void**)&d_a, N * sizeof(int))};
  CheckCudaError(err, "Failed to allocate device memory for d_a");

  err = cudaMalloc((void**)&d_b, N * sizeof(int));
  CheckCudaError(err, "Failed to allocate device memory for d_b");

  err = cudaMalloc((void**)&d_c, N * sizeof(int));
  CheckCudaError(err, "Failed to allocate device memory for d_c");

  // Transfer data from host to device
  err = cudaMemcpy(d_a, h_a, N * sizeof(int),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);
  CheckCudaError(err, "Failed to copy h_a to d_a");

  err = cudaMemcpy(d_b, h_b, N * sizeof(int),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);
  CheckCudaError(err, "Failed to copy h_b to d_b");

  // Define the block size and grid size
  int blockSize{256};
  int gridSize{static_cast<int>((N + blockSize - 1) / blockSize)};

  // Calling the kernel
  VectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

  // Check for kernel errors
  err = cudaGetLastError();
  CheckCudaError(err, "Kernel execution failed");

  // Transfer data from device to host
  err = cudaMemcpy(h_c, d_c, N * sizeof(int),
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);
  CheckCudaError(err, "Failed to copy d_c to h_c");

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

void VectorAdd(cudaStream_t stream, int const* h_a, int const* h_b, int* h_c,
               unsigned int N) {
  int *d_a, *d_b, *d_c;

  // Assign device memory
  cudaError_t err{cudaMalloc((void**)&d_a, N * sizeof(int))};
  CheckCudaError(err, "Failed to allocate device memory for d_a");

  err = cudaMalloc((void**)&d_b, N * sizeof(int));
  CheckCudaError(err, "Failed to allocate device memory for d_b");

  err = cudaMalloc((void**)&d_c, N * sizeof(int));
  CheckCudaError(err, "Failed to allocate device memory for d_c");

  // Transfer data from host to device
  err = cudaMemcpy(d_a, h_a, N * sizeof(int),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);
  CheckCudaError(err, "Failed to copy h_a to d_a");

  err = cudaMemcpy(d_b, h_b, N * sizeof(int),
                   cudaMemcpyKind::cudaMemcpyHostToDevice);
  CheckCudaError(err, "Failed to copy h_b to d_b");

  // Define the block size and grid size
  int blockSize{256};
  int gridSize{static_cast<int>((N + blockSize - 1) / blockSize)};

  // Calling the kernel
  // VectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
  VectorAddKernel<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_c, N);

  // Check for kernel errors
  err = cudaGetLastError();
  CheckCudaError(err, "Kernel execution failed");

  // Transfer data from device to host
  err = cudaMemcpy(h_c, d_c, N * sizeof(int),
                   cudaMemcpyKind::cudaMemcpyDeviceToHost);
  CheckCudaError(err, "Failed to copy d_c to h_c");

  // Free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

}  // namespace cuda_lab::timing