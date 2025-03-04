#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "hello_cuda.h"

namespace cuda_image_lab::hello_cuda {
__global__ void HelloCudaKernel() { printf("HelloCuda, CUDA!\n"); }

void HelloCuda() {
  // Launch 1 block of 1 thread.
  HelloCudaKernel<<<1, 1>>>();
  cudaError_t cudaStatus{cudaDeviceSynchronize()};
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s\n",
            cudaStatus, cudaGetErrorString(cudaStatus));
  }
}


__global__ void ExampleKernel1D() {
  auto thread_id_x{threadIdx.x};  // Thread index within the block
  auto block_id_x{blockIdx.x};    // Block index within the grid
  auto global_id_x{block_id_x * blockDim.x +
                   thread_id_x};  // Global index across the entire grid

  auto thread_in_block{thread_id_x};
  auto block_in_grid{block_id_x};
  auto one_block_size{blockDim.x};
  auto global_in_grid{thread_in_block + block_in_grid * one_block_size};

  printf("Thread %u in Block %u has Global ID %u, global_in_grid %u\n",
         thread_id_x, block_id_x, global_id_x, global_in_grid);
}

__global__ void ExampleKernel2D() {
  auto thread_id_x{threadIdx.x};  // Thread index within the block
  auto block_id_x{blockIdx.x};    // Block index within the grid
  auto global_id_x{block_id_x * blockDim.x +
                   thread_id_x};  // Global index across the entire grid

  auto thread_id_y{threadIdx.y};  // Thread index within the block
  auto block_id_y{blockIdx.y};    // Block index within the grid
  auto global_id_y{block_id_y * blockDim.y +
                   thread_id_y};  // Global index across the entire grid

  auto thread_in_block{thread_id_x + thread_id_y * blockDim.x};
  auto block_in_grid{block_id_x + block_id_y * gridDim.x};
  auto one_block_size{blockDim.x * blockDim.y};
  auto global_in_grid{thread_in_block + block_in_grid * one_block_size};

  printf(
      "Thread (%u, %u) in Block (%u, %u) has Global ID (%u, %u), "
      "global_in_grid %u\n",
      thread_id_x, thread_id_y, block_id_x, block_id_y, global_id_x,
      global_id_y, global_in_grid);
}

__global__ void ExampleKernel3D() {
  auto thread_id_x{threadIdx.x};  // Thread index within the block
  auto block_id_x{blockIdx.x};    // Block index within the grid
  auto global_id_x{block_id_x * blockDim.x +
                   thread_id_x};  // Global index across the entire grid

  auto thread_id_y{threadIdx.y};  // Thread index within the block
  auto block_id_y{blockIdx.y};    // Block index within the grid
  auto global_id_y{block_id_y * blockDim.y +
                   thread_id_y};  // Global index across the entire grid

  auto thread_id_z{threadIdx.z};  // Thread index within the block
  auto block_id_z{blockIdx.z};    // Block index within the grid
  auto global_id_z{block_id_z * blockDim.z +
                   thread_id_z};  // Global index across the entire grid

  auto thread_in_block{thread_id_x + thread_id_y * blockDim.x +
                       thread_id_z * blockDim.x * blockDim.y};
  auto block_in_grid{block_id_x + block_id_y * gridDim.x +
                     block_id_z * gridDim.x * gridDim.y};
  auto one_block_size{blockDim.x * blockDim.y * blockDim.z};
  auto global_in_grid{thread_in_block + block_in_grid * one_block_size};

  printf(
      "Thread (%u, %u, %u) in Block (%u, %u, %u) has Global ID (%u, %u, %u), "
      "global_in_grid: %u\n",
      thread_id_x, thread_id_y, thread_id_z, block_id_x, block_id_y, block_id_z,
      global_id_x, global_id_y, global_id_z, global_in_grid);
}

void KernalExample() {
  dim3 block_per_grid_1d(3);
  dim3 thredd_per_block_1d(3);
  ExampleKernel1D<<<block_per_grid_1d, thredd_per_block_1d>>>();
  cudaDeviceSynchronize();

  dim3 block_per_grid_2d(2, 2);
  dim3 thread_per_block_2d(2, 2);
  ExampleKernel2D<<<block_per_grid_2d, thread_per_block_2d>>>();
  cudaDeviceSynchronize();

  dim3 block_per_grid_3d(2, 2, 3);
  dim3 thread_per_block_3d(4, 2, 4);
  ExampleKernel3D<<<block_per_grid_3d, thread_per_block_3d>>>();
  cudaDeviceSynchronize();
}
}  // namespace cuda_image_lab::hello_cuda