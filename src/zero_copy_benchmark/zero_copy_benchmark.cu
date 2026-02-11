#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>

namespace cuda_lab::zero_copy_benchmark {

__global__ void kernel_zero_copy(float* ptr, float* out, int n) {
  int idx{blockIdx.x * blockDim.x + threadIdx.x};
  if (idx < n) {
    // Directly access host memory through zero-copy
    out[idx] = ptr[idx] * 2;
  }
}

__global__ void kernel_device_copy(float* ptr, float* out, int n) {
  int idx{blockIdx.x * blockDim.x + threadIdx.x};
  if (idx < n) {
    // Access device memory (ptr is expected to be in device memory)
    out[idx] = ptr[idx] * 2;
  }
}

void KernelZeroCopy(float* device, float* host, int n) {
  auto block_size{256};
  auto grid_size{(n + block_size - 1) / block_size};

  kernel_zero_copy<<<grid_size, block_size>>>(device, host, n);
  cudaDeviceSynchronize();
}

void KernelDeviceCopy(float* device, float* host, int n) {
  auto block_size{256};
  auto grid_size{(n + block_size - 1) / block_size};

  kernel_device_copy<<<grid_size, block_size>>>(device, host, n);
  cudaDeviceSynchronize();
}
}  // namespace cuda_lab::zero_copy_benchmark
