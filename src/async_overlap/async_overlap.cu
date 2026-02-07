#include "async_overlap.h"

namespace cuda_lab::async_overlap {

// CUDA kernel: adds one to each element
__global__ void add_one_kernel(float* data, std::size_t n) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    ++data[idx];
  }

  // Sleep for 1ms.
}

// Host wrapper
void LaunchAddOne(float* device_ptr, std::size_t n, cudaStream_t stream) {
  dim3 block_size{256};
  dim3 grid_size{static_cast<unsigned int>(( n + block_size.x - 1 ) / block_size.x)};
  add_one_kernel<<<grid_size, block_size, 0, stream>>>(device_ptr, n);
}

}  // namespace cuda_lab::async_overlap
