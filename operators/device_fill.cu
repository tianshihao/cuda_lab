#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_check.hpp"

namespace cuda_lab::operators {

template <typename T>
__global__ void device_fill_kernel(T* ptr, std::size_t n, T value) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    ptr[idx] = value;
  }
}

template <typename T>
void device_fill_kernel_launcher(T* ptr, std::size_t n, T value) {
  constexpr std::size_t block_size{256};
  std::size_t grid_size = (n + block_size - 1) / block_size;
  device_fill_kernel<<<grid_size, block_size>>>(ptr, n, value);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit instantiations for common types (add more as needed)
template void device_fill_kernel_launcher<int>(int*, std::size_t, int);
template void device_fill_kernel_launcher<float>(float*, std::size_t, float);
template void device_fill_kernel_launcher<double>(double*, std::size_t, double);

}  // namespace cuda_lab::operators
