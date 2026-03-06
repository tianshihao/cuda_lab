#include <cuda_runtime.h>

#include <cstddef>

#include "cuda_check.hpp"

namespace cuda_lab::operators {

template <typename T>
__global__ void add_one_kernel(T* ptr, std::size_t n) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    ptr[idx] += T{1};
  }
}

template <typename T>
void add_one_kernel_launcher(T* ptr, std::size_t n) {
  constexpr std::size_t block_size{256};
  auto grid_size{(n + block_size - 1) / block_size};
  add_one_kernel<<<grid_size, block_size>>>(ptr, n);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// Explicit instantiations for common types
template void add_one_kernel_launcher<int>(int*, std::size_t);
template void add_one_kernel_launcher<float>(float*, std::size_t);
template void add_one_kernel_launcher<double>(double*, std::size_t);

}  // namespace cuda_lab::operators
