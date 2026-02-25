#include <cuda_runtime.h>

#include "l2_cache.h"

namespace cuda_lab::l2_cache {

__global__ void sliding_window_kernel(int* persist_data, int* stream_data,
                                      int data_size, int freq_size) {
  auto const tid{blockIdx.x * blockDim.x + threadIdx.x};
  persist_data[tid % freq_size] += tid;
  stream_data[tid % data_size] += tid;
}

void LaunchSlidingWindowKernel(cudaStream_t stream, int* d_persistent,
                               int* d_stream, int data_size, int freq_size) {
  int const block{256};
  int const grid{static_cast<int>((data_size + block - 1) / block)};
  for (int i{0}; i < 100; ++i) {
    sliding_window_kernel<<<grid, block, 0, stream>>>(d_persistent, d_stream,
                                                      data_size, freq_size);
  }
}

}  // namespace cuda_lab::l2_cache
