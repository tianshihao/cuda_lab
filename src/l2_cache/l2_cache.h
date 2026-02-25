#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>

namespace cuda_lab::l2_cache {

constexpr std::size_t kNumTests{5};

void LaunchSlidingWindowKernel(cudaStream_t stream, int* d_persistent,
                               int* d_stream, int data_size, int freq_size);

std::array<float, kNumTests> RunL2CacheBenchmark();

}  // namespace cuda_lab::l2_cache
