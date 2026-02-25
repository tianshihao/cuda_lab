#include "l2_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <iostream>

#include "cuda_scoped_timer.hpp"
#include "memory.hpp"
#include "size_literals.hpp"

namespace cuda_lab::l2_cache {

cudaAccessPolicyWindow InitAccessPolicyWindow() {
  cudaAccessPolicyWindow access_policy_window{0};
  access_policy_window.base_ptr = (void*)0;
  access_policy_window.num_bytes = 0;
  access_policy_window.hitRatio = 0.f;
  access_policy_window.hitProp = cudaAccessPropertyNormal;
  access_policy_window.missProp = cudaAccessPropertyStreaming;
  return access_policy_window;
}

void PrintL2CacheInfo(cudaDeviceProp const& prop) {
  std::cout << "L2 cache size: " << prop.l2CacheSize / 1_MB << " MB\n";
  std::cout << "Max persisting L2 cache size: "
            << prop.persistingL2CacheMaxSize / 1_MB << " MB\n";
  std::cout << "Access policy max window size: "
            << prop.accessPolicyMaxWindowSize / 1_MB << " MB\n";
}

std::array<float, kNumTests> RunL2CacheBenchmark() {
  int const device{0};
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device);
  PrintL2CacheInfo(prop);

  // 1. Set the persisting L2 cache size to the maximum allowed by the device.
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                     prop.persistingL2CacheMaxSize);

  // 2. Prepare data that exceeds the L2 cache size to ensure eviction.
  int const l2_mb{static_cast<int>(prop.l2CacheSize / 1_MB)};
  int const set_aside_mb{
      static_cast<int>(prop.persistingL2CacheMaxSize / 1_MB)};
  int const data_mb{std::max(16, l2_mb)};
  std::cout
      << "Using total data size of " << data_mb
      << " MB (l2_cache_size=" << l2_mb << " MB, set_aside=" << set_aside_mb
      << " MB) to ensure eviction and test different freq window sizes.\n";
  std::size_t const data_size{static_cast<std::size_t>(data_mb * 1_MB) /
                              sizeof(int)};

  Memory<int, MemoryType::kHost> h_data{data_size, 0};
  for (std::size_t i{0}; i < data_size; ++i) {
    h_data.data()[i] = static_cast<int>(i);
  }

  Memory<int, MemoryType::kDevice> d_persistent{data_size};
  Memory<int, MemoryType::kDevice> d_stream{data_size};
  cudaMemcpy(d_persistent.data(), h_data.data(), sizeof(int) * data_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_stream.data(), h_data.data(), sizeof(int) * data_size,
             cudaMemcpyHostToDevice);

  // 3. Initialize the stream's access policy window to cover the "frequently
  // accessed"
  cudaStreamAttrValue stream_attribute{};
  stream_attribute.accessPolicyWindow = InitAccessPolicyWindow();

  int const test_freq_mbs[kNumTests]{1, set_aside_mb - 1, set_aside_mb,
                                     set_aside_mb + 1, l2_mb};
  std::array<float, kNumTests> results{};

  cudaStream_t stream{};
  cudaStreamCreate(&stream);

  for (std::size_t j{0}; j < kNumTests; ++j) {
    int const freq_mb{test_freq_mbs[j]};
    if (freq_mb <= 0 || freq_mb >= data_mb) {
      results[j] = -1.0f;
      continue;
    }
    int const freq_size{static_cast<int>((freq_mb * 1_MB) / sizeof(int))};

    // 4. Set the access policy window to cover the "frequently accessed"
    // portion of the data.
    stream_attribute.accessPolicyWindow.base_ptr = d_persistent.data();
    stream_attribute.accessPolicyWindow.num_bytes = freq_mb * 1_MB;

    stream_attribute.accessPolicyWindow.hitRatio = 1.f;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow,
                           &stream_attribute);

    CudaScopedTimer timer{"freq_window=" + std::to_string(freq_mb) + "MB"};
    LaunchSlidingWindowKernel(stream, d_persistent.data(), d_stream.data(),
                              static_cast<int>(data_size), freq_size);
    cudaStreamSynchronize(stream);
    results[j] = timer.finish();

    std::cout << "freq_window=" << freq_mb << "MB, time=" << results[j]
              << " ms\n";
  }

  cudaStreamDestroy(stream);

  return results;
}

}  // namespace cuda_lab::l2_cache
