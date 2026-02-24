#include "zero_copy_benchmark.h"

#include <cuda_runtime.h>

#include "cuda_scoped_timer.hpp"
#include "memory.hpp"

namespace cuda_lab::zero_copy_benchmark {
// Size for test (small & large)
constexpr int kSize{(1 << 22)};  // 4MB
constexpr int kIter{10};

void TestMemcpyH2d(float* host, float* device, bool pinned) {
  CudaScopedTimer timer(pinned ? "PinnedMemcpyH2d" : "NormalMemcpyH2d");

  for (int i{0}; i < kIter; ++i) {
    cudaMemcpy(device, host, kSize * sizeof(float), cudaMemcpyHostToDevice);
  }

  auto ms{timer.finish()};
  printf("%s h2d memcpy (GB/s): %.2f\n", pinned ? "Pinned" : "Normal",
         kSize * sizeof(float) * kIter / (ms * 1e6));
}

void TestZeroCopy(float* device, float* host) {
  CudaScopedTimer timer("KernelZeroCopy");

  for (int i{0}; i < kIter; ++i) {
    KernelZeroCopy(device, host, kSize);
  }

  auto ms{timer.finish()};
  printf("Zero-copy kernel read (GB/s): %.2f\n",
         kSize * sizeof(float) * kIter / (ms * 1e6));
}

void TestDeviceCopy(float* device, float* host) {
  CudaScopedTimer timer("KernelDeviceCopy");

  for (int i{0}; i < kIter; ++i) {
    KernelDeviceCopy(device, host, kSize);
  }

  auto ms{timer.finish()};
  printf("Device mem kernel read (GB/s): %.2f\n",
         kSize * sizeof(float) * kIter / (ms * 1e6));
}

void ZeroCopyBenchmark() {
  // 1. Test normal h2d memcpy
  Memory<float, MemoryType::kHost> h_normal(kSize, 1.0f);
  Memory<float, MemoryType::kDevice> d_mem(kSize, 0.0f);
  TestMemcpyH2d(h_normal.data(), d_mem.data(), false);

  // 2. Test pinned memory
  Memory<float, MemoryType::kPinned> h_pinned(kSize, 1.0f);
  TestMemcpyH2d(h_pinned.data(), d_mem.data(), true);

  // 3. Test zero-copy mapped pinned memory
  Memory<float, MemoryType::kMappedPinned> h_zc(kSize, 2.0f);
  Memory<float, MemoryType::kDevice> d_out(kSize, 0.0f);
  TestZeroCopy(h_zc.device_ptr(), d_out.data());

  // 4. Test device memory copy kernel (for reference)
  TestDeviceCopy(d_mem.data(), d_out.data());
}

}  // namespace cuda_lab::zero_copy_benchmark
