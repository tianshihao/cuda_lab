#pragma once

namespace cuda_lab::zero_copy_benchmark {
void ZeroCopyBenchmark();
void KernelZeroCopy(float* device, float* host, int n);
void KernelDeviceCopy(float* device, float* host, int n);
}  // namespace cuda_lab::zero_copy_benchmark
