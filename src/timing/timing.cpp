#include "timing.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace cuda_lab::timing {

constexpr static unsigned int size{100000000};

void TimingWithDefaultStreamUsingCpuTimer() {
  std::vector<int> a(size, 1);
  std::vector<int> b(size, 2);
  std::vector<int> c(size);

  auto start{std::chrono::high_resolution_clock::now()};
  cuda_lab::timing::VectorAdd(a.data(), b.data(), c.data(), size);
  cudaDeviceSynchronize();
  auto end{std::chrono::high_resolution_clock::now()};
  std::chrono::duration<double> elapsed_seconds{end - start};
  std::cout << "Elapsed time with default stream: " << elapsed_seconds.count()
            << "s\n";
}

void TimingWithCustomStreamUsingCpuTimer() {
  std::vector<int> a(size, 1);
  std::vector<int> b(size, 2);
  std::vector<int> c(size);

  cudaStream_t stream1, stream2, stream3;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);
  cudaStreamCreate(&stream3);

  cuda_lab::timing::VectorAdd(stream1, a.data(), b.data(), c.data(), size);
  cuda_lab::timing::VectorAdd(stream2, a.data(), b.data(), c.data(), size);

  auto start{std::chrono::high_resolution_clock::now()};
  cuda_lab::timing::VectorAdd(stream3, a.data(), b.data(), c.data(), size);
  cudaDeviceSynchronize();
  auto end{std::chrono::high_resolution_clock::now()};
  std::chrono::duration<double> elapsed_seconds{end - start};
  std::cout << "Elapsed time with custom stream: " << elapsed_seconds.count()
            << "s\n";
}

void TimingWithDefaultStreamUsingGpuTimer() {
  std::vector<int> a(size, 1);
  std::vector<int> b(size, 2);
  std::vector<int> c(size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cuda_lab::timing::VectorAdd(a.data(), b.data(), c.data(), size);
  cudaEventRecord(stop);
  cudaDeviceSynchronize();
  cudaEventSynchronize(stop);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "Elapsed time with default stream: " << elapsed_time << "ms\n";
}

}  // namespace cuda_lab::timing

int main() {
  cuda_lab::timing::TimingWithDefaultStreamUsingCpuTimer();
  // cuda_lab::timing::TimingWithCustomStreamUsingCpuTimer();
  cuda_lab::timing::TimingWithDefaultStreamUsingGpuTimer();

  return 0;
}