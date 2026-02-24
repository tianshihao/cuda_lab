#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

namespace cuda_lab {

class CudaScopedTimer {
 public:
  explicit CudaScopedTimer(std::string name = std::string("cuda_scoped_timer"))
      : name_{name}, finished_{false}, ms_{0.0f} {
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);
    cudaEventRecord(start_);
  }

  ~CudaScopedTimer() {
    if (!finished_) {
      finish();
      print();
    }
    cudaEventDestroy(start_);
    cudaEventDestroy(end_);
  }

  float finish() {
    if (!finished_) {
      cudaEventRecord(end_);
      cudaEventSynchronize(end_);
      cudaEventElapsedTime(&ms_, start_, end_);
      finished_ = true;
    }
    return ms_;
  }

  float elapsed_ms() { return finish(); }

  void print() const { printf("%s took %.2f ms.\n", name_.c_str(), ms_); }

 private:
  std::string name_{};
  cudaEvent_t start_{nullptr};
  cudaEvent_t end_{nullptr};
  float ms_{0.0f};
  bool finished_{false};
};

#define CUDA_SCOPED_TIMER(name) \
  CudaScopedTimer cuda_scoped_timer_##__LINE__(name)

}  // namespace cuda_lab
