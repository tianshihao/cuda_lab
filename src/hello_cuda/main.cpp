#include "hello_cuda.h"

int main() {
  cuda_lab::hello_cuda::HelloCuda();
  cuda_lab::hello_cuda::KernalExample();
  // cuda_lab::hello_cuda::TwoSumCuda({}, {}, {});

  return 0;
}
