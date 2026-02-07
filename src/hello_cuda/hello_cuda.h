#pragma once

#include <vector>

namespace cuda_lab::hello_cuda {
void HelloCuda();
void KernalExample();

void TwoSumCuda(std::vector<int> const& input1, std::vector<int> const& input2,
                std::vector<int>& output);
}  // namespace cuda_lab::hello_cuda
