#pragma once

#include <string_view>
#include <vector>

namespace cuda_image_lab::hello_cuda {
void HelloCuda();
void AddCuda(std::vector<int> const &input1, std::vector<int> const &input2,
             std::vector<int> &output);
void KernalExample();

void ShowLena();
void ShowImage();

}  // namespace cuda_image_lab::hello_cuda
