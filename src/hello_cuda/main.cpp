#include <chrono>
#include <iostream>
#include <vector>

#include "hello_cuda.h"

void Hello() { cuda_image_lab::hello_cuda::HelloCuda(); }

void KernalExample() { cuda_image_lab::hello_cuda::KernalExample(); }

void ShowLena() { cuda_image_lab::hello_cuda::ShowLena(); }
void ShowImage() { cuda_image_lab::hello_cuda::ShowImage(); }

int main() {
  Hello();
  // ShowLena();
  // ShowImage();
  // KernalExample();

  return 0;
}