#pragma once

#include <chrono>
#include <iostream>

namespace cuda_lab {
class ScopedTimer {
 public:
  explicit ScopedTimer(std::string name) : name_{name} {
    start_ = std::chrono::high_resolution_clock::now();
  }

  ~ScopedTimer() {
    auto end{std::chrono::high_resolution_clock::now()};
    auto duration{
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_)
            .count()};
    std::cout << name_ << " took " << duration << " ms." << std::endl;
  }

 private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};

#define SCOPED_TIMER(name) ScopedTimer timer##__LINE__(name)

}  // namespace cuda_lab
