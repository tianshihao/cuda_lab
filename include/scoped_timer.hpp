#pragma once

#include <chrono>
#include <iostream>
#include <string_view>

class ScopedTimer {
 public:
  explicit ScopedTimer(std::string_view name) : name_{name} {
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
  std::string_view name_;
  std::chrono::high_resolution_clock::time_point start_;
};

#define SCOPED_TIMER(name) ScopedTimer timer##__LINE__(name)
