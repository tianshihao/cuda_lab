#include <array>
#include <iostream>

#include "l2_cache.h"

int main() {
  auto const results{cuda_lab::l2_cache::RunL2CacheBenchmark()};
  std::cout << "L2 cache benchmark results (ms):\n";
  for (std::size_t i{0}; i < results.size(); ++i) {
    std::cout << "Test " << i << ": " << results[i] << " ms\n";
  }
  return 0;
}
