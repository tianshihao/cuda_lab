#include <gtest/gtest.h>

#include "memory.hpp"

using cuda_lab::Memory;
using cuda_lab::MemoryType;

TEST(MemoryTest, HostMemoryBasic) {
  constexpr std::size_t n{10};
  Memory<int, MemoryType::kHost> mem{n, 42};
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(mem.data()[i], 42);
  }
  mem.data()[5] = 99;
  EXPECT_EQ(mem.data()[5], 99);
}

TEST(MemoryTest, DeviceMemoryBasic) {
  constexpr std::size_t n{10};
  Memory<int, MemoryType::kDevice> mem{n, 7};
  int host[n];
  cudaMemcpy(host, mem.data(), n * sizeof(int), cudaMemcpyDeviceToHost);
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(host[i], 7);
  }
}
