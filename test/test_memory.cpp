#include <gtest/gtest.h>

#include "cuda_check.hpp"
#include "memory.hpp"
#include "memory_type.hpp"

using cuda_lab::Memory;
using cuda_lab::MemoryType;

TEST(MemoryTest, HostMemoryBasic) {
  constexpr std::size_t n{4};
  Memory<int, MemoryType::kHost> mem{n};
  for (std::size_t i{0}; i < n; ++i) {
    mem.data()[i] = static_cast<int>(i * 10);
  }
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(mem.data()[i], static_cast<int>(i * 10));
  }
}

TEST(MemoryTest, HostMemoryConstructionFill) {
  constexpr std::size_t n{4};
  Memory<int, MemoryType::kHost> mem{n, 42};
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(mem.data()[i], 42);
  }
}

TEST(MemoryTest, HostMemoryFill) {
  constexpr std::size_t n{4};
  Memory<double, MemoryType::kHost> mem{n};
  mem.fill(3.14);
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_DOUBLE_EQ(mem.data()[i], 3.14);
  }
}

TEST(MemoryTest, PinnedMemoryBasic) {
  constexpr std::size_t n{4};
  Memory<float, MemoryType::kPinned> mem{n};
  for (std::size_t i{0}; i < n; ++i) {
    mem.data()[i] = static_cast<float>(i) * 1.5f;
  }
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_FLOAT_EQ(mem.data()[i], static_cast<float>(i) * 1.5f);
  }
}

TEST(MemoryTest, PinnedMemoryFill) {
  constexpr std::size_t n{4};
  Memory<float, MemoryType::kPinned> mem{n};
  mem.fill(2.718f);
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_FLOAT_EQ(mem.data()[i], 2.718f);
  }
}

TEST(MemoryTest, DeviceMemoryBasic) {
  constexpr std::size_t n{4};
  Memory<int, MemoryType::kDevice> mem{n};
  int host[n]{1, 2, 3, 4};
  // Copy data from host to device
  CUDA_CHECK(
      cudaMemcpy(mem.data(), host, n * sizeof(int), cudaMemcpyHostToDevice));
  // Zero out host, then copy back from device
  for (std::size_t i{0}; i < n; ++i) host[i] = 0;
  CUDA_CHECK(
      cudaMemcpy(host, mem.data(), n * sizeof(int), cudaMemcpyDeviceToHost));
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(host[i], static_cast<int>(i + 1));
  }
}

TEST(MemoryTest, DeviceMemoryFill) {
  constexpr std::size_t n{4};
  Memory<float, MemoryType::kDevice> mem{n};
  mem.fill(1.618f);
  float host[n]{0};
  CUDA_CHECK(
      cudaMemcpy(host, mem.data(), n * sizeof(float), cudaMemcpyDeviceToHost));
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_FLOAT_EQ(host[i], 1.618f);
  }
}
