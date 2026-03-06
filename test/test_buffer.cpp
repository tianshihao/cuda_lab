#include <gtest/gtest.h>

#include "buffer.hpp"
#include "cuda_check.hpp"

namespace cuda_lab::operators {
template <typename T>
void add_one_kernel_launcher(T* ptr, std::size_t n);
}

TEST(BufferTest, CreateHostBuffer) {
  constexpr std::size_t n{10};
  cuda_lab::Buffer<int, cuda_lab::MemoryType::kHost> buf(n);
  EXPECT_EQ(buf.size(), n);
  EXPECT_NE(buf.host_ptr(), nullptr);
}

TEST(BufferTest, DevicePtr) {
  constexpr std::size_t n{10};
  cuda_lab::Buffer<int, cuda_lab::MemoryType::kHost> buf(n);
  buf.sync_to_device();
  auto device_ptr{buf.device_ptr()};
  EXPECT_NE(device_ptr, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());
}

TEST(BufferTest, AddOneDeviceKernel) {
  constexpr std::size_t n{8};
  cuda_lab::Buffer<int, cuda_lab::MemoryType::kHost> buf(n);
  // 0, 1, 2, ..., n-1
  for (std::size_t i{0}; i < n; ++i) {
    buf.host_ptr()[i] = static_cast<int>(i);
  }

  // Sync to device
  auto device_ptr{buf.device_ptr()};
  CUDA_CHECK(cudaDeviceSynchronize());

  // Launch add_one kernel on device buffer
  // 1, 2, 3, ..., n
  cuda_lab::operators::add_one_kernel_launcher(device_ptr, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Sync back to host
  int const* host_ptr{buf.host_ptr()};
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(host_ptr[i], static_cast<int>(i + 1));
  }

  // 3, 4, 5, ..., n+2
  for (std::size_t i{0}; i < n; ++i) {
    buf.host_ptr()[i] += 2;
    EXPECT_EQ(buf.host_ptr()[i], static_cast<int>(i + 3));
  }

  device_ptr = buf.device_ptr();
  CUDA_CHECK(cudaDeviceSynchronize());
  // 4, 5, 6, ..., n+3
  cuda_lab::operators::add_one_kernel_launcher(device_ptr, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  host_ptr = buf.host_ptr();
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(host_ptr[i], static_cast<int>(i + 4));
  }
}

TEST(BufferTest, SetGetHostDevice) {
  constexpr std::size_t n{5};
  cuda_lab::Buffer<int, cuda_lab::MemoryType::kHost> buf(n);

  // Set values on host
  for (std::size_t i{0}; i < n; ++i) {
    buf.set_host(i, static_cast<int>(i * 10));
  }
  // Get values from host
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(buf.get_host(i), static_cast<int>(i * 10));
  }

  // (Optional) Launch a device kernel to set device values, if implemented
  // buf.set_device(2, 42); // Uncomment if set_device is implemented
  // EXPECT_EQ(buf.get_device(2), 42); // Uncomment if get_device is implemented
}
