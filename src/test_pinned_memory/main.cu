#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>

#include "memory.hpp"

namespace cuda_lab {

// Kernel adds 42 to each element.
__global__ void test_kernel(int const* src, int* out, int const n) {
  int const tid{static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x)};
  if (tid < n) {
    out[tid] = src[tid] + 42;
  }
}

// Prints device properties and UVA support.
void CheckUva() {
  int const dev{0};
  cudaDeviceProp prop{};
  auto const err{cudaGetDeviceProperties(&prop, dev)};
  if (err != cudaSuccess) {
    std::cerr << "[Error] cudaGetDeviceProperties: " << cudaGetErrorString(err)
              << '\n';
    return;
  }
  std::cout << "[Check] Device: " << prop.name
            << ", Compute Capability: " << prop.major << "." << prop.minor
            << '\n'
            << "[Check] Unified Addressing (UVA) support: "
            << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
}

}  // namespace cuda_lab

int main() {
  using namespace cuda_lab;

  CheckUva();

  constexpr size_t n{8};

  // 1. Standard pinned memory
  Memory<int, MemoryType::kPinned> mem_pinned{n, 7};
  int const* host_addr_pinned{mem_pinned.data()};

  // Get device pointer for pinned memory
  int* dev_addr_pinned{nullptr};
  auto const err{cudaHostGetDevicePointer(
      &dev_addr_pinned, static_cast<void*>(const_cast<int*>(host_addr_pinned)),
      0)};

  std::cout << "[Pinned] Host pointer: "
            << static_cast<void const*>(host_addr_pinned) << "\n";
  if (err == cudaSuccess) {
    std::cout << "[Pinned] Device pointer: "
              << static_cast<void*>(dev_addr_pinned) << "\n";
  } else {
    std::cout << "[Pinned] Device pointer: <n/a, not mapped>\n";
  }

  // 2. Pinned Memory using cudaHostRegister (optional, may not be supported on
  // all platforms)
  int host_array[n]{7, 7, 7, 7, 7, 7, 7, 7};
  auto const reg_err{cudaHostRegister(host_array, sizeof(host_array),
                                      cudaHostRegisterDefault)};
  std::cout << "[HostRegister Pinned] Host array pointer: "
            << static_cast<void const*>(host_array) << "\n";
  int* dev_ptr_host_array{nullptr};
  if (reg_err == cudaSuccess) {
    if (cudaHostGetDevicePointer(&dev_ptr_host_array, host_array, 0) ==
        cudaSuccess) {
      std::cout << "[HostRegister Pinned] Device pointer: "
                << static_cast<void*>(dev_ptr_host_array) << "\n";
    } else {
      std::cout << "[HostRegister Pinned] Device pointer: <n/a, not mapped>\n";
    }
  } else {
    std::cout << "[HostRegister Pinned] Registration failed: "
              << cudaGetErrorString(reg_err) << "\n";
  }

  // 3. Mapped pinned memory
  Memory<int, MemoryType::kMappedPinned> mem_mapped{n, 42};
  int const* host_addr_mapped{mem_mapped.data()};
  int* dev_addr_mapped{mem_mapped.device_ptr()};

  std::cout << "[Mapped] Host pointer: "
            << static_cast<void const*>(host_addr_mapped) << "\n";
  std::cout << "[Mapped] Device pointer: "
            << static_cast<void*>(dev_addr_mapped) << "\n";

  // 4. Device memory output buffers
  Memory<int, MemoryType::kDevice> mem_out_pinned{n, 0};
  Memory<int, MemoryType::kDevice> mem_out_mapped{n, 0};

  // 5. Kernel validation
  // 5-1. Standard pinned: kernel access (may fail depending on UVA/platform)
  auto e1{cudaSuccess};
  test_kernel<<<1, static_cast<int>(n)>>>(
      dev_addr_pinned, mem_out_pinned.device_ptr(), static_cast<int>(n));
  e1 = cudaDeviceSynchronize();
  std::cout << "[Kernel access pinned] Status: " << cudaGetErrorString(e1)
            << std::endl;
  if (e1 == cudaSuccess) {
    int out[n]{};
    cudaMemcpy(out, mem_out_pinned.device_ptr(), n * sizeof(int),
               cudaMemcpyDeviceToHost);
    std::cout << "[Kernel output pinned]: ";
    for (auto const i : out) std::cout << i << ' ';
    std::cout << '\n';
  }

  // 5-2
  auto e2{cudaSuccess};
  test_kernel<<<1, static_cast<int>(n)>>>(
      dev_ptr_host_array, mem_out_pinned.device_ptr(), static_cast<int>(n));
  e2 = cudaDeviceSynchronize();
  std::cout << "[Kernel access HostRegister pinned] Status: "
            << cudaGetErrorString(e2) << std::endl;
  if (e2 == cudaSuccess) {
    int out[n]{};
    cudaMemcpy(out, mem_out_pinned.device_ptr(), n * sizeof(int),
               cudaMemcpyDeviceToHost);
    std::cout << "[Kernel output HostRegister pinned]: ";
    for (auto const i : out) std::cout << i << ' ';
    std::cout << '\n';
  }

  // 5-3. Mapped pinned: kernel direct access (usually supported)
  auto e3{cudaSuccess};
  test_kernel<<<1, static_cast<int>(n)>>>(
      dev_addr_mapped, mem_out_mapped.device_ptr(), static_cast<int>(n));
  e3 = cudaDeviceSynchronize();
  std::cout << "[Kernel access mapped] Status: " << cudaGetErrorString(e3)
            << std::endl;
  if (e3 == cudaSuccess) {
    int out[n]{};
    cudaMemcpy(out, mem_out_mapped.device_ptr(), n * sizeof(int),
               cudaMemcpyDeviceToHost);
    std::cout << "[Kernel output mapped]: ";
    for (auto const i : out) std::cout << i << ' ';
    std::cout << '\n';
  }

  return 0;
}
