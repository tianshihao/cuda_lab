#include "async_overlap.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "memory.hpp"
#include "scoped_timer.hpp"

namespace cuda_lab::async_overlap {
void SequentialCopyAndExecute() {
  Memory<float, MemoryType::kPinned> a_h(kDataSize, 1.0f);
  Memory<float, MemoryType::kDevice> a_d(kDataSize, 0.0f);

  {
    SCOPED_TIMER("SequentialCopyAndExecute");
    cudaMemcpy(a_d.data(), a_h.data(), a_h.bytes(), cudaMemcpyHostToDevice);
    LaunchAddOne(a_d.data(), a_d.size());
  }

  cudaMemcpy(a_h.data(), a_d.data(), a_d.bytes(), cudaMemcpyDeviceToHost);

  std::cout << "Result: ";
  std::cout << a_h.data()[0] << std::endl;
}

void StagedCopyAndExecute() {
  Memory<float, MemoryType::kPinned> a_h(kDataSize, 1.0f);
  Memory<float, MemoryType::kDevice> a_d(kDataSize, 0.0f);

  constexpr std::size_t kStreamNum{100};
  std::vector<cudaStream_t> streams(kStreamNum);
  for (auto& stream : streams) {
    cudaStreamCreate(&stream);
  }

  {
    SCOPED_TIMER("StagedCopyAndExecute");

    std::size_t chunk_size{a_h.size() / kStreamNum};
    std::size_t remainder{a_h.size() % kStreamNum};

    for (std::size_t i{0}; i < kStreamNum; ++i) {
      std::size_t offset{i * chunk_size};
      std::size_t this_chunk{chunk_size};
      if (i == kStreamNum - 1) {
        this_chunk += remainder;  // last chunk gets remainder
      }
      std::size_t this_bytes = this_chunk * sizeof(float);
      cudaMemcpyAsync(a_d.slice(offset), a_h.slice(offset), this_bytes,
                      cudaMemcpyHostToDevice, streams[i]);
      LaunchAddOne(a_d.slice(offset), this_chunk, streams[i]);
    }

    for (auto& stream : streams) {
      cudaStreamSynchronize(stream);
    }
  }

  cudaMemcpy(a_h.data(), a_d.data(), a_d.bytes(), cudaMemcpyDeviceToHost);

  std::cout << "Result: ";
  std::cout << a_h.data()[0] << std::endl;
}

}  // namespace cuda_lab::async_overlap
