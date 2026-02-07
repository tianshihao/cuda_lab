#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace cuda_lab::async_overlap {

static constexpr std::size_t kDataSize{(1 << 20) * 100};

void SequentialCopyAndExecute();
void StagedCopyAndExecute();

void LaunchAddOne(float* device_ptr, std::size_t n,
                  cudaStream_t stream = nullptr);

}  // namespace cuda_lab::async_overlap
