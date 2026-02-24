#pragma once

#include <array>
#include <string>

#include "size_literals.hpp"

namespace cuda_lab::misaligned_access {
static constexpr int kDataSize{10_MB};
static constexpr int kOffset{32};
static constexpr int kArraySize{kOffset + 1};

void KernelMisalignedAccess(float* device_odata, float* device_idata,
                            int offset, int n);

std::array<float, kArraySize> TestMisalignedAccess();

void DumpOffsetsToFile(std::array<float, kArraySize> const& offsets,
                       std::string const& filename);
}  // namespace cuda_lab::misaligned_access
