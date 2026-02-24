#include "misaligned_access.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <string>

#include "cuda_scoped_timer.hpp"
#include "file_utils.hpp"
#include "memory.hpp"

namespace cuda_lab::misaligned_access {

std::array<float, kArraySize> TestMisalignedAccess() {
  std::array<float, kArraySize> offsets{};

  std::for_each(offsets.begin(), offsets.end(), [i = 0](float& ms) mutable {
    CudaScopedTimer timer;
    Memory<float, MemoryType::kDevice> device_idata(kDataSize, 1.0f);
    Memory<float, MemoryType::kDevice> device_odata(kDataSize, 0.0f);

    KernelMisalignedAccess(device_odata.data(), device_idata.data(), i,
                           kDataSize);

    ms = timer.finish();

    std::cout << "Misaligned access with offset " << i << " took " << ms
              << " ms." << std::endl;
    ++i;
  });

  return offsets;
}

void DumpOffsetsToFile(std::array<float, kArraySize> const& offsets,
                       std::string const& filename) {
  std::ofstream ofs(JoinRepoRootWith(filename));
  if (!ofs) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return;
  }
  for (std::size_t i{0}; i < offsets.size(); ++i) {
    ofs << i << "," << offsets[i] << '\n';
  }
}

}  // namespace cuda_lab::misaligned_access
