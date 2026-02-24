#include <cuda_runtime.h>

#include "misaligned_access.h"

namespace cuda_lab::misaligned_access {
__global__ void kernel_offset_copy(float* odata, float* idata, int offset,
                                   int n) {
  auto idx{blockIdx.x * blockDim.x + threadIdx.x + offset};
  if (idx < n) {
    odata[idx] = idata[idx];
  }
}

void KernelMisalignedAccess(float* device_odata, float* device_idata,
                            int offset, int n) {
  int block_size{256};
  int grid_size{(n + block_size - 1) / block_size};
  kernel_offset_copy<<<grid_size, block_size>>>(device_odata, device_idata,
                                                offset, n);
}
}  // namespace cuda_lab::misaligned_access
