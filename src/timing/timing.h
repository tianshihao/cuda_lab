#pragma once

#include <cuda_runtime.h>

namespace cuda_lab::timing {
void VectorAdd(int const* h_a, int const* h_b, int* h_c, unsigned int N);
void VectorAdd(cudaStream_t stream, int const* h_a, int const* h_b, int* h_c,
               unsigned int N);
void CheckCudaError(cudaError_t const err, char const* msg);
}  // namespace cuda_lab::timing