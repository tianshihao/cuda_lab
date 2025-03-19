#pragma once

#include <cuda_runtime.h>

namespace cuda_lab::matrix_multiply {

constexpr int TILE_DIM{32};

void CheckCudaError(cudaError_t const err, char const* msg);
void MatrixMultiply(float* h_a, float* h_b, float* h_c, int A_rows, int A_cols,
                    int B_cols);
void PrintMatrix(const float* matrix, int rows, int cols, int tile_size,
                 const char* name);

__global__ void SimpleMultiply(float* a, float* b, float* c, int A_rows,
                               int A_cols, int B_cols);
inline int DivUp(int a, int b) { return (a + b - 1) / b; }
}  // namespace cuda_lab::matrix_multiply