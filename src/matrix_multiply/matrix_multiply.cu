#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#include "matrix.h"
#include "matrix_multiply.h"

namespace cuda_lab::matrix_multiply {

__device__ float GetElement(Matrix const m, std::size_t const row,
                            std::size_t const col) {
  return m.elements[row * m.stride + col];
}

__device__ float SetElement(Matrix const m, std::size_t const row,
                            std::size_t const col, float value) {
  m.elements[row * m.stride + col] = value;
}

// Get the kBlockSize * kBlockSize sub-matrix m_sub of m that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of m
// ! DO NOT SUPPORT PARTIAL BLOCKS!!!
__device__ Matrix GetSubMatrix(Matrix const m, std::size_t const row,
                               std::size_t const col) {
  Matrix m_sub;
  m_sub.height = kBlockSize;
  m_sub.width = kBlockSize;
  m_sub.stride = m.stride;
  m_sub.elements = &m.elements[m.stride * kBlockSize * row + kBlockSize * col];

  return m_sub;
}

__global__ void SharedABMultiply(Matrix const a, Matrix const b, Matrix c) {
  auto row_in_c{blockIdx.y * kBlockSize + threadIdx.y};
  auto col_in_c{blockIdx.x * kBlockSize + threadIdx.x};

  if (row_in_c >= c.height && col_in_c >= c.width) {
    return;
  }

  auto block_row{blockIdx.y};
  auto block_col{blockIdx.x};

  // Each thread block computes one sub-matrix of c
  auto c_sub{GetSubMatrix(c, block_row, block_col)};

  // Each thread computes one element of c_sub by accumulating results into
  // c_value
  auto c_value{0.0f};

  // Thread row and within the c_sub
  auto row_in_block{threadIdx.y};
  auto col_in_block{threadIdx.x};

  // Loop over all the sub-matrices of A and B that are
  // required to compute Csub
  // Multiply each pair of sub-matrices together
  // and accumulate the results
  for (std::size_t m{0}; m < DivUp(a.width, kBlockSize); ++m) {
    auto a_sub{GetSubMatrix(a, block_row, m)};
    auto b_sub{GetSubMatrix(b, m, block_col)};

    __shared__ float a_tile[kBlockSize][kBlockSize];
    __shared__ float b_tile[kBlockSize][kBlockSize];

    a_tile[row_in_block][col_in_block] =
        GetElement(a_sub, row_in_block, col_in_block);
    b_tile[row_in_block][col_in_block] =
        GetElement(b_sub, row_in_block, col_in_block);

    __syncthreads();  // Synchronize to make sure the tile is loaded

    for (std::size_t e{0}; e < kBlockSize; ++e) {
      c_value += a_tile[row_in_block][e] * b_tile[e][col_in_block];
    }

    __syncthreads();  // Synchronize to make sure the tile is loaded
  }

  SetElement(c_sub, row_in_block, col_in_block, c_value);
}

__global__ void CoalescedMultiply(Matrix const a, Matrix const b, Matrix c) {
  auto row_in_c{blockIdx.y * kBlockSize + threadIdx.y};
  auto col_in_c{blockIdx.x * kBlockSize + threadIdx.x};

  // Skip the threads that are outside the bounds of c
  if (row_in_c >= c.height && col_in_c >= c.width) {
    return;
  }

  auto block_row{blockIdx.y};
  auto block_col{blockIdx.x};
  auto c_sub{GetSubMatrix(c, block_row, block_col)};
  auto c_value{0.0f};

  auto row_in_block{threadIdx.y};
  auto col_in_block{threadIdx.x};

  // auto flag{row_in_c == 1 && true};

  // Slide the window to load a_tile
  // window size is equal to block size
  auto window_num{DivUp(a.width, kBlockSize)};
  // if (flag) {
  //   printf("window_num: %u, a.width: %u, c.width: %u, block size: %u\n",
  //          window_num, a.width, c.width, kBlockSize);
  // }
  for (std::size_t window_idx{0}; window_idx < window_num; ++window_idx) {
    // Move a_sub to next slide window
    auto a_sub{GetSubMatrix(a, block_row, window_idx)};

    // Share memory on chip
    __shared__ float a_tile[kBlockSize][kBlockSize];

    // Recalculate col_in_c for each sliding window
    // tmp += window_idx * kBlockSize;
    // printf("tmp: %u\n", tmp);

    // Load a_tile from global memory to shared memory
    // auto new_global_col{col_in_block + window_idx * kBlockSize};
    // if (flag && window_idx == 1) {
    //   printf("new_global_col: %u, window_idx: %u\n", new_global_col,
    //          window_idx);
    //   // printf(
    //   //     "new_global_col: %d, col_in_block: %d, window_idx: %d, a.width: %d\n",
    //   //     new_global_col, col_in_block, window_idx, a.width);
    //   // printf("window_idx: %d\n", window_idx);
    // }
    // if (new_global_col < a.width) {
      // Load a_tile from a_sub
      a_tile[row_in_block][col_in_block] =
          GetElement(a_sub, row_in_block, col_in_block);
    // } else {
      // If the last window is incompleted
      // a_tile[row_in_block][col_in_block] = 0.0f;
      // return;
    // }

    // Synchronize to make sure the tile is loaded
    __syncthreads();

    for (std::size_t e{0}; e < kBlockSize; ++e) {
      c_value += a_tile[row_in_block][e] * GetElement(b, e, col_in_c);
    }

    __syncthreads();
  }

  SetElement(c_sub, row_in_block, col_in_block, c_value);
}

__global__ void SimpleMultiply(Matrix const a, Matrix const b, Matrix c) {
  auto row{blockIdx.y * blockDim.y + threadIdx.y};
  auto col{blockIdx.x * blockDim.x + threadIdx.x};

  if (row >= a.height || col >= b.width) {
    return;
  }

  auto sum{0.0f};
  for (std::size_t i{0}; i < a.width; ++i) {
    sum += GetElement(a, row, i) * GetElement(b, i, col);
  }
  SetElement(c, row, col, sum);
}

void MatrixMultiply(Matrix const& h_a, Matrix const& h_b, Matrix& h_c,
                    MatrixMultiplyType const type) {
  Matrix d_a;
  d_a.width = d_a.stride = h_a.width;
  d_a.height = h_a.height;
  std::size_t size_a{h_a.width * h_a.height * sizeof(float)};
  cudaError_t err{cudaMalloc(&d_a.elements, size_a)};
  cudaMemcpy(d_a.elements, h_a.elements, size_a, cudaMemcpyHostToDevice);

  Matrix d_b;
  d_b.width = d_b.stride = h_b.width;
  d_b.height = h_b.height;
  std::size_t size_b{h_b.width * h_b.height * sizeof(float)};
  err = cudaMalloc(&d_b.elements, size_b);
  cudaMemcpy(d_b.elements, h_b.elements, size_b, cudaMemcpyHostToDevice);

  Matrix d_c;
  d_c.width = d_c.stride = h_c.width;
  d_c.height = h_c.height;
  std::size_t size_c{h_c.width * h_c.height * sizeof(float)};
  err = cudaMalloc(&d_c.elements, size_c);

  dim3 block_size(kBlockSize, kBlockSize);
  dim3 grid_size(DivUp(h_b.width, block_size.x),
                 DivUp(h_a.height, block_size.y));

  // Print kernel info
  std::cout << "Kernel info: " << std::endl;
  std::cout << "Data size: " << "row: " << h_a.height << " x col: " << h_b.width
            << std::endl;
  std::cout << "Element number: " << h_a.height * h_b.width << std::endl;
  std::cout << "Block size: " << block_size.x << " x " << block_size.y
            << std::endl;
  std::cout << "Grid size: " << grid_size.x << " x " << grid_size.y << std::endl
            << std::endl;

  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Record the start event
  cudaEventRecord(start, 0);

  // Calling the kernel
  switch (type) {
    case MatrixMultiplyType::kSimple: {
      std::cout << "SimpleMultiply is called" << std::endl << std::endl;
      SimpleMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c);
      break;
    }
    case MatrixMultiplyType::kCoalesced: {
      std::cout << "CoalescedMultiply is called" << std::endl << std::endl;
      CoalescedMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c);
      break;
    }
    case MatrixMultiplyType::kSharedAB: {
      std::cout << "SharedABMultiply is called" << std::endl << std::endl;
      SharedABMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c);
      break;
    }
    default: {
      break;
    }
  }

  // Record the stop event
  cudaEventRecord(stop, 0);

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  // Calculate the elapsed time
  float elapsed_time_ms;
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  float elapsed_time_s{elapsed_time_ms / 1000.0f};

  std::cout << "Kernel execution time: " << elapsed_time_ms << " ms"
            << std::endl;

  // Calculate bandwidth
  float total_data_size_gb{(size_a + size_b + size_c) /
                           (1024.0f * 1024.0f * 1024.0f)};
  float bandwidth_gb_per_s{total_data_size_gb / elapsed_time_s};

  std::cout << "Total data size: " << total_data_size_gb << " GB" << std::endl;
  std::cout << "Bandwidth: " << bandwidth_gb_per_s << " GB/s" << std::endl
            << std::endl;

  // Destroy CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaStreamSynchronize(0);  // Wait for the kernel to finish
  // Check for kernel errors
  err = cudaGetLastError();
  CheckCudaError(err, "Kernel execution failed");

  // Transfer data from device to host
  err = cudaMemcpy(h_c.elements, d_c.elements, size_c, cudaMemcpyDeviceToHost);
  CheckCudaError(err, "Failed to copy d_c to h_c");

  // Free device memory
  cudaFree(d_a.elements);
  cudaFree(d_b.elements);
  cudaFree(d_c.elements);
}

}  // namespace cuda_lab::matrix_multiply