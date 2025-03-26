#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#include "matrix.h"
#include "matrix_multiply.h"

namespace cuda_lab::matrix_multiply {

__device__ float GetElement(Matrix const m, std::size_t const row,
                            std::size_t const col) {
  // return m.elements[row * m.stride + col];
  return m.elements[row * m.stride + col];
}

__device__ float SetElement(Matrix const m, std::size_t const row,
                            std::size_t const col, float value) {
  m.elements[row * m.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix const m, std::size_t const row,
                               std::size_t const col) {
  Matrix a_sub;
  a_sub.height = kBlockSize;
  a_sub.width = kBlockSize;
  a_sub.stride = m.stride;
  a_sub.elements = &m.elements[m.stride * kBlockSize * row + kBlockSize * col];

  return a_sub;
}
// __global__ void SharedABMultiply(Matrix const a, Matrix const b, Matrix c) {
//   auto block_row{blockIdx.x};
//   auto block_col{blockIdx.y};

//   // Each thread block computes one sub-matrix of c
//   auto c_sub{GetSubMatrix(c, block_row, block_col)};

//   // Each thread computes one element of c_sub by accumulating results into
//   // c_value
//   auto c_value{0.0f};

//   // Thread row and within the c_sub
//   auto row{threadIdx.y};
//   auto col{threadIdx.x};

//   for (std::size_t m{0}; m < a.width / BLOCK_SIZE; ++m) {
//     auto a_sub{GetSubMatrix(a, block_row, m)};
//     auto b_sub{GetSubMatrix(b, m, block_col)};

//     __shared__ float a_tile[BLOCK_SIZE]
//                            [BLOCK_SIZE];  // Shared memory for a sub-matrix
//     __shared__ float b_tile[BLOCK_SIZE]
//                            [BLOCK_SIZE];  // Shared memory for b sub-matrix

//     a_tile[row][col] = GetElement(a_sub, row, col);
//     b_tile[row][col] = GetElement(b_sub, row, col);

//     __syncthreads();  // Synchronize to make sure the tile is loaded

//     for (std::size_t e{0}; e < BLOCK_SIZE; ++e) {
//       c_value += a_tile[row][e] * b_tile[e][col];
//     }

//     __syncthreads();  // Synchronize to make sure the tile is loaded
//   }

//   SetElement(c_sub, row, col, c_value);  // Write the result to c
// }

__global__ void CoalescedMultiply(Matrix const a, Matrix const b, Matrix c) {
  // Share memory on chip
  __shared__ float a_tile[kBlockSize][kBlockSize];

  auto row{blockIdx.y * blockDim.y + threadIdx.y};
  auto col{blockIdx.x * blockDim.x + threadIdx.x};

  if (row >= a.height || col >= b.width) {
    return;
  }

  // Load a_tile from global memory to shared memory
  a_tile[threadIdx.y][threadIdx.x] = a.elements[row * a.width + threadIdx.x];

  // Synchronize to make sure the tile is loaded
  __syncthreads();

  auto sum{0.0f};

  // Perform the computation
  for (std::size_t i{0}; i < kBlockSize; ++i) {
    sum += a_tile[threadIdx.y][i] * b.elements[i * b.width + col];
  }

  c.elements[row * b.width + col] = sum;
}

__global__ void SimpleMultiply(Matrix const a, Matrix const b, Matrix c) {
  auto row{blockIdx.y * blockDim.y + threadIdx.y};
  auto col{blockIdx.x * blockDim.x + threadIdx.x};

  if (row >= a.height || col >= b.width) {
    return;
  }

  // Output sample kernel info
  if (row == a.height / 2 && col == b.width / 2) {
    printf("Sample kernel info:\n");
    printf(
        "Block (%d, %d), Thread (%d, %d)\nRow = blockIdx.y * blockDim.y + "
        "threadIdx.y = %d * %d + %d = %d\nCol = blockIdx.x * blockDim.x + "
        "threadIdx.x = %d * %d + %d = %d\n\n",
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockIdx.y,
        blockDim.y, threadIdx.y, row, blockIdx.x, blockIdx.y, threadIdx.x, col);
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
  std::size_t size{h_a.width * h_a.height * sizeof(float)};
  cudaError_t err{cudaMalloc(&d_a.elements, size)};
  cudaMemcpy(d_a.elements, h_a.elements, size, cudaMemcpyHostToDevice);

  Matrix d_b;
  d_b.width = d_b.stride = h_b.width;
  d_b.height = h_b.height;
  size = h_b.width * h_b.height * sizeof(float);
  err = cudaMalloc(&d_b.elements, size);
  cudaMemcpy(d_b.elements, h_b.elements, size, cudaMemcpyHostToDevice);

  Matrix d_c;
  d_c.width = d_c.stride = h_c.width;
  d_c.height = h_c.height;
  size = h_c.width * h_c.height * sizeof(float);
  err = cudaMalloc(&d_c.elements, size);

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
      SimpleMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c);
      break;
    }
    case MatrixMultiplyType::kCoalesced: {
      CoalescedMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c);
      break;
    }
    case MatrixMultiplyType::kSharedAB: {
      // SharedABMultiply<<<grid_size, block_size>>>(d_a, d_b, d_c);
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

  switch (type) {
    case MatrixMultiplyType::kSimple:
      std::cout << "SimpleMultiply executed." << std::endl;
      break;

    case MatrixMultiplyType::kCoalesced:
      std::cout << "CoalescedMultiply executed." << std::endl;
      break;

    default:
      break;
  }

  // Calculate the elapsed time
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);
  std::cout << "Kernel execution time: " << elapsed_time << " ms" << std::endl;

  // Calculate bandwidth
  // A and B matrices
  float data_size_gb{2.0f * h_a.height * h_a.stride * sizeof(float) /
                     (1024.0f * 1024.0f * 1024.0f)};
  data_size_gb += h_a.height * h_a.stride * sizeof(float) /
                  (1024.0f * 1024.0f * 1024.0f);             // C matrix
  float bandwidth{data_size_gb / (elapsed_time / 1000.0f)};  // Convert ms to s
  std::cout << "Kernel bandwidth: " << bandwidth << " GB/s" << std::endl
            << std::endl;

  // Destroy CUDA events
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaStreamSynchronize(0);  // Wait for the kernel to finish
  // Check for kernel errors
  err = cudaGetLastError();
  CheckCudaError(err, "Kernel execution failed");

  // Transfer data from device to host
  err = cudaMemcpy(h_c.elements, d_c.elements, size, cudaMemcpyDeviceToHost);
  CheckCudaError(err, "Failed to copy d_c to h_c");

  // // Free device memory
  cudaFree(d_a.elements);
  cudaFree(d_b.elements);
  cudaFree(d_c.elements);
}

}  // namespace cuda_lab::matrix_multiply