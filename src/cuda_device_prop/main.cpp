#include <cuda_runtime.h>

#include <iostream>
int main() {
  int dev{0};
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);

  std::cout << "================ CUDA Device Properties ================\n";
  std::cout << "General:\n";
  std::cout << "  Device: " << prop.name << "\n";
  std::cout << "  Compute Capability: " << prop.major << "." << prop.minor
            << "\n";
  std::cout << "  Number of SMs (multiprocessors): " << prop.multiProcessorCount
            << "\n";
  std::cout << "  " << (prop.integrated ? "Integrated GPU" : "Discrete GPU")
            << "\n";
  std::cout << "  Unified Addressing (UVA) support: "
            << (prop.unifiedAddressing ? "Yes" : "No") << "\n\n";

  std::cout << "Memory Hierarchy:\n";
  std::cout << "  Global memory size: "
            << static_cast<double>(prop.totalGlobalMem) / (1024 * 1024 * 1024)
            << " GB (" << prop.totalGlobalMem << " bytes)\n";
  std::cout << "  Shared memory per SM: "
            << prop.sharedMemPerMultiprocessor / 1024 << " KB ("
            << prop.sharedMemPerMultiprocessor << " bytes)\n";
  std::cout << "  L1 cache (shared memory, proxy) per SM: "
            << prop.sharedMemPerMultiprocessor / 1024 << " KB ("
            << prop.sharedMemPerMultiprocessor << " bytes)\n";
  std::cout << "  Total L1 cache (shared memory, proxy) (all SMs): "
            << (prop.sharedMemPerMultiprocessor * prop.multiProcessorCount) /
                   (1024 * 1024)
            << " MB ("
            << (prop.sharedMemPerMultiprocessor * prop.multiProcessorCount)
            << " bytes)\n";
  std::cout << "  Total shared memory (all SMs): "
            << (prop.sharedMemPerMultiprocessor * prop.multiProcessorCount) /
                   (1024 * 1024)
            << " MB ("
            << (prop.sharedMemPerMultiprocessor * prop.multiProcessorCount)
            << " bytes)\n";
  std::cout << "  L2 cache size: " << prop.l2CacheSize / (1024 * 1024)
            << " MB\n";
  std::cout << "  Max persisting L2 cache size: "
            << prop.persistingL2CacheMaxSize / (1024 * 1024) << " MB\n\n";

  std::cout << "DMA Engines:\n";
  switch (prop.asyncEngineCount) {
    case 2:
      std::cout << "  2 copy engines: bidirectional memory copies concurrent "
                   "with kernel execution\n";
      break;
    case 1:
      std::cout << "  1 copy engine: unidirectional memory copies concurrent "
                   "with kernel execution\n";
      break;
    case 0:
      std::cout << "  0 copy engines: cannot perform concurrent memory copies "
                   "with kernel execution\n";
      break;
    default:
      std::cout << "  Unknown copy engine configuration\n";
      break;
  }
  std::cout << "=======================================================\n";
  return 0;
}
