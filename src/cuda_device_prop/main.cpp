#include <cuda_runtime.h>

#include <iostream>
int main() {
  int dev{0};
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);

  // Check if the device is integrated or discrete
  if (prop.integrated) {
    std::cout << "This is an integrated GPU." << std::endl;
  } else {
    std::cout << "This is a discrete GPU." << std::endl;
  }

  // Check copy engine number and output it.
  switch (prop.asyncEngineCount) {
    case 2:
      std::cout
          << "This GPU has 2 copy engines, which means it can perform "
             "bidirectional memory copies concurrently with kernel execution."
          << std::endl;
      break;
    case 1:
      std::cout
          << "This GPU has 1 copy engine, which means it can perform "
             "unidirectional memory copies concurrently with kernel execution."
          << std::endl;
      break;
    case 0:
      std::cout << "This GPU has no copy engines, which means it cannot "
                   "perform concurrent memory copies with kernel execution."
                << std::endl;

    default:
      break;
  }

  std::cout << "Device: " << prop.name << ", Compute Capability: " << prop.major
            << "." << prop.minor << '\n'
            << "Unified Addressing (UVA) support: "
            << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;

  return 0;
}
