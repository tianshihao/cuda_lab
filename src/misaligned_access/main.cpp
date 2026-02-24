#include "misaligned_access.h"

int main() {
  using namespace cuda_lab::misaligned_access;
  DumpOffsetsToFile(TestMisalignedAccess(), "misaligned_access_results.csv");

  return 0;
}
