#include "async_overlap.h"

int main() {
  // Attention, the cost of both methods is the same, analyze the bottleneck
  // using Nsight when have time.
  cuda_lab::async_overlap::SequentialCopyAndExecute();
  cuda_lab::async_overlap::StagedCopyAndExecute();

  return 0;
}
