#pragma once

namespace cuda_lab {
enum class MemoryType { kHost = 0, kPinned, kDevice, kMappedPinned };
}
