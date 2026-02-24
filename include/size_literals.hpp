#pragma once

#include <cstddef>

namespace cuda_lab {

// User-defined literals for readable buffer size specification.
// Usage: 3_MB, 4_GB, etc.
constexpr std::size_t operator"" _KB(unsigned long long value) {
  return static_cast<std::size_t>(value) * 1024ULL;
}

constexpr std::size_t operator"" _MB(unsigned long long value) {
  return static_cast<std::size_t>(value) * 1024_KB;
}

constexpr std::size_t operator"" _GB(unsigned long long value) {
  return static_cast<std::size_t>(value) * 1024_MB;
}

constexpr std::size_t operator"" _k(unsigned long long value) {
  return static_cast<std::size_t>(value) * 1000ULL;
}
constexpr std::size_t operator"" _M(unsigned long long value) {
  return static_cast<std::size_t>(value) * 1000000ULL;
}
constexpr std::size_t operator"" _B(unsigned long long value) {
  return static_cast<std::size_t>(value) * 1000000000ULL;
}

}  // namespace cuda_lab
