#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace cuda_lab {

// Returns the repository root path as a string_view (set by CMake macro
// REPO_ROOT_PATH).
constexpr std::string_view GetSourceDir() { return REPO_ROOT_PATH; }

// Returns the full path by joining the repo root and the given filename, using
// platform-correct separators.
inline std::string JoinRepoRootWith(std::string const& filename) {
  auto const path{std::filesystem::path{REPO_ROOT_PATH} / filename};
  return path.string();
}

}  // namespace cuda_lab
