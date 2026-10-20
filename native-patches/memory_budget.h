// Native allocator headroom policy. The 10% physical-RAM reserve is an
// explicit user requirement (2026-10-17), not a workload/data limit.
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>

namespace mlx::core::metal::headroom {
inline uint64_t add(uint64_t a, uint64_t b) {
  return b > std::numeric_limits<uint64_t>::max() - a
      ? std::numeric_limits<uint64_t>::max() : a + b;
}
inline uint64_t reserve(uint64_t physical) {
  return physical / 10 + (physical % 10 != 0);
}
// available counts free + explicitly purgeable physical pages. Inactive
// anonymous pages may belong to owned buffers and must not be counted again.
// Speculative pages are already included in free.
inline uint64_t budget(uint64_t physical, uint64_t available, uint64_t owned) {
  const auto held = reserve(physical);
  const auto capacity = std::min(physical, add(owned, available));
  return capacity > held ? capacity - held : 0;
}
inline uint64_t excess(uint64_t active, uint64_t cached, uint64_t incoming,
                       uint64_t budget) {
  const auto wanted = add(add(active, cached), incoming);
  return wanted > budget ? std::min(cached, wanted - budget) : 0;
}
} // namespace mlx::core::metal::headroom
