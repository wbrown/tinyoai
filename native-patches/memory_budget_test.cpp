#include "memory_budget.h"
#include <cassert>
#include <cstdint>
#include <iostream>
using namespace mlx::core::metal::headroom;
int main() {
  constexpr uint64_t GiB = 1ULL << 30;
  for (const uint64_t gib : {8, 32, 512}) {
    auto total = gib * GiB;
    assert(budget(total, total, 0) == total - reserve(total));
    assert(budget(total, total / 2, total / 4) == total * 3 / 4 - reserve(total));
  }
  assert(reserve(101) == 11);
  assert(budget(1000, 300, 400) == 600);
  // Reusing an idle buffer does not increase the overall allocation budget.
  assert(excess(300, 400, 0, 600) == 100);
  assert(excess(500, 200, 0, 600) == 100);
  // Allocation releases only the idle excess; it never discards live tensors.
  assert(excess(300, 400, 150, 600) == 250);
  assert(excess(900, 20, 500, 600) == 20);
  // External memory pressure shrinks the budget, even below current live use.
  assert(budget(1000, 20, 50) == 0);
  assert(budget(1000, 20, 500) == 420);
  assert(budget(1000, UINT64_MAX, UINT64_MAX) == 900);
  assert(excess(UINT64_MAX, 20, 500, 600) == 20);
  std::cout << "memory headroom arithmetic: pass\n";
}
