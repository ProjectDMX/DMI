#pragma once

#include "d2h_window_config.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ring {

struct D2HWindowOccurrence {
    uint64_t window_index{0};
    uint64_t occurrence{0};
    uint64_t absolute_begin{0};
    uint64_t absolute_end{0};
};

class D2HWindowPatternMatcher {
  public:
    D2HWindowPatternMatcher(uint64_t period, std::vector<D2HWindowOffset> windows);

    std::optional<D2HWindowOccurrence> match(uint64_t counter) const noexcept;
    uint64_t period() const noexcept { return period_; }
    uint64_t window_count() const noexcept { return windows_.size(); }

  private:
    uint64_t period_{0};
    std::vector<D2HWindowOffset> windows_;
};

}  // namespace ring
