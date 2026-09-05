#include "d2h_window_pattern.h"
#include "d2h_window_progress_layout.h"

#include <stdexcept>

namespace ring {

D2HWindowPatternMatcher::D2HWindowPatternMatcher(uint64_t period,
                                                 std::vector<D2HWindowOffset> windows)
    : period_(period), windows_(std::move(windows)) {
    if (period_ == 0 || period_ >= D2HWindowPackedProgressLayout::kCounterLimit) {
        throw std::invalid_argument(
            "D2H window period must be in the packed counter domain");
    }
    if (windows_.empty()) {
        throw std::invalid_argument("D2H window pattern must not be empty");
    }
    uint64_t prior_end = 0;
    for (size_t index = 0; index < windows_.size(); ++index) {
        const auto& window = windows_[index];
        if (window.begin >= window.end || window.end > period_) {
            throw std::invalid_argument(
                "D2H windows must satisfy begin < end <= period");
        }
        if (index != 0 && window.begin < prior_end) {
            throw std::invalid_argument(
                "D2H windows must be sorted and non-overlapping");
        }
        prior_end = window.end;
    }
}

std::optional<D2HWindowOccurrence>
D2HWindowPatternMatcher::match(uint64_t counter) const noexcept {
    const uint64_t occurrence = counter / period_;
    const uint64_t phase = counter % period_;
    for (uint64_t index = 0; index < windows_.size(); ++index) {
        const auto& window = windows_[index];
        if (phase < window.begin)
            break;
        if (phase < window.end) {
            const uint64_t base = occurrence * period_;
            return D2HWindowOccurrence{
                index,
                occurrence,
                base + window.begin,
                base + window.end,
            };
        }
    }
    return std::nullopt;
}

}  // namespace ring
