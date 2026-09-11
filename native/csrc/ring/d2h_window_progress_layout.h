#pragma once

#include <cstdint>
#include <limits>

namespace ring {

struct D2HWindowPackedProgressLayout {
    using Word = uint64_t;
    using Version = uint16_t;
    using Counter = uint64_t;

    static constexpr unsigned kVersionBits = std::numeric_limits<Version>::digits;
    static constexpr unsigned kWordBits = std::numeric_limits<Word>::digits;
    static constexpr unsigned kCounterBits = kWordBits - kVersionBits;
    static constexpr unsigned kVersionShift = kCounterBits;
    static constexpr Word kCounterLimit = Word{1} << kCounterBits;
    static constexpr Word kCounterMask = kCounterLimit - 1;
    static constexpr Version kNoPatternVersion = 0;
    static constexpr Version kFirstPatternVersion = 1;
    static constexpr Version kMaxPatternVersion = std::numeric_limits<Version>::max();

#ifdef __CUDACC__
    __host__ __device__
#endif
        static constexpr Word pack(Version version, Counter counter) {
        return (Word{version} << kVersionShift) | counter;
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
        static constexpr Version version(Word packed) {
        return static_cast<Version>(packed >> kVersionShift);
    }

#ifdef __CUDACC__
    __host__ __device__
#endif
        static constexpr Counter counter(Word packed) {
        return packed & kCounterMask;
    }
};

static_assert(D2HWindowPackedProgressLayout::kVersionBits +
                  D2HWindowPackedProgressLayout::kCounterBits ==
              D2HWindowPackedProgressLayout::kWordBits);

}  // namespace ring
