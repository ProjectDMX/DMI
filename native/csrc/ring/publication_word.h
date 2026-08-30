// ring/publication_word.h -- shared task publication word encoding.

#pragma once

#include <cstdint>

namespace ring {

static constexpr uint64_t PUBLICATION_READY = uint64_t{1} << 63;
static constexpr uint64_t PUBLICATION_SIZE_MASK = PUBLICATION_READY - 1;

constexpr uint64_t encode_publication(uint64_t actual_bytes) {
    return PUBLICATION_READY | actual_bytes;
}

}  // namespace ring
