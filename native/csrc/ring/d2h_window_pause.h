#pragma once

#include <cstdint>

namespace ring {

struct DrainPauseToken {
    uint64_t generation{0};
};

class DrainPauseControl {
  public:
    virtual ~DrainPauseControl() = default;
    virtual DrainPauseToken pause_after_flush_and_wait() = 0;
    virtual void resume(DrainPauseToken token) = 0;
};

}  // namespace ring
