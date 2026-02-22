#include "engine_utils.h"

#include <algorithm>

namespace monitoring {

double clamp_ratio(double value) {
  if (value <= 0.0) return 0.8;
  if (value > 1.0) return 1.0;
  return value;
}

int64_t mb_to_bytes(int64_t mb) {
  if (mb <= 0) return 0;
  constexpr int64_t kMb = 1024ll * 1024ll;
  return mb * kMb;
}

int64_t tensor_nbytes(const at::Tensor& tensor) {
  if (!tensor.defined()) return 0;
  return static_cast<int64_t>(tensor.nbytes());
}

}  // namespace monitoring
