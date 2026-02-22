#ifndef MONITORING_ENGINE_UTILS_H_
#define MONITORING_ENGINE_UTILS_H_

#include <ATen/Tensor.h>

#include <cstdint>

namespace monitoring {

double clamp_ratio(double value);
int64_t mb_to_bytes(int64_t mb);
int64_t tensor_nbytes(const at::Tensor& tensor);

}  // namespace monitoring

#endif  // MONITORING_ENGINE_UTILS_H_
