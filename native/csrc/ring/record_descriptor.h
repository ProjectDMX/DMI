// Framework-neutral encoded record descriptors.
//
// Integrations encode semantic metadata into literal cells before publishing
// a payload.  The native record consumer only associates the next descriptor
// with the next physical payload and materializes the declared payload slices.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ring {

enum class PayloadMaterialization : uint8_t {
    TENSOR = 0,
    FLOAT_SCALAR = 1,
    INT_SCALAR = 2,
};

// Dtype is the numeric at::ScalarType value.  Keeping it encoded avoids an
// ATen dependency in the descriptor itself while allowing the consumer to
// validate and construct an at::Tensor.
struct PayloadSlice {
    uint64_t offset_bytes{0};

    // A missing length means "all actual payload bytes after offset".  This is
    // required for a dynamic producer whose exact byte count is device-known
    // only when its publication becomes ready. An explicit zero remains a real
    // zero-length slice.
    std::optional<uint64_t> length_bytes;

    PayloadMaterialization materialization{PayloadMaterialization::TENSOR};
    int32_t dtype{0};
    std::vector<int64_t> logical_shape;

    // -1 means every logical dimension is fixed.  Otherwise the selected
    // dimension is inferred from the actual slice byte count while all other
    // dimensions remain fixed.
    int32_t inferred_dynamic_dim{-1};
};

using EncodedRecordCell = std::variant<
    std::string,
    int32_t,
    int64_t,
    double,
    std::vector<int64_t>,
    PayloadSlice>;

struct EncodedRecordRow {
    std::vector<EncodedRecordCell> cells;
};

struct RecordDescriptor {
    std::string layout;
    std::vector<EncodedRecordRow> rows;
};

}  // namespace ring
