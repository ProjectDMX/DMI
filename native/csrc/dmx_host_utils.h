#ifndef DMX_HOST_UTILS_H__
#define DMX_HOST_UTILS_H__
#include <variant>
#include <vector>
#include <string>

#include <ATen/ATen.h>

#include <cstdint>


namespace dmx_host{

// Released fixed inference row cell types. Keep this alias and its eight-cell
// contract unchanged.
using ClickHouseValue = std::variant<std::string, int32_t, std::vector<int64_t>, at::Tensor>;
using ClickHouseRow = std::vector<ClickHouseValue>;

// Additive schema-driven record cells. The integration has already encoded
// semantic metadata before this host-only representation is constructed.
using RecordValue = std::variant<std::string, int32_t, int64_t, double,
                                 std::vector<int64_t>, at::Tensor>;

struct GenericRecordRow {
    std::string layout;
    std::vector<RecordValue> cells;
};

using DmxHostRow = std::variant<ClickHouseRow, GenericRecordRow>;

struct dmx_host_queue_item{
    uint64_t item_size;
    DmxHostRow row;

    dmx_host_queue_item(ClickHouseRow queued_core, uint64_t size)
        : item_size(size), row(std::move(queued_core)) {}

    dmx_host_queue_item(GenericRecordRow queued_record, uint64_t size)
        : item_size(size), row(std::move(queued_record)) {}

    uint64_t size() const {
        return this->item_size;
    }
};

}
#endif
