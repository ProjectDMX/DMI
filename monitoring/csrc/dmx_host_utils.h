#ifndef DMX_HOST_UTILS_H__
#define DMX_HOST_UTILS_H__
#include <variant>
#include <vector>
#include <string>

#include <ATen/ATen.h>

#include <cstdint>


namespace dmx_host{

// Input cell types for a row.
using ClickHouseValue = std::variant<std::string, int32_t, std::vector<int64_t>, at::Tensor>;

// One item == one row (vector of cells).
using ClickHouseRow = std::vector<ClickHouseValue>;

struct dmx_host_queue_item{
    uint64_t item_size;
    ClickHouseRow core;
    dmx_host_queue_item(ClickHouseRow queued_core, uint64_t size){
        this->item_size = size;
        this->core = std::move(queued_core);
    };
    uint64_t size() const {
        return this->item_size;
    };
};

}
#endif
