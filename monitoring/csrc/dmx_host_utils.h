#ifndef DMX_HOST_UTILS_H__
#define DMX_HOST_UTILS_H__
#include <variant>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <map>

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

// ---------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------
std::vector<std::string> split_string(const std::string& s, char delimiter);
bool starts_with(const std::string& str, const std::string& prefix);
bool ends_with(const std::string& str, const std::string& suffix);

// ---------------------------------------------------------
// Main Functions
// ---------------------------------------------------------

std::pair<int32_t, std::string> parse_internal_id(const std::string& internal_id);

int64_t get_delta_token_len(const std::vector<int64_t>& shape, const std::string& act_name);

}
#endif
