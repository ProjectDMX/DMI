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

#include "native_engine.h"


namespace dmx_host{

using FutureProcessValue = std::variant<std::string, int32_t, monitoring::BackendFuture>;
using FutureProcessRow = std::vector<FutureProcessValue>;

// Input cell types for a row.
using ClickHouseValue = std::variant<std::string, int32_t, std::vector<int64_t>, at::Tensor>;

// One item == one row (vector of cells).
using ClickHouseRow = std::vector<ClickHouseValue>;

using QueuedCoreType = std::variant<ClickHouseRow, FutureProcessRow>;

struct dmx_host_queue_item{
    uint64_t item_size;
    QueuedCoreType core;
    dmx_host_queue_item(QueuedCoreType queued_core, uint64_t size){
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

// Helper to check if a string starts with a prefix (C++20 has .starts_with())
bool starts_with(const std::string& str, const std::string& prefix);

// Helper to check if a string ends with a suffix (C++20 has .ends_with())
bool ends_with(const std::string& str, const std::string& suffix);

// ---------------------------------------------------------
// Main Functions
// ---------------------------------------------------------

/**
 * Parses internal ID.
 * Returns: A pair containing {int32_t, string}
 */
std::pair<int32_t, std::string> parse_internal_id(const std::string& internal_id);

/**
 * Gets delta token length.
 * shape: Passed as a vector<int64_t> to represent the tuple
 */
int64_t get_delta_token_len(const std::vector<int64_t>& shape, const std::string& act_name);

std::vector<dmx_host_queue_item> input_handler_v1(std::vector<std::vector<std::string> > keys, std::vector<int32_t> start_token_idxs, 
    std::vector<std::map<std::string, monitoring::BackendFuture> > cache_dicts);
}
#endif