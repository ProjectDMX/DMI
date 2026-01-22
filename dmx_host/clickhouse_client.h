#ifndef __DMX_INTERFACE_H__
#define __DMX_INTERFACE_H__
#include <cstdint>
#include <string>

#include <pybind11/pybind11.h>

#include <clickhouse/client.h>

namespace dmx_host {

namespace py = pybind11;

/**
 * Python-exposed config class for ClickHouse connection + schema init behavior.
 *
 * Notes:
 * - client_settings: either None or a dict[str, Union[str,int,bool]]
 * - client_side_compress: either False/True or a string ("lz4" / "zstd" / "none")
 */
struct ClickHouseClientConfig {
    std::string host = "localhost";
    int port = 9000;
    std::string username = "default";
    std::string password = "";
    std::string database = "default";
    std::string table = "offload";
    bool secure = false;

    py::object client_settings = py::none();        // Optional[Dict[str, Union[str,int,bool]]]
    bool create_database_if_missing = true;
    bool drop_existing_database = false;            // For testing
    py::object client_side_compress = py::bool_(false);  // Union[bool,str]
    int index_granularity = 8192;
};

// Free functions exported to Python.
void clickhouse_init(int thread_idx, const ClickHouseClientConfig& cfg);
py::list clickhouse_insert(py::sequence items);
void clickhouse_cleanup();

}  // namespace dmx
#endif