#ifndef DMX_HOST_RECORD_SCHEMA_H_
#define DMX_HOST_RECORD_SCHEMA_H_

#include <cstdint>
#include <string>
#include <vector>

namespace dmx_host {

// Framework-neutral logical cell types accepted by the schema-driven host
// sink. TENSOR is one logical cell that expands into the three explicitly
// named ClickHouse columns in RecordColumn.
enum class RecordCellType : std::uint8_t {
  STRING = 0,
  INT32 = 1,
  INT64 = 2,
  FLOAT64 = 3,
  INT64_ARRAY = 4,
  TENSOR = 5,
};

struct RecordColumn {
  std::string name;
  RecordCellType type = RecordCellType::STRING;

  // Required only for TENSOR. The integration chooses all physical names.
  std::string dtype_column;
  std::string shape_column;
  std::string bytes_column;
};

struct RecordLayout {
  // Descriptor-visible layout identifier and destination table.
  std::string name;
  std::string table;
  std::vector<RecordColumn> columns;
  std::vector<std::string> primary_key;
  std::vector<std::string> order_by;
};

struct RecordSchema {
  std::vector<RecordLayout> layouts;
  int index_granularity = 8192;
};

// Validate once while constructing the stage. Native worker threads receive a
// copied immutable schema and do not access Python objects.
void ValidateRecordSchema(const RecordSchema& schema);
std::string RecordSchemaIdentity(const RecordSchema& schema);
const RecordLayout& FindRecordLayout(const RecordSchema& schema,
                                     const std::string& name);

}  // namespace dmx_host

#endif  // DMX_HOST_RECORD_SCHEMA_H_
