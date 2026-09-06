// C1: the native capture reader, ported from clickhouse_reader.py.
//
// The SQL is the Python module's, textually. The cursor codec is
// byte-compatible with cursor.py: same JSON envelope (sorted keys, compact
// separators), same unpadded url-safe base64 with canonical-alphabet
// validation, same filter_hash binding — so a cursor either side issues,
// the other side accepts, and a read-parity test walks pages across both.
// The derivations (one aggregate so a row cannot be mixed, the total
// ordering argument, the membership pair) live in the Python module.

#ifndef DMI_CATALOG_READER_H
#define DMI_CATALOG_READER_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clickhouse_client.h"

namespace dmi_catalog {

struct ReaderConfig {
  std::string database;
  std::string table_prefix;
  int max_capture_ids = 10'000;
  uint64_t max_rows_to_read = 50'000'000ull;
  uint64_t max_bytes_to_read = 4ull * 1024 * 1024 * 1024;
  uint64_t max_execution_time_s = 15;
  bool consistent_snapshot_reads = true;
};

struct SearchFilters {
  std::optional<std::string> tenant_id;
  std::optional<std::string> experiment_id;
  std::optional<std::string> run_id;
  std::optional<std::string> session_id;
  std::optional<std::string> model_id;
  std::vector<std::string> hook_names;
  std::vector<int64_t> layer_numbers;
  std::optional<uint64_t> captured_after_ns;
  std::optional<uint64_t> captured_before_ns;
  std::optional<std::string> cursor;
  int limit = 1000;
};

struct SearchPage {
  // Each item is one descriptor: 32 fields in CAPTURE_COLUMNS[:-1] order —
  // the same layout the writer's rows use, so parity compares them 1:1.
  std::vector<std::vector<std::string>> items;
  std::optional<std::string> next_cursor;
  std::string watermark;
};

class NativeCaptureCatalog {
 public:
  NativeCaptureCatalog(std::shared_ptr<const ClickHouseClient> client,
                       ReaderConfig config);

  std::string current_watermark() const;
  uint64_t published_head(bool deciding) const;
  SearchPage search(const SearchFilters& filters) const;
  std::vector<std::vector<std::string>> get_by_ids(
      const std::vector<std::string>& capture_ids,
      const std::string& tenant_id, const std::string& watermark) const;

 private:
  std::string membership() const;
  std::string projection() const;
  std::string qualified(const std::string& table) const;
  std::map<std::string, std::string> settings() const;
  std::map<std::string, std::string> bounded_read_settings() const;

  std::shared_ptr<const ClickHouseClient> client_;
  ReaderConfig config_;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_READER_H
