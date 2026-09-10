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
#include <map>
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
  // Each item is one descriptor: 32 fields, the five SORT-KEY columns
  // first (tenant_id, experiment_id, run_id, captured_at_ns, capture_id)
  // and then the resolved columns in projection-minus-sort-key order --
  // NOT CAPTURE_COLUMNS order. The parity suite's normalizer and its
  // hard-coded indices (capture_id at item[4]) depend on this layout; a
  // caller indexing by the writer's column order reads the wrong fields.
  std::vector<std::vector<std::string>> items;
  std::optional<std::string> next_cursor;
  std::string watermark;
};

// sha256 over the compact sorted-keys JSON of the filters — byte-compatible
// with CaptureQuery.filter_hash, so cursors cross implementations.
std::string filter_hash(const SearchFilters& filters);

// One TSV-rendered tuple field -- the argMax aggregate that carries the
// resolved columns -- split into its values, quotes and escapes undone.
//
// Exposed for the conformance driver. A SQL NULL comes back as the empty
// string and the four-character string 'NULL' comes back as "NULL": that
// distinction only exists while the quotes are still in the text, it is
// what hydration's footer binding compares against, and it was live-only
// until this declaration put it on the CPU gate.
std::vector<std::string> parse_tsv_tuple(const std::string& text);

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
