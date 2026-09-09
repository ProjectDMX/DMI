// C2: hydration and core summary, ported from reader.py and summary.py.
//
// The catalog is the query index; the pack footer is the authority for
// what each payload IS — every catalog descriptor is bound to the footer
// before any payload range is fetched, in a separate first phase so a
// later pack cannot fail after earlier payloads have been fetched. The
// core summary keeps Python's numeric contract: float64 accumulators,
// order statistics off the raw dtype, the scale-before-square L2 norm,
// bfloat16 widened by a 16-bit left shift.

#ifndef DMI_CATALOG_HYDRATION_H
#define DMI_CATALOG_HYDRATION_H

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../store/s3_client.h"
#include "reader.h"

namespace dmi_catalog {

struct Selection {
  std::string selection_id;
  std::vector<std::string> capture_ids;
  std::string catalog_watermark;
  std::string filter_hash;
  std::string tenant_id;
};

struct HydrationEstimateData {
  uint64_t capture_count = 0;
  uint64_t object_count = 0;
  uint64_t request_count = 0;
  uint64_t logical_bytes = 0;
  uint64_t stored_bytes = 0;
  uint64_t request_bytes = 0;
};

struct CoreSummaryData {
  int summary_version = 2;
  uint64_t element_count = 0;
  uint64_t finite_count = 0;
  uint64_t nan_count = 0;
  uint64_t inf_count = 0;
  double zero_fraction = 0.0;
  double mean = 0.0;
  double minimum = 0.0;   // int dtypes carry the raw integer as a double
  double maximum = 0.0;
  double abs_max = 0.0;
  double l2_norm = 0.0;
  bool order_stats_are_integers = false;
  int64_t minimum_int = 0;
  int64_t maximum_int = 0;
  int64_t abs_max_int = 0;
};

// One field of a rendered footer VALUES row, typed the way the renderer
// typed it: SQL NULL is a kind, not a spelling, and a quoted string is
// DECODED (the inverse of sql_quote) rather than carried with its escapes.
struct FooterField {
  bool is_null = false;
  std::string text;  // the decoded string, or the raw number/array token
};

// Split one rendered footer row (pack_index's VALUES renderer) into its
// typed fields. Exposed for the conformance driver: the footer binding is
// a comparison between two representations, and its decoder is pinned on
// the CPU gate rather than only behind a live catalog.
std::vector<FooterField> split_footer_row(const std::string& rendered);

// Whether one DECODED catalog field (the reader's TSV layer already undid
// ClickHouse's escapes; a NULL arrives as the empty string) agrees with its
// footer counterpart.
bool footer_field_matches(const std::string& catalog_value,
                          const FooterField& footer_value);

class NativeCaptureReader {
 public:
  NativeCaptureReader(dmi_store::S3Client* s3, std::string bucket,
                      std::shared_ptr<const ClickHouseClient> client,
                      ReaderConfig catalog_config,
                      int64_t max_coalesce_gap_bytes = 4096);

  // search → one bounded page → the selection. page.next_cursor present
  // means the query exceeded one bounded page.
  Selection select(const SearchFilters& filters) const;
  HydrationEstimateData estimate(const Selection& selection) const;
  // Returns one payload per capture, in selection order.
  std::vector<std::string> hydrate(const Selection& selection,
                                   int64_t byte_limit,
                                   int64_t request_limit) const;
  std::vector<std::pair<std::string, CoreSummaryData>> summarize_core(
      const Selection& selection, int64_t byte_limit, int64_t request_limit,
      uint64_t max_summary_captures, uint64_t max_summary_elements) const;

 private:
  std::vector<std::vector<std::string>> resolve(const Selection& selection)
      const;

  dmi_store::S3Client* s3_;
  std::string bucket_;
  NativeCaptureCatalog catalog_;
  int64_t max_coalesce_gap_bytes_;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_HYDRATION_H
