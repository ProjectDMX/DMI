// Minimal ClickHouse HTTP client for the catalog protocols.
//
// Statements are sent over the HTTP interface (libcurl, already a
// dependency via the S3 client) with client-side parameter substitution,
// matching what clickhouse-driver does for `%(name)s` placeholders: the
// server receives the same statement text from both implementations,
// which is the identity the sole-claimant protocols are audited against.
// Consistency settings (DECIDING_READ and friends) ride as URL params.

#ifndef DMI_CATALOG_CLICKHOUSE_CLIENT_H
#define DMI_CATALOG_CLICKHOUSE_CLIENT_H

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <curl/curl.h>

namespace dmi_catalog {

using Param = std::variant<int64_t, uint64_t, std::string>;
using Params = std::map<std::string, Param>;
using Row = std::vector<std::string>;

class ClickHouseError : public std::runtime_error {
 public:
  explicit ClickHouseError(const std::string& what)
      : std::runtime_error(what) {}
};

// clickhouse-driver's client-side `%(name)s` substitution: one
// left-to-right pass, as `query % escaped` is. Exposed (rather than kept
// private to the client) so the conformance driver can gate it against
// the driver on the CPU gate, the way `sql_quote` already is — the
// escaper had that gate and the scanner did not.
std::string substitute(const std::string& query, const Params& params);

// Parse a TSV-rendered unsigned integer field; empty renders as 0.
uint64_t parse_u64_field(const std::string& text, const char* what);

// Settings for the statements whose answers DECIDE something: which
// claimant owns a term or a version, whether the fence admits. A
// read-back is only as sound as the visibility it is given; see
// clickhouse_sql.py for the derivation.
std::map<std::string, std::string> deciding_read();

class ClickHouseClient {
 public:
  ClickHouseClient(std::string host, uint16_t port);
  ~ClickHouseClient();

  ClickHouseClient(const ClickHouseClient&) = delete;
  ClickHouseClient& operator=(const ClickHouseClient&) = delete;

  // Runs one statement with `%(name)s` parameters substituted client-side
  // and `settings` appended as URL parameters. Returns the parsed
  // FORMAT TSV rows (empty for writes).
  std::vector<Row> execute(
      const std::string& query, const Params& params = {},
      const std::map<std::string, std::string>& settings = {}) const;

 private:
  std::string host_;
  uint16_t port_;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_CLICKHOUSE_CLIENT_H
