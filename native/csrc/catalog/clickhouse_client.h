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
#include <functional>
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
  explicit ClickHouseError(const std::string& what, bool timed_out = false,
                           bool sent = true)
      : std::runtime_error(what), timed_out_(timed_out), sent_(sent) {}

  // The request ran out of time: the client's request (or connect) timeout,
  // or the RequestDeadline in force, ended it, or that deadline had already
  // passed before it could be sent, or the server gave up on it for a time
  // limit the request set (max_execution_time, lock_acquire_timeout,
  // insert_quorum_timeout). The message names the bound.
  bool timed_out() const { return timed_out_; }
  // Whether the statement may have reached the server. False only when it
  // cannot have: every attempt was refused its connection (or the name did
  // not resolve), or the deadline had passed before one could go out. A
  // write that failed with sent() false wrote nothing; with sent() true its
  // outcome is unknown (a connect timeout counts as sent, conservatively).
  bool sent() const { return sent_; }

 private:
  bool timed_out_;
  bool sent_;
};

// steady_clock nanoseconds: the clock a RequestDeadline is measured on.
uint64_t steady_now_ns();

// A deadline for every ClickHouse request made on the calling thread while
// the scope lives, on the steady clock. Each attempt's timeout (connect
// included) is cut to the time left, a request is not sent at all once the
// deadline has passed (a timed-out ClickHouseError, sent() false), and a
// retry's backoff never sleeps past it. Scopes nest; the tightest deadline
// in force wins. `bound` says what set the deadline -- the knobs behind it --
// and a timeout it causes names it.
//
// The publisher lease is what uses it (lease_coordinator.h): while a lease is
// held, every request its holder makes has to be answered before the lease
// row can expire, not only the lease's own statements.
//
// A scope may also carry a hook that execute() runs once before each request
// it sends, before the deadline is read -- so the hook can move it: the
// storage service renews its lease there when that falls due, whatever the
// request. Only the innermost scope's hook runs, so a scope nested inside
// one (the lease coordinator's, around a renewal the hook started) keeps the
// hook from running for its own requests; and a hook never runs inside
// itself.
class RequestDeadline {
 public:
  // A fixed deadline, steady ns.
  RequestDeadline(uint64_t deadline_ns, std::string bound);
  // A deadline read afresh before every request attempt, so that its owner
  // can move it while the scope lives (a lease that renews mid-pass). 0
  // means no deadline is in force at the moment.
  RequestDeadline(std::function<uint64_t()> deadline_ns, std::string bound,
                  std::function<void()> before_request = {});
  ~RequestDeadline();
  RequestDeadline(const RequestDeadline&) = delete;
  RequestDeadline& operator=(const RequestDeadline&) = delete;

  // The tightest deadline in force on this thread, 0 if none; `bound`, when
  // given, receives what set it.
  static uint64_t current(std::string* bound = nullptr);
  // Runs the innermost scope's before_request hook, if it has one and it is
  // not already running. What the hook throws propagates.
  static void before_request();

  // What the last request made on this thread while the scope lived (in a
  // nested scope too) timed out with; empty if it did not time out, or if
  // none was made. execute() records it, by note_outcome().
  const std::string& last_timeout() const { return last_timeout_; }
  static void note_outcome(const std::string& timeout);

 private:
  uint64_t fixed_ns_ = 0;
  std::function<uint64_t()> moving_ns_;
  std::string bound_;
  std::function<void()> before_request_;
  std::string last_timeout_;
  RequestDeadline* outer_;
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

// Every request is bounded: a server that accepts the connection and never
// answers must not hold a caller -- a lease renewal, a publish, a flush --
// indefinitely. The defaults bound the drivers too; the storage service and
// its reader pass their configured values.
struct ClickHouseTimeouts {
  double connect_s = 10.0;
  double request_s = 60.0;  // the whole request, connect included
};

// Where the catalog lives and how to reach it: the HTTP interface, plain or
// TLS, with optional credentials. Validated when a client is built from it
// (validate() below), so an inconsistent connection is refused before any
// request rather than at the first statement.
struct ClickHouseConnection {
  std::string scheme = "http";  // "http" or "https", lower case
  // A bare host name, IPv4 address or bracketed IPv6 address -- never a
  // URL: no scheme, port, path or userinfo ("user:pw@host").
  std::string host = "127.0.0.1";
  uint16_t port = 8123;
  // Sent as X-ClickHouse-User / X-ClickHouse-Key headers, never in the URL
  // (a URL lands in proxy, server and error logs). Empty user: no auth
  // headers, which the server reads as its `default` user.
  std::string user;
  std::string password;
  // https always verifies the peer and its name, against libcurl's built-in
  // CA bundle and/or directory unless these name a private CA
  // (CURLOPT_CAINFO / CURLOPT_CAPATH). Each REPLACES that option's built-in
  // default rather than adding to it, so whether the system roots are still
  // trusted depends on the libcurl build: one configured with both a bundle
  // and a directory (Debian, Ubuntu) keeps the other; a bundle-only build
  // (RHEL, Fedora) trusts only ca_file once it is set. To trust both, pass a
  // bundle holding the system roots and the private CA. Refused with http,
  // where they would silently do nothing.
  std::string ca_file;
  std::string ca_path;
  // A password over plain http must be opted into. Refused with https:
  // the flag admits plain http, it never downgrades TLS.
  bool allow_insecure_http = false;
  ClickHouseTimeouts timeouts;
  // Attempts per statement, first included, for the failures it is safe to
  // repeat (see execute()). 1 disables retries.
  int max_attempts = 3;
};

// Throws ClickHouseError naming the first inconsistent field. Messages never
// contain the password.
void validate(const ClickHouseConnection& connection);

// Settings computed for each attempt of a statement from the time that
// attempt is given -- the client's request timeout, cut to the
// RequestDeadline in force -- in whole milliseconds. See execute().
using AttemptSettings =
    std::function<void(uint64_t attempt_ms,
                       std::map<std::string, std::string>* settings)>;

// Whether a statement only reads, judged by its first keyword: SELECT,
// SHOW, DESCRIBE/DESC, EXISTS, CHECK, or WITH when the word INSERT appears
// nowhere in the statement (ClickHouse reads `WITH ... INSERT INTO ...` as
// an INSERT). Anything else -- including a statement that opens with a
// parenthesis or a comment -- counts as a write, the safe default, since a
// write is never repeated once it may have reached the server.
bool is_read_statement(const std::string& statement);

class ClickHouseClient {
 public:
  explicit ClickHouseClient(ClickHouseConnection connection);
  // Plain http, no credentials: the conformance drivers and local servers.
  ClickHouseClient(std::string host, uint16_t port,
                   ClickHouseTimeouts timeouts = {});
  ~ClickHouseClient();

  ClickHouseClient(const ClickHouseClient&) = delete;
  ClickHouseClient& operator=(const ClickHouseClient&) = delete;

  double request_timeout_s() const {
    return connection_.timeouts.request_s;
  }

  // Runs one statement with `%(name)s` parameters substituted client-side
  // and `settings` appended as URL parameters. Returns the parsed
  // FORMAT TSV rows (empty for writes).
  //
  // Retries, up to connection.max_attempts, with a short backoff:
  //   * a connection that was never made (refused, or the name did not
  //     resolve) -- for ANY statement, since nothing reached the server;
  //   * a transport error after connecting (reset, empty reply, short read)
  //     or a 5xx -- for reads only. A write that may have reached the
  //     server has an unknown outcome, and repeating it is the caller's
  //     decision (the fenced publish quarantines instead). ClickHouse
  //     answers 500 for permanent errors too (a row limit, a denied grant),
  //     so a 5xx that names a ClickHouse error (X-ClickHouse-Exception-Code,
  //     or a "Code: N." body) is retried only for the few transient codes
  //     listed in the .cpp; a 5xx naming none (a proxy's 502/503/504) is.
  // A timeout is never retried -- the client's own, or a time limit the
  // server enforced (see ClickHouseError::timed_out) -- so each attempt's
  // bound is the whole call's: at most max_attempts request timeouts plus the backoff, and a
  // single request timeout for anything that timed out. TLS failures (an
  // untrusted or misnamed certificate) and 4xx answers are not retried.
  // Timeouts go to libcurl in whole milliseconds, rounded up, so a positive
  // timeout below 1 ms bounds the request at 1 ms rather than not at all.
  //
  // Under a RequestDeadline the whole call, retries and backoff included,
  // ends by that deadline: each attempt's timeouts are cut to the whole
  // milliseconds left (rounded down, so never past it), no attempt starts
  // with less than a millisecond left, and no backoff sleeps past it -- the
  // last attempt's error is thrown instead, saying so. A timeout's message
  // names the bound that ended it. The innermost scope's before_request
  // hook runs first, once per call.
  //
  // Reads also carry wait_end_of_query=1, so the server buffers the result
  // and an exception part-way through it arrives as an error status rather
  // than as a 200 whose truncated body would parse as rows. A URL setting:
  // the statement bytes are unchanged.
  //
  // `attempts`, when given, receives the number of requests the statement
  // took (1 without a retry); it is set only when execute() returns.
  //
  // `per_attempt`, when given, adds settings to each attempt from the time
  // that attempt has, so that a server-side cap sent with a retry reflects
  // the time left then, not before the first attempt (the lease INSERTs'
  // max_execution_time, lease_coordinator.cpp).
  std::vector<Row> execute(
      const std::string& query, const Params& params = {},
      const std::map<std::string, std::string>& settings = {},
      int* attempts = nullptr, const AttemptSettings& per_attempt = {}) const;

 private:
  ClickHouseConnection connection_;
};

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_CLICKHOUSE_CLIENT_H
