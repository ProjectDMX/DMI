// The ONE SQL string escaper for the ported catalog statements.
//
// clickhouse-driver renders a str parameter through
// `clickhouse_driver/util/escape.py`'s `escape_param`: the value's
// characters mapped through `escape_chars_map`, wrapped in single quotes.
// Every statement this port renders inline has to reproduce that text,
// because the server must receive byte-identical SQL from either
// implementation.
//
// There used to be four copies of this loop -- the writer's row/member
// renderer, the client's `%(name)s` substitution, the pack-index
// descriptor renderer and the conformance driver's -- and all four
// escaped four of the map's TEN entries: backslash, quote, newline, tab.
// \b \f \r \0 \a \v went through raw. Both text validators admit those
// bytes (`model.py`'s `_validate_text` checks non-empty UTF-8 inside a
// byte limit and nothing more; the native side only bounds the lease
// holder's length), so the divergence was reachable with a legal value:
// a CR-bearing store_id reached the server as a raw 0x0D from here and as
// a two-character `\r` from the driver. A NUL was worse than divergent --
// it left the literal unterminated, and the statement was refused.
//
// So: one function, the whole map, no second copy to drift.

#ifndef DMI_CATALOG_SQL_ESCAPE_H
#define DMI_CATALOG_SQL_ESCAPE_H

#include <string>

namespace dmi_catalog {

// SQL string literal, ClickHouse escaping — shared by every renderer.
inline std::string sql_quote(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('\'');
  // escape_chars_map, entry for entry. Switched on the char itself
  // rather than searched for in a string of needles, because '\0'
  // terminates such a needle and would silently drop out of the set.
  for (const char c : value) {
    switch (c) {
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\r': out += "\\r"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\0': out += "\\0"; break;
      case '\a': out += "\\a"; break;
      case '\v': out += "\\v"; break;
      case '\\': out += "\\\\"; break;
      case '\'': out += "\\'"; break;
      default: out.push_back(c); break;
    }
  }
  out.push_back('\'');
  return out;
}

}  // namespace dmi_catalog

#endif  // DMI_CATALOG_SQL_ESCAPE_H
