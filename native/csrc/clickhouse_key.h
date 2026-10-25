#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dmx_host::detail {

inline bool IsSafeIdentifier(const std::string& value) {
  if (value.empty()) return false;
  const auto first = static_cast<unsigned char>(value.front());
  if (!(std::isalpha(first) || value.front() == '_')) return false;
  return std::all_of(value.begin() + 1, value.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '_';
  });
}

inline std::optional<std::vector<std::string>> ParseIdentifierKey(
    std::string_view expression) {
  while (!expression.empty() &&
         std::isspace(static_cast<unsigned char>(expression.front()))) {
    expression.remove_prefix(1);
  }
  while (!expression.empty() &&
         std::isspace(static_cast<unsigned char>(expression.back()))) {
    expression.remove_suffix(1);
  }
  // ClickHouse 26.x can render a single-column key as `(column)`.
  // Remove only one outer pair; the identifier parser still rejects nested
  // tuples, function calls, and parentheses anywhere inside the list.
  const bool wrapped = expression.size() >= 2 &&
                       expression.front() == '(' && expression.back() == ')';
  if (wrapped) {
    expression.remove_prefix(1);
    expression.remove_suffix(1);
  }
  std::vector<std::string> names;
  std::size_t cursor = 0;
  while (cursor < expression.size()) {
    while (cursor < expression.size() &&
           std::isspace(static_cast<unsigned char>(expression[cursor]))) {
      ++cursor;
    }
    if (cursor == expression.size()) break;

    std::string name;
    if (expression[cursor] == '`') {
      ++cursor;
      bool closed = false;
      while (cursor < expression.size()) {
        const char character = expression[cursor++];
        if (character != '`') {
          name.push_back(character);
          continue;
        }
        if (cursor < expression.size() && expression[cursor] == '`') {
          name.push_back('`');
          ++cursor;
          continue;
        }
        closed = true;
        break;
      }
      if (!closed) return std::nullopt;
    } else {
      const std::size_t begin = cursor;
      while (cursor < expression.size()) {
        const unsigned char character =
            static_cast<unsigned char>(expression[cursor]);
        if (!std::isalnum(character) && expression[cursor] != '_') break;
        ++cursor;
      }
      if (cursor == begin) return std::nullopt;
      name.assign(expression.substr(begin, cursor - begin));
    }
    if (!IsSafeIdentifier(name)) return std::nullopt;
    names.push_back(std::move(name));

    while (cursor < expression.size() &&
           std::isspace(static_cast<unsigned char>(expression[cursor]))) {
      ++cursor;
    }
    if (cursor == expression.size()) break;
    if (expression[cursor] != ',') return std::nullopt;
    ++cursor;
    std::size_t next = cursor;
    while (next < expression.size() &&
           std::isspace(static_cast<unsigned char>(expression[next]))) {
      ++next;
    }
    if (next == expression.size()) return std::nullopt;
  }
  if (wrapped && names.empty()) return std::nullopt;
  return names;
}

}  // namespace dmx_host::detail
