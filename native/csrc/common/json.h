// Shared JSON scanning helpers for the stdin/stdout conformance drivers.
//
// These drivers speak a small JSON protocol to the pytest suite. The helpers
// accept both compact and spaced separators (", "/": " vs ","/":") because
// Python's json.dumps uses the latter by default. No external dependencies.

#ifndef DMI_COMMON_JSON_H_
#define DMI_COMMON_JSON_H_

#include <cstdint>
#include <string>
#include <vector>

namespace dmi_common {

// JSON string escape sequences back to raw bytes; \uXXXX to UTF-8, with a
// UTF-16 surrogate pair combined into its one non-BMP code point.
// `q` must be positioned just after the opening quote; returns with `q` on
// the closing quote.
std::string Unescape(const std::string& text, size_t& q);

// First string value for `key` anywhere in `text` (either separator style).
std::string FindString(const std::string& text, const std::string& key,
                       size_t from = 0);

// Outcome of a bounded integer scan.
enum class IntFind {
  kOk,          // found, and representable in 64 bits
  kAbsent,      // no integer value for the key
  kOutOfRange,  // the literal does not fit in 64 bits at all
};

// Which half of the 64-bit union a caller is willing to be handed.
//
// The union is the right default -- see FindIntChecked below -- but only for
// a field whose destination type is unsigned. A field the oracle types as
// SIGNED must say so, because the two's-complement bit pattern the union
// returns is indistinguishable from a negative literal once the sign of the
// input is gone: layer_number reads its int64 straight, so
// 18446744073709551615 arrives as -1, which is that field's legal "no layer"
// sentinel and is admitted. The sign is only knowable while the literal is
// still text, so the domain has to be declared here rather than checked by
// the caller afterwards.
enum class IntDomain {
  kUnion,     // [-2^63, 2^64-1] -- the default
  kSigned,    // [-2^63, 2^63-1] -- destinations the oracle types as signed
  kUnsigned,  // [0, 2^64-1] -- destinations the oracle types as UNSIGNED:
              // a negative literal is refused here, while the text still
              // carries its sign, instead of wrapping in the uint64_t cast
};

// First integer value for `key` (either separator style), bounded before it
// can overflow.
//
// Accepts the union of the signed and unsigned 64-bit ranges --
// [-2^63, 2^64-1] -- and stores the two's-complement bit pattern, so callers
// that read the result back as uint64_t recover values above INT64_MAX
// exactly (step_number, token_start/end and captured_at_ns are UInt64 in the
// catalog, and CaptureMetadata admits their whole range). Pass
// IntDomain::kSigned to accept only the signed half.
//
// A literal outside the requested domain is kOutOfRange. It is not
// representable in the type the caller asked for, so there is no int64_t this
// function could return that would be the right answer: only the caller can
// decide how to refuse.
//
// The value has to be a COMPLETE integer token: digits followed by a JSON
// delimiter. `"layer_number": 0.5` is kAbsent (present, but not an integer),
// not 0 -- the digit prefix is not the value.
IntFind FindIntChecked(const std::string& text, const std::string& key,
                       int64_t* out, IntDomain domain = IntDomain::kUnion);

// True when `key` is present with any value (either separator style).
// Needed where -1 is a legal value (layer_number) and the -1 that
// FindIntChecked's callers use for kAbsent would collide with it.
bool HasKey(const std::string& text, const std::string& key);

// First boolean value for `key`.
bool FindBool(const std::string& text, const std::string& key);

// True when the value for `key` is the literal null.
bool FindNull(const std::string& text, const std::string& key);

// Raw slice of the object value for `key`, including braces.
std::string FindObject(const std::string& text, const std::string& key);

// Raw slice of the array value for `key`, including brackets.
std::string FindArray(const std::string& text, const std::string& key);

// Top-level elements of array/object *content* (outer brackets stripped).
std::vector<std::string> SplitElements(const std::string& inside);

// Strip one []/{} layer (after trimming spaces).
std::string Unwrap(const std::string& wrapped);

// A "..." literal (with quotes) to raw text.
std::string ParseLiteral(const std::string& literal);

bool DecodeBase64(const std::string& in, std::vector<uint8_t>* out);
void EncodeBase64(const std::vector<uint8_t>& data, std::string* out);
void EncodeBase64(const uint8_t* data, size_t n, std::string* out);
void EscapeJson(const std::string& value, std::string* out);

}  // namespace dmi_common

#endif  // DMI_COMMON_JSON_H_
