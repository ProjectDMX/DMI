// Unescape's \uXXXX FAILURE path must leave the decoder on the string it is
// decoding. hex4 validates the length and the four digits, but the advance
// (`q += 5`) ran BEFORE the verdict was looked at, so a malformed escape
// whose "digits" include the field's real closing quote stepped past that
// quote, and the while loop kept copying the FOLLOWING JSON text -- comma,
// separator, the next field's key -- into this field's value until the next
// quote. Every FindString caller that passes no `ok` flag (pack_index.cpp's
// footer parsing, reader.cpp's cursor "fh") silently received a value
// containing structural JSON. A `\u12` at the very end of the buffer left
// `q` strictly past text.size(), breaking json.h's contract that Unescape
// "returns with `q` on the closing quote" (or at the end of an unterminated
// text).
//
// On failure the decoder consumes only the `\u` and resumes at the
// character after it: the characters that failed hex4 are copied as
// literals, the real closing quote terminates the string, and the `ok`
// latch still records the refusal.
//
// Built and run by tests/test_native_json_unescape.py.

#include <iostream>
#include <string>

#include "common/json.h"

namespace jc = dmi_common;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #cond \
                << "\n";                                                   \
      ++g_failures;                                                        \
    }                                                                      \
  } while (0)

// U+FFFD, the replacement character the decoder emits for a refusal.
const std::string kFFFD = "\xEF\xBF\xBD";

// A short escape whose "digits" run into the closing quote, spaced
// separator style (json.dumps's default). The refusal must latch AND the
// decoder must stop on the field's real closing quote: the two characters
// that failed hex4 come through as literals, and nothing past the quote --
// no comma, no separator, no next key -- leaks into the value.
void ShortEscapeStopsOnTheClosingQuote() {
  const std::string text =
      "{\"hook_name\": \"x\\u12\", \"model_id\": \"m\"}";
  const size_t start = text.find("x\\u12");
  size_t q = start;
  bool ok = true;
  const std::string value = jc::Unescape(text, q, &ok);
  CHECK(!ok);
  CHECK(value == "x" + kFFFD + "12");
  CHECK(q == start + 5);  // the field's real closing quote
  CHECK(q < text.size() && text[q] == '"');

  // The no-ok FindString path, as pack_index.cpp and reader.cpp call it:
  // the value must not contain structural JSON from beyond the quote.
  const std::string hook = jc::FindString(text, "hook_name");
  CHECK(hook.find(',') == std::string::npos);
  CHECK(hook.find("model_id") == std::string::npos);

  // A subsequent field in the same object still decodes correctly.
  bool ok_model = true;
  CHECK(jc::FindString(text, "model_id", 0, &ok_model) == "m");
  CHECK(ok_model);
}

// Compact separators: the unconditional advance landed exactly on the
// first character of the NEXT KEY, so the no-ok value came back as
// "x�model_id" -- the neighbouring field's name, spelled by nothing
// in the input's own value.
void CompactShortEscapeDoesNotSwallowTheNextKey() {
  const std::string text = "{\"hook_name\":\"x\\u1\",\"model_id\":\"m\"}";
  const std::string hook = jc::FindString(text, "hook_name");
  CHECK(hook == "x" + kFFFD + "1");
  CHECK(hook.find("model_id") == std::string::npos);
  CHECK(jc::FindString(text, "model_id") == "m");
}

// Sequential decoding: after a rejected escape in the FIRST field, `q` must
// sit on that field's closing quote so later fields scanned from `q` stay
// aligned with the object.
void SequentialFieldsStayAlignedAfterARejectedEscape() {
  const std::string text =
      "{\"a\": \"p\\uZZ\", \"b\": \"second\", \"c\": \"third\"}";
  const size_t start = text.find("p\\uZZ");
  size_t q = start;
  bool ok = true;
  const std::string a = jc::Unescape(text, q, &ok);
  CHECK(!ok);
  CHECK(a == "p" + kFFFD + "ZZ");
  CHECK(q == start + 5);  // on the closing quote of "a"'s value
  CHECK(jc::FindString(text, "b", q) == "second");
  CHECK(jc::FindString(text, "c", q) == "third");
}

// A truncated escape at the very end of the buffer: `q` must come to rest
// AT text.size() (the unterminated-string exit), never past it.
void TruncatedEscapeAtEndOfBufferKeepsQInBounds() {
  {
    std::string text = "\"x\\u12";  // opening quote, then x\u12, no close
    size_t q = 1;
    bool ok = true;
    const std::string value = jc::Unescape(text, q, &ok);
    CHECK(!ok);
    CHECK(value == "x" + kFFFD + "12");
    CHECK(q == text.size());
  }
  {
    std::string text = "\"x\\u1";
    size_t q = 1;
    bool ok = true;
    const std::string value = jc::Unescape(text, q, &ok);
    CHECK(!ok);
    CHECK(value == "x" + kFFFD + "1");
    CHECK(q == text.size());
  }
}

// A lead surrogate whose PARTNER escape is the malformed short one: the
// lone-surrogate refusal fires first, then the second escape is decoded on
// its own and refused too -- and the decoder still ends on the real
// closing quote.
void LeadSurrogateWithMalformedPartnerStaysAligned() {
  const std::string text = "{\"k\": \"a\\ud83d\\u12\", \"m\": \"v\"}";
  const size_t start = text.find("a\\ud83d");
  size_t q = start;
  bool ok = true;
  jc::Unescape(text, q, &ok);
  CHECK(!ok);
  CHECK(q < text.size() && text[q] == '"');
  CHECK(q == text.find("\", \"m\"", start));
  CHECK(jc::FindString(text, "m", q) == "v");
}

// Valid escapes are untouched: BMP, a combined surrogate pair, and the
// two-character escapes decode as before, `ok` stays latched true, and `q`
// ends on the closing quote.
void ValidEscapesAreUnchanged() {
  const std::string text =
      "{\"k\": \"a\\u2603\\ud83d\\ude00\\n\", \"m\": \"v\"}";
  const size_t start = text.find("a\\u2603");
  size_t q = start;
  bool ok = true;
  const std::string value = jc::Unescape(text, q, &ok);
  CHECK(ok);
  CHECK(value == "a\xE2\x98\x83\xF0\x9F\x98\x80\n");
  CHECK(q < text.size() && text[q] == '"');
  CHECK(jc::FindString(text, "m", q) == "v");
}

}  // namespace

int main() {
  ShortEscapeStopsOnTheClosingQuote();
  CompactShortEscapeDoesNotSwallowTheNextKey();
  SequentialFieldsStayAlignedAfterARejectedEscape();
  TruncatedEscapeAtEndOfBufferKeepsQInBounds();
  LeadSurrogateWithMalformedPartnerStaysAligned();
  ValidEscapesAreUnchanged();
  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "ok\n";
  return 0;
}
