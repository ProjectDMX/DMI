// Object-key derivation for staged packs, byte-identical to the Python
// object_key_for (pipeline.py) + key_component (pack.py).
//
// Layout: v1/tenant=<enc>/date=<YYYY-MM-DD>/session=<enc>/rank=<N>/<pack_id>.dmi-pack
// key_component: quote(safe="-_.=") with '~' forced to %7E; identifiers over
// 160 bytes — or already shaped like the reserved "sha256-<hex>" digest —
// collapse to "sha256-" + sha256(identifier).

#ifndef DMI_SINK_OBJECT_KEY_H_
#define DMI_SINK_OBJECT_KEY_H_

#include <cstdint>
#include <string>

namespace dmi_sink {

std::string KeyComponent(const std::string& value);

// captured_at_ns (unix nanos) -> "YYYY-MM-DD" in UTC.
std::string DateSegment(uint64_t captured_at_ns);

std::string ObjectKeyFor(const std::string& tenant_id,
                         const std::string& session_id, uint64_t producer_rank,
                         uint64_t captured_at_ns, const std::string& pack_id);

}  // namespace dmi_sink

#endif  // DMI_SINK_OBJECT_KEY_H_
