// Host-only tests of ToggleRegistry exec rebinding.  Appended by
// tests/test_toggle_registry_host.py to the host half of
// ring/toggle_registry.cu (namespace ring_py) and compiled against the CPU
// CUDA stub in cpu_cuda_stub/; the device-kernel half is not compiled.

#include <cstdio>
#include <set>
#include <vector>

static int failures = 0;
#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                           \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

namespace {

constexpr uint64_t kGraph = 0x10;
constexpr uint64_t kOldExec = 0x20;
constexpr uint64_t kNewExec = 0x21;
const std::vector<uint64_t> kNodes = {0x30, 0x31, 0x32};
// Node i belongs to hook (0, i).

using Set = std::vector<std::pair<int, int>>;

bool exec_matches(uint64_t exec, const std::set<int>& on_layers) {
    for (size_t i = 0; i < kNodes.size(); ++i) {
        const bool want = on_layers.count(static_cast<int>(i)) > 0;
        if (fake_node_on(exec, kNodes[i]) != want) return false;
    }
    return true;
}

ring_py::ToggleRegistry* make_registry() {
    auto* r = new ring_py::ToggleRegistry();
    r->set_capture(true);
    for (size_t i = 0; i < kNodes.size(); ++i)
        r->register_node(kGraph, 0, static_cast<int>(i), kNodes[i]);
    r->set_capture(false);
    return r;
}

// Disable every hook on the old exec, then rebind to a fresh exec (all
// nodes enabled, as CUDA instantiates them) and apply `desired` lazily.
void rebind_then_lazy(const Set& desired, const std::set<int>& expect_on,
                      bool reapply_same_set) {
    fake_enabled.clear();
    auto* r = make_registry();
    r->bind_exec(kGraph, kOldExec);
    r->set_enabled({});
    EXPECT(r->apply_all() == 0);
    EXPECT(exec_matches(kOldExec, {}));

    r->bind_exec(kGraph, kNewExec);
    EXPECT(!r->applied_current(kGraph));   // the new exec is stale
    if (reapply_same_set) r->set_enabled({});
    r->set_enabled(desired);
    const unsigned before = fake_set_enabled_calls;
    EXPECT(r->ensure_current(kGraph) == 0);
    EXPECT(r->applied_current(kGraph));
    EXPECT(exec_matches(kNewExec, expect_on));
    // Every host-managed node of the new exec was set explicitly.
    EXPECT(fake_set_enabled_calls - before == kNodes.size());
    // Current now: no further calls.
    const unsigned settled = fake_set_enabled_calls;
    EXPECT(r->ensure_current(kGraph) == 0);
    EXPECT(fake_set_enabled_calls == settled);
    delete r;
}

void test_rebind_then_lazy_apply() {
    rebind_then_lazy({}, {}, false);                       // empty set
    rebind_then_lazy({}, {}, true);                        // same set again
    rebind_then_lazy({{0, 1}}, {1}, false);                // subset
    rebind_then_lazy({{0, 0}, {0, 1}, {0, 2}}, {0, 1, 2}, false);  // all
}

void test_rebind_without_set_change_is_not_reported_current() {
    // The audit's sequence: bind, disable, rebind, ensure with no new set.
    fake_enabled.clear();
    auto* r = make_registry();
    r->bind_exec(kGraph, kOldExec);
    r->set_enabled({});
    EXPECT(r->apply_all() == 0);
    r->bind_exec(kGraph, kNewExec);
    EXPECT(!r->applied_current(kGraph));
    EXPECT(r->ensure_current(kGraph) == 0);
    EXPECT(exec_matches(kNewExec, {}));
    EXPECT(!r->is_enabled(0, 0));
    delete r;
}

void test_rebind_then_eager_apply() {
    fake_enabled.clear();
    auto* r = make_registry();
    r->bind_exec(kGraph, kOldExec);
    r->set_enabled({{0, 2}});
    EXPECT(r->apply_all() == 0);
    EXPECT(exec_matches(kOldExec, {2}));
    r->bind_exec(kGraph, kNewExec);
    EXPECT(r->apply_all() == 0);
    EXPECT(exec_matches(kNewExec, {2}));
    EXPECT(r->applied_current(kGraph));
    delete r;
}

void test_same_exec_rebind_keeps_state() {
    fake_enabled.clear();
    auto* r = make_registry();
    r->bind_exec(kGraph, kOldExec);
    r->set_enabled({{0, 0}});
    EXPECT(r->ensure_current(kGraph) == 0);
    r->bind_exec(kGraph, kOldExec);          // idempotent
    EXPECT(r->applied_current(kGraph));
    const unsigned before = fake_set_enabled_calls;
    EXPECT(r->ensure_current(kGraph) == 0);
    EXPECT(fake_set_enabled_calls == before);
    delete r;
}

void test_failed_apply_after_rebind_retries() {
    fake_enabled.clear();
    auto* r = make_registry();
    r->bind_exec(kGraph, kOldExec);
    r->set_enabled({});
    EXPECT(r->apply_all() == 0);
    r->bind_exec(kGraph, kNewExec);
    fake_set_enabled_error = 700;
    EXPECT(r->ensure_current(kGraph) == 700);
    EXPECT(!r->applied_current(kGraph));
    fake_set_enabled_error = 0;
    EXPECT(r->ensure_current(kGraph) == 0);
    EXPECT(exec_matches(kNewExec, {}));
    delete r;
}

void test_clear_after_rebind_reenables_the_new_exec() {
    fake_enabled.clear();
    auto* r = make_registry();
    r->bind_exec(kGraph, kOldExec);
    r->set_enabled({});
    EXPECT(r->apply_all() == 0);
    r->bind_exec(kGraph, kNewExec);
    // Something disabled a node of the new exec that the registry has no
    // record for: clear() must not trust the old exec's cache.
    fake_enabled[{kNewExec, kNodes[0]}] = false;
    r->clear();
    EXPECT(exec_matches(kNewExec, {0, 1, 2}));
    delete r;
}

void test_rebind_drops_the_old_replay_event() {
    fake_enabled.clear();
    auto* r = make_registry();
    r->set_pre_apply_sync(true);
    r->bind_exec(kGraph, kOldExec);
    EXPECT(r->record_replay_event(kGraph, nullptr) == 0);
    EXPECT(fake_events_live == 1);
    r->bind_exec(kGraph, kNewExec);
    EXPECT(fake_events_live == 0);
    delete r;
}

}  // namespace

int main() {
    test_rebind_then_lazy_apply();
    test_rebind_without_set_change_is_not_reported_current();
    test_rebind_then_eager_apply();
    test_same_exec_rebind_keeps_state();
    test_failed_apply_after_rebind_retries();
    test_clear_after_rebind_reenables_the_new_exec();
    test_rebind_drops_the_old_replay_event();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("toggle registry host: ok\n");
    return 0;
}
