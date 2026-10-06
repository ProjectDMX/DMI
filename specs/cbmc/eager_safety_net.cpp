// Bounded model check of the legacy ring's eager safety net (#160): the
// reserve / flush / publish / drain protocol between the adapter's
// commit_step, RingEnginePy, HookPoint's eager branch, the GPU producers and
// the drain thread.
//
// Source of truth (main @ 5b3b632):
//   native/csrc/ring/ring_engine_py.cu
//     legacy_ring_room                              :141-144  copied
//     prepare_step, from pcap to STEP_RING_FLUSHED  :618-651  copied
//     available_capacity / available_task_slots     :937-949  copied
//     reserve_one                                   :957-974  copied
//     flush_and_wait                                :980-984  copied
//   src/dmi/adapters/base.py:395-404   commit_step's reserve / force_eager
//                                      decision                transcribed
//   src/dmi/hooks/point.py:290-337     HookPoint.forward's eager branch,
//                                      non-strip hooks         transcribed
//   native/csrc/ring/producer.cu:182-213,129-160  producer_static_kernel's
//                                      alloc and publish       abstracted
//   native/csrc/ring/drain_thread.cpp:121-139,256-260,341-375,594-601
//                                      reserve, force_flush_and_wait, the
//                                      flush's tail updates    abstracted
//
// The copied blocks sit between BEGIN COPY / END COPY markers and are
// checked against the source by specs/cbmc/sync_check.sh. They compile
// unchanged against the stand-ins defined above them: a RingEnginePy whose
// impl_->engine is the model below, and a `throw` that records which refusal
// reserve_one made (goto-cc 6.5 cannot parse libstdc++'s <stdexcept>, and
// its C++ exception support crashes; reserve_one's only statement after
// rethrow_drain_failure() is the throw, so recording the first of the two
// and returning is the same control flow).
//
// The model.
//   * CPU accounting: cpu_task_head_, cpu_payload_head_ (advanced only by
//     reserve), cpu_task_tail_, cpu_payload_tail_committed_ (advanced only
//     by the drain), as in drain_thread.h:117-121.
//   * GPU: dev_task_head, dev_payload_head -- *ring.task_head and
//     *ring.payload_head. A producer publishes at the device heads, not at
//     the CPU's reservation: it never reads the tails (producer.cu:186-213).
//     Kernels run in stream order and cudaStreamSynchronize precedes every
//     flush, so a dispatched producer is modelled as publishing at once.
//   * The drain consumes published entries in order, any number at any
//     lock boundary: every CPU accessor (each takes mgmt_mu_ on its own)
//     lets it run first. A forced flush consumes every published entry.
//     It may fail at any point; from then on it consumes nothing and
//     force_flush_and_wait returns at once (drain_thread.cpp:128,389-397).
//   * A step may stop after prepare_step at any hook (an exception in the
//     forward): the reservations it made and nothing published stay in the
//     CPU accounting for good. That is the phantom #160 is about.
//   * Staging back-pressure only delays the drain, which the model already
//     allows; strip (CPU-direct) hooks never touch the ring and are left out.
//
// Bounds: STEPS steps of 1..HMAX hooks, task ring 1..3 entries, payload and
// staging 16..64 bytes, hooks of 1..64 bytes.
//
// Properties (labels lead each assertion's text; check.sh orders by them):
//   E1 a producer never publishes into a task slot the drain has not
//      consumed (the drain alive)
//   E2 a producer never writes payload bytes the drain has not consumed
//      (with payload_ring_span.cpp's P5 this is "no overwrite")
//   E3 available_capacity() <= payload cap, available_task_slots() <= task
//      cap, at every read
//   E4 no phantom without a failure: after every step, when no step has
//      stopped early, every CPU reservation has been published exactly
//      once -- a step is never reserved twice
//   E5 reserve_one refuses only when a step has stopped early or the drain
//      has failed: a hook that fits an empty ring is never refused
//   E6 when reserve_one refuses on a failed drain, it raises the drain's
//      failure, not the "call flush_and_wait first" logic_error
//   V1 (vacuity, expected FAILURE) an eager hook publishes after a flush
//   V2 (vacuity, expected FAILURE) reserve_one refuses with logic_error
//   V3 (vacuity, expected FAILURE) the CPU task accounting passes the ring
//
// Variants (-D), each turning one #160 change back:
//   PRE160_TASK_CHECK  HookPoint checks bytes only; reserve_one does not
//                      check the task ring
//   PRE160_STEP_RESERVE  commit_step reserves a needs_eager step as well
//   UNSATURATED        legacy_ring_room without the saturation
//   PRE160             all three: the code before #160

#define __host__
#define __device__
#include "../../native/csrc/ring/payload_ring.cuh"
#include "../../native/csrc/ring/ring_config.h"

#ifndef STEPS
#define STEPS 3
#endif
#ifndef HMAX
#define HMAX 3
#endif
#define MAXE (STEPS * HMAX)

#ifdef PRE160
#define PRE160_TASK_CHECK
#define PRE160_STEP_RESERVE
#define UNSATURATED
#endif

uint64_t nondet_u64();
bool nondet_bool();

namespace std {
// By value: CBMC 6.5 symex crashes on the reference-returning form.
template <class T> T min(T a, T b) { return b < a ? b : a; }
struct logic_error { explicit logic_error(const char*) {} };
}  // namespace std
typedef int cudaStream_t;
inline int cudaStreamSynchronize(cudaStream_t) { return 0; }  // see "the model"
namespace at { namespace cuda {
struct Stream { cudaStream_t s; cudaStream_t stream() const { return s; } };
inline Stream getCurrentCUDAStream() { Stream st; st.s = 0; return st; }
} }  // namespace at::cuda

// ---- model state ---------------------------------------------------------
static uint64_t TCAP, PCAP, SCAP;
static uint64_t cpu_task_head_, cpu_payload_head_;
static uint64_t cpu_task_tail_, cpu_payload_tail_committed_;
static uint64_t dev_task_head, dev_payload_head;
static uint64_t payload_end[MAXE];       // dev_payload_head after entry i
static bool drain_failed;
static bool stopped_early;               // some step stopped after prepare_step

enum Refusal { NONE = 0, DRAIN_FAILURE = 1, LOGIC_ERROR = 2 };
static int refusal;

// The drain consumes published entries in order (drain_thread.cpp:594-601
// for the task tail, :365 for the committed payload tail). Consuming up to
// entry n sets the payload tail to where entry n-1 ended.
static void drain_consume_to(uint64_t n) {
    cpu_task_tail_ = n;
    cpu_payload_tail_committed_ = n == 0 ? (uint64_t)0 : payload_end[n - 1];
}
// One drain pass at a lock boundary: nothing, fail, or consume through
// any published entry.
static void drain_runs() {
    if (drain_failed) return;
    if (nondet_bool()) { drain_failed = true; return; }
    const uint64_t n = nondet_u64();
    if (n > cpu_task_tail_ && n <= dev_task_head) drain_consume_to(n);
}

struct DrainModel {
    int unused;  // CBMC 6.5 mishandles references to empty structs
    uint64_t cpu_payload_head() const { drain_runs(); return cpu_payload_head_; }
    uint64_t cpu_payload_tail_committed() const { drain_runs(); return cpu_payload_tail_committed_; }
    uint64_t cpu_task_head() const { drain_runs(); return cpu_task_head_; }
    uint64_t cpu_task_tail_committed() const { drain_runs(); return cpu_task_tail_; }
    void reserve(uint64_t payload_bytes, uint32_t num_tasks) const {
        cpu_payload_head_ += payload_bytes;
        cpu_task_head_ += num_tasks;
    }
    // A flush consumes everything published, unless the drain fails part
    // way (record_drain_failure completes the waiting generation).
    void force_flush_and_wait() const {
        if (drain_failed) return;
        const uint64_t n = nondet_u64();
        if (nondet_bool()) {
            // Fails part way: what it consumed stays consumed.
            if (n > cpu_task_tail_ && n <= dev_task_head) drain_consume_to(n);
            drain_failed = true;
            return;
        }
        drain_consume_to(dev_task_head);
    }
    void rethrow_drain_failure() const {
        if (drain_failed && refusal == NONE) refusal = DRAIN_FAILURE;
    }
};
static DrainModel* the_drain;  // goto-cc 6.5 cannot zero-initialise a
                               // static object of a class with methods

struct EngineModel {
    int unused;
    uint64_t payload_cap() const { return PCAP; }
    uint64_t staging_cap() const { return SCAP; }
    uint64_t task_cap() const { return TCAP; }
    DrainModel& drain_thread() const { return *the_drain; }
};
struct Impl { EngineModel engine; bool record_mode; };

static void note_logic_error() { if (refusal == NONE) refusal = LOGIC_ERROR; }
static void refuse_on_record_ring(bool, const char*) {}  // legacy ring: record_mode is false

#ifndef UNSATURATED
// BEGIN COPY ring_engine_py.cu legacy_ring_room :141-144
uint64_t legacy_ring_room(uint64_t cap, uint64_t head, uint64_t tail) {
    const uint64_t outstanding = head - tail;
    return outstanding >= cap ? 0 : cap - outstanding;
}
// END COPY
#else
// Before #160: cap - (head - tail), unsaturated.
uint64_t legacy_ring_room(uint64_t cap, uint64_t head, uint64_t tail) {
    return cap - (head - tail);
}
#endif

struct RingEnginePy {
    static constexpr int STEP_RING_OK      = 0;
    static constexpr int STEP_RING_FLUSHED = 1;
    static constexpr int STEP_OVERSIZED    = 2;
    Impl* impl_;
    int prepare_step(uint64_t step_total_bytes, uint32_t num_hooks, bool reserve);
    uint64_t available_capacity() const;
    uint64_t available_task_slots() const;
    void reserve_one(uint64_t nbytes);
    void flush_and_wait();
};

// goto-cc 6.5 mis-deduces `auto&`; every `auto` in the copied blocks is the
// drain.
#define auto DrainModel
int RingEnginePy::prepare_step(uint64_t step_total_bytes,
                               uint32_t num_hooks,
                               bool reserve)
{
    // :580-616 (record-ring refusal, hook index reset, alignment check, the
    // disabled counter read) do not touch the accounting; the step totals
    // the model passes are aligned.
// BEGIN COPY ring_engine_py.cu prepare_step :618-651
    const uint64_t pcap = impl_->engine.payload_cap();
    const uint64_t scap = impl_->engine.staging_cap();
    const uint64_t effective_cap = std::min(pcap, scap);
    const uint64_t tcap = impl_->engine.task_cap();

    auto& drain = impl_->engine.drain_thread();

    // Case B: single step exceeds capacity (payload OR task entries).
    // Caller falls back to the per-hook safety net (force_eager + eager
    // dispatch).  We still flush so the ring is empty when the safety
    // net starts firing.
    if (step_total_bytes > effective_cap || num_hooks > tcap) {
        cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamSynchronize(ms);
        drain.force_flush_and_wait();
        return STEP_OVERSIZED;
    }

    // Case A: step fits.  Check available space for BOTH payload AND tasks.
    const uint64_t payload_avail = available_capacity();
    const uint64_t task_avail = available_task_slots();

    if (step_total_bytes <= payload_avail && num_hooks <= task_avail) {
        if (reserve) drain.reserve(step_total_bytes, num_hooks);
        return STEP_RING_OK;  // fast path -- no CUDA or thread interaction
    }

    // Either payload or task ring full from prior steps.  Sync main
    // stream so all producer kernels finish writing, then flush.
    cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(ms);
    drain.force_flush_and_wait();
    if (reserve) drain.reserve(step_total_bytes, num_hooks);
    return STEP_RING_FLUSHED;
// END COPY
}

// BEGIN COPY ring_engine_py.cu available_capacity/available_task_slots :937-949
uint64_t RingEnginePy::available_capacity() const {
    auto& drain = impl_->engine.drain_thread();
    const uint64_t head = drain.cpu_payload_head();
    return legacy_ring_room(impl_->engine.payload_cap(), head,
                            drain.cpu_payload_tail_committed());
}

uint64_t RingEnginePy::available_task_slots() const {
    auto& drain = impl_->engine.drain_thread();
    const uint64_t head = drain.cpu_task_head();
    return legacy_ring_room(impl_->engine.task_cap(), head,
                            drain.cpu_task_tail_committed());
}
// END COPY

#ifndef PRE160_TASK_CHECK
#define throw return note_logic_error(), (void)
// BEGIN COPY ring_engine_py.cu reserve_one :957-974
void RingEnginePy::reserve_one(uint64_t nbytes) {
    refuse_on_record_ring(impl_->record_mode, "legacy per-hook reservation");
    auto& drain = impl_->engine.drain_thread();
    if (available_task_slots() == 0) {
        // A failed drain never frees an entry again, and on a legacy ring
        // its flush_and_wait returns at once without raising (the failure
        // went to stderr), so a caller that just flushed lands here.  Name
        // that failure rather than asking for the flush it already made.
        drain.rethrow_drain_failure();
        throw std::logic_error(
            "reserve_one: every task-ring entry is reserved; call "
            "flush_and_wait first. A flush frees only entries a producer "
            "published, so right after one the entries belong to "
            "reservations nothing publishes, such as a step that failed "
            "after prepare_step");
    }
    drain.reserve(ring::align_up(nbytes, ring::PAYLOAD_ALIGN), 1);
}
// END COPY
#undef throw
#else
// reserve_one before #160 (20b58b4^).
void RingEnginePy::reserve_one(uint64_t nbytes) {
    refuse_on_record_ring(impl_->record_mode, "legacy per-hook reservation");
    impl_->engine.drain_thread().reserve(
        ring::align_up(nbytes, ring::PAYLOAD_ALIGN), 1);
}
#endif

// BEGIN COPY ring_engine_py.cu flush_and_wait :980-984
void RingEnginePy::flush_and_wait() {
    cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(ms);
    impl_->engine.drain_thread().force_flush_and_wait();
}
// END COPY

#undef auto
static RingEnginePy* engine_p;
#define engine (*engine_p)

// ---- GPU producer: producer_static_kernel + publish_last_block_arrives ----
static bool eager_branch_two;  // set while HookPoint is in its flush branch
static bool v1_seen;

static void dispatch_producer(uint64_t nbytes) {
    const uint64_t alloc_bytes = ring::align_up(nbytes, ring::PAYLOAD_ALIGN);
    if (!drain_failed) {
        __CPROVER_assert(dev_task_head - cpu_task_tail_ < TCAP,
                         "E1 publish lands in a consumed task slot");
        __CPROVER_assert(dev_payload_head + alloc_bytes
                             - cpu_payload_tail_committed_ <= PCAP,
                         "E2 payload write stays clear of unconsumed bytes");
    }
    if (eager_branch_two) v1_seen = true;
    dev_payload_head += alloc_bytes;
    if (dev_task_head < MAXE) payload_end[dev_task_head] = dev_payload_head;
    dev_task_head += 1;
}

static uint64_t checked_available_capacity() {
    const uint64_t v = engine.available_capacity();
    const uint64_t t = engine.available_task_slots();
    __CPROVER_assert(v <= PCAP && t <= TCAP, "E3 room never exceeds the ring");
    return v;
}

static uint64_t align_up_py(uint64_t x, uint64_t a) { return (x + a - 1) & ~(a - 1); }

// HookPoint.forward, eager branch (point.py:290-337), for a non-strip hook.
// Returns false when the hook raised.
static bool hook_eager(uint64_t nbytes) {
    const uint64_t transport_bytes = align_up_py(nbytes, 16);
    const uint64_t effective_cap = std::min(PCAP, SCAP);  // transport.effective_cap
    refusal = NONE;
    eager_branch_two = false;
    if (transport_bytes <= std::min(checked_available_capacity(), effective_cap)
#ifndef PRE160_TASK_CHECK
            && engine.available_task_slots() > 0
#endif
       ) {
        engine.reserve_one(nbytes);
    } else if (transport_bytes <= effective_cap) {
        engine.flush_and_wait();
        eager_branch_two = true;
        engine.reserve_one(nbytes);
    } else {
        engine.flush_and_wait();
        return true;  // submit_cpu_direct: the ring is not used
    }
    if (refusal != NONE) {
        __CPROVER_assert(stopped_early || drain_failed,
                         "E5 refused only after a phantom or a failed drain");
        __CPROVER_assert(!drain_failed || refusal == DRAIN_FAILURE,
                         "E6 a refusal on a failed drain names the drain");
        __CPROVER_assert(refusal != LOGIC_ERROR,
                         "V2 vacuity: reserve_one refuses with logic_error");
        return false;
    }
    dispatch_producer(nbytes);
    eager_branch_two = false;
    return true;
}

int main() {
    TCAP = nondet_u64();
    __CPROVER_assume(TCAP >= 1 && TCAP <= 3);
    const uint64_t pg = nondet_u64(), sg = nondet_u64();
    __CPROVER_assume(pg >= 1 && pg <= 4 && sg >= 1 && sg <= 4);
    PCAP = 16 * pg;
    SCAP = 16 * sg;
    DrainModel drain;
    Impl impl;
    impl.record_mode = false;
    RingEnginePy eng;
    eng.impl_ = &impl;
    the_drain = &drain;
    engine_p = &eng;

    for (int step = 0; step < STEPS; ++step) {
        // plan_step (base.py:446-479): per-hook aligned bytes, hook count.
        uint64_t h = nondet_u64();
        __CPROVER_assume(h >= 1 && h <= HMAX);
        uint64_t nbytes[HMAX];
        uint64_t total = 0;
        for (int i = 0; i < HMAX; ++i) {
            nbytes[i] = nondet_u64();
            __CPROVER_assume(nbytes[i] >= 1 && nbytes[i] <= 64);
            if ((uint64_t)i < h) total += align_up_py(nbytes[i], 16);
        }
        const bool needs_eager = nondet_bool();

        // commit_step (base.py:395-404).
#ifndef PRE160_STEP_RESERVE
        const bool reserve = !needs_eager;
#else
        const bool reserve = true;
#endif
        const int rc = engine.prepare_step(total, (uint32_t)h, reserve);
        const bool force_eager =
            rc == RingEnginePy::STEP_OVERSIZED || needs_eager;

        // The forward: each hook in turn, possibly stopping early.
        const uint64_t stop_at = nondet_u64();   // >= h: runs to the end
        bool ok = true;
        for (int i = 0; i < HMAX; ++i) {
            if ((uint64_t)i >= h) break;
            if ((uint64_t)i == stop_at) { stopped_early = true; ok = false; break; }
            drain_runs();
            if (!force_eager) {
                dispatch_producer(nbytes[i]);     // fast path
            } else if (!hook_eager(nbytes[i])) {
                ok = false;                       // the hook raised
                break;
            }
        }

        if (ok && !stopped_early) {
            __CPROVER_assert(cpu_task_head_ == dev_task_head &&
                             cpu_payload_head_ == dev_payload_head,
                             "E4 every reservation published exactly once");
        }
        __CPROVER_assert(cpu_task_head_ - cpu_task_tail_ <= TCAP,
                         "V3 vacuity: the task accounting passes the ring");
    }
    __CPROVER_assert(!v1_seen, "V1 vacuity: an eager hook publishes after a flush");
    return 0;
}
