// ring/ring_engine_py.h -- Plain C++ interface for RingEngine, usable from g++.
//
// All CUDA/ATen details are hidden behind a pimpl so bindings.cpp (compiled
// with g++) can include this header without needing CUDA or nvcc compilation.
//
// Implementation is in ring_engine_py.cu (compiled with nvcc).

#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ring/tensor_meta.h"   // TensorMeta, TensorMetaFifo
#include "ring/record_descriptor.h"
#include "ring/record_failure_policy.h"

// Forward-declare ATen and the generic sink so this plain interface does not
// expose their implementation headers.
namespace at { class Tensor; }
namespace ring {
class RecordSink;
class RecordSinkLease;
}

namespace ring_py {

// Plain C++ mirror of ring::RingConfig -- no CUDA types.
struct RingConfig {
    uint32_t task_ring_entries          = 1024;
    uint64_t payload_ring_bytes         = 256ULL * 1024 * 1024;
    uint64_t pinned_staging_bytes       = 0;
    uint64_t drain_poll_timeout_us      = 100;
    // Drain flush thresholds
    float    drain_flush_task_ratio     = 0.0f;
    float    drain_flush_payload_ratio  = 0.5f;
    uint64_t drain_flush_entry_threshold = 0;
    uint64_t drain_flush_byte_threshold  = 0;
    uint64_t drain_flush_timeout_us      = 0;
    // Clone per-request slices
    bool     clone_slices               = false;
    // ClickHouse insert queue limits
    uint64_t insert_queue_max_bytes     = 4096ULL * 1024 * 1024;
    uint64_t insert_queue_max_items     = 65536;
    // Drain policy: "autonomous" (flush on drain_flush rules and the poll
    // timeout) or "step_gated" (record rings only: flush what
    // allow_flush_step() grants; see ring::RingConfig::drain_step_gated).
    std::string drain_policy            = "autonomous";
};

// How a record ring treats a sink that refuses or stalls.  Fixed for the
// ring's lifetime.
struct RecordRuntimeOptions {
    ring::RecordFailurePolicy failure_policy =
        ring::RecordFailurePolicy::kRaiseAtProducer;
    // Cap on the time the record reservations of one step may wait for the
    // drain to free ring space.  A step starts at begin_record_step(), which
    // the integration calls once per model step: the record ring sees no
    // step boundary of its own (see begin_record_step), so with a budget a
    // reservation before the first begin_record_step() is refused.  Past
    // the budget the step's remaining records and every record still queued
    // for the sink, from any step, are discarded (counted in discarded_*,
    // skipped_steps and steps_with_discards), and capture resumes at the
    // next step; under kRaiseAtProducer the exhaustion is then raised at the
    // next begin_record_step() and at flush.  The skip still waits for the
    // one envelope the sink is admitting, so a budget needs a sink with an
    // admission bound (RecordSink::admission_bound) and construction
    // refuses one without; past that bound plus ring::record_drain_grace
    // the reservation fails the ring.  0 waits without bound, as before the
    // budget existed.
    uint64_t step_stall_budget_ms = 0;
};

// The record runtime's failure state and forward-side stall counters.
struct RecordCaptureStatus {
    ring::RecordFailurePolicy failure_policy =
        ring::RecordFailurePolicy::kRaiseAtProducer;
    bool failed = false;
    std::string failure;
    uint64_t discarded_descriptors = 0;
    uint64_t discarded_payloads = 0;
    uint64_t step_stall_budget_ms = 0;
    uint64_t stall_budget_exhaustions = 0;
    // Steps whose budget ran out; always equal to stall_budget_exhaustions.
    // Each such skip also discards every record still queued for the sink,
    // earlier steps' too, so steps_with_discards can exceed it.
    uint64_t skipped_steps = 0;
    // Distinct steps that lost at least one record to a spent budget: the
    // skipped steps plus the earlier steps whose records were still queued.
    uint64_t steps_with_discards = 0;
    // Time record reservations spent waiting for the drain, in total and
    // for the worst step.  The producer-stream synchronisation before each
    // wait is not included: that is the forward's own GPU work.
    uint64_t reserve_wait_ns = 0;
    uint64_t max_step_wait_ns = 0;
};

// Called by the p2p thread for each per-request tensor slice.
using SubmitFn = std::function<void(
    const std::string& model_id,
    int32_t            shard_rank,
    const std::string& req_id,
    const std::string& act_name,
    int32_t            layer_no,
    int32_t            start_token,
    int32_t            end_token,
    at::Tensor         slice)>;

// Opaque RAII engine.
class RingEnginePy {
public:
    explicit RingEnginePy(RingConfig cfg, SubmitFn submit_fn);
    explicit RingEnginePy(RingConfig cfg,
                          std::shared_ptr<ring::RecordSink> sink,
                          RecordRuntimeOptions options = {});
    explicit RingEnginePy(RingConfig cfg,
                          std::shared_ptr<ring::RecordSinkLease> lease,
                          RecordRuntimeOptions options = {});
    ~RingEnginePy();

    RingEnginePy(const RingEnginePy&)            = delete;
    RingEnginePy& operator=(const RingEnginePy&) = delete;

    void init(uint64_t stream_handle = 0);
    void start();
    void stop();

    // Enable/disable null mode (same kernel launch, no ring writes).
    void set_null_mode(bool enabled);

    // Push a complete step: context (heap-allocated, ownership transferred)
    // + all hook metas.  Single lock on the FIFO.
    void push_step(StepContext* ctx, std::vector<TensorMeta>& metas);

    // Launch producer kernel unconditionally (no condition gating).
    // Space must be guaranteed by pre-forward capacity check.
    // Three variants matching the three torch ops.  Legacy rings only:
    // each throws std::logic_error on a record ring, as push_step does, and
    // so do prepare_step, reserve_one and submit_cpu_direct below.

    // Static: copies all `nbytes`; today's behavior.
    void hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                        uint32_t hook_type,
                        uint64_t stream_handle);

    // Prefix: reads row_count[0] from device, copies
    // (row_count[0] * row_bytes) bytes.  Used by ring::producer_prefix.
    void hook_no_notify_prefix(uint64_t d_ptr, uint64_t nbytes_upper,
                                uint64_t row_count_dev_ptr,
                                uint64_t row_bytes,
                                uint32_t hook_type,
                                uint64_t stream_handle);

    // Chunked: K>1 chunked-suffix; per-chunk bytes from
    // chunk_bytes_dev_ptr[K].  Used by ring::producer_chunked.
    void hook_no_notify_chunked(uint64_t d_ptr, uint64_t nbytes_upper,
                                 uint64_t chunk_bytes_dev_ptr,
                                 uint32_t K,
                                 uint32_t hook_type,
                                 uint64_t stream_handle);

    // Additive schema-driven producers.  These methods are reachable only
    // through the separately registered ring::record_producer* operations.
    void record_no_notify(uint64_t d_ptr, uint64_t nbytes,
                          uint64_t emit_gate_ptr, int32_t emit_value,
                          uint64_t stream_handle);
    void record_no_notify_prefix(uint64_t d_ptr, uint64_t nbytes_upper,
                                 uint64_t row_count_dev_ptr,
                                 uint64_t row_bytes,
                                 uint64_t emit_gate_ptr, int32_t emit_value,
                                 uint64_t stream_handle);
    void record_no_notify_chunked(uint64_t d_ptr, uint64_t nbytes_upper,
                                  uint64_t chunk_bytes_dev_ptr,
                                  uint32_t chunk_count,
                                  uint64_t emit_gate_ptr, int32_t emit_value,
                                  uint64_t stream_handle);
    void record_no_notify_seq_prefix_pack(
        uint64_t d_ptr, uint64_t nbytes_upper,
        uint64_t valid_count_dev_ptr, uint64_t valid_prefix_sum_dev_ptr,
        uint32_t batch, uint64_t feature_bytes,
        uint64_t emit_gate_ptr, int32_t emit_value,
        uint64_t stream_handle);
    void record_no_notify_segmented_pack(
        uint64_t d_ptr, uint64_t nbytes_upper,
        uint64_t segment_start_dev_ptr, uint64_t segment_end_dev_ptr,
        uint32_t segment_count, uint64_t feature_bytes,
        uint64_t emit_gate_ptr, int32_t emit_value,
        uint64_t stream_handle);

    // Lightweight wake-up for the drain thread.
    void notify_drain();

    // Pre-forward capacity check + conditional flush.
    //
    // Called once per generate() step from _prepare_wrapper (GIL released).
    // Reads ring/staging counters internally and decides whether this step's
    // tensors fit in the ring (Case A) or must fall back to CPU-direct
    // copies (Case B).
    //
    // When a flush is needed (Case A ring-full, or Case B), this method
    // synchronises the CUDA current stream (via at::cuda::getCurrentCUDAStream)
    // and asks the drain thread to flush.  The caller does NOT need to pass a
    // stream handle -- the C++ side resolves it only when sync is required.
    //
    // Returns:
    //   0  RING_OK        -- Case A, ring has space.  No sync, no flush.
    //                        Forward may use ring::producer normally.
    //   1  RING_FLUSHED   -- Case A, ring was full.  Synced + flushed.
    //                        Ring now has space; forward may use ring::producer.
    //   2  CPU_DIRECT     -- Case B, step exceeds effective ring capacity.
    //                        Synced + flushed.  All hooks must use .cpu() path.
    //
    // For cases 0 and 1, advances cpu_payload_head_ and cpu_task_head_
    // under mgmt_mu_ to pre-allocate ring space for this step's producers,
    // unless `reserve` is false: then it only makes room.  A step whose
    // hooks all take the eager safety net although it fits (the adapter's
    // needs_eager) passes false: each hook reserves its own entry with
    // reserve_one, and a step reservation on top would never be released,
    // since the producers publish one entry per hook.
    // Also resets the internal hook index counter for hook_no_notify.
    static constexpr int STEP_RING_OK      = 0;
    static constexpr int STEP_RING_FLUSHED = 1;
    // Step's total bytes > ring capacity (or n_hooks > task ring entries):
    // even an empty ring can't hold it.  Caller falls back to the per-hook
    // safety net in HookPoint.forward via transport.force_eager = True.
    static constexpr int STEP_OVERSIZED    = 2;

    int prepare_step(uint64_t step_total_bytes, uint32_t num_hooks,
                     bool reserve = true);

    // Each item is (aligned upper bound, needs actual-byte reconciliation).
    // When the ring has no room, waits for the drain within the step's
    // remaining stall budget; see RecordRuntimeOptions.
    int reserve_record(
        const std::vector<std::pair<uint64_t, bool>>& reservation_items);

    // Start a new step for the stall budget: the reservations after this
    // call share one fresh budget, and a step skipped for a spent budget
    // ends.  Under kRaiseAtProducer it first raises a failure latched
    // during the previous step, including a spent budget, so the error
    // surfaces here, outside the forward.  Explicit because the ring has no
    // reliable step boundary of its own: eager hooks reserve per output
    // (and output ids repeat within a step), and a CUDA-graph step replays
    // one plan, or several with piecewise graphs.
    void begin_record_step();

    // Step-gated drain (drain_policy "step_gated"): close the current step's
    // gap.  Call once per model step on the producing thread, after the
    // step's producers are enqueued.  Grants the drain every ring record
    // reserved since the previous call, flushable once the GPU passes an
    // event recorded now on the current stream.  A gap without ring records
    // (none reserved, or only oversized records, which bypass the ring)
    // records no event.  Non-blocking.
    void allow_flush_step();

    RecordCaptureStatus record_capture_status() const;

    void push_record_descriptors(std::vector<ring::RecordDescriptor> descriptors);
    void submit_record_cpu_direct(at::Tensor cpu_tensor,
                                  uint64_t tensor_bytes);

    // Complete the GPU-to-host record prefix and validate descriptor and
    // reservation accounting and the configured sink's durability boundary
    // within one timeout.  Returns false only on timeout.
    bool flush_records_and_wait(uint64_t timeout_ms);

    // Submit a CPU-direct tensor to drain -> p2p pipeline.
    void submit_cpu_direct(at::Tensor cpu_tensor, uint64_t tensor_bytes);

    // Capacity queries (for startup warning only -- not called per-step).
    uint64_t payload_cap() const;
    uint64_t staging_cap() const;
    uint64_t task_cap() const;

    // Return a torch.Tensor view of the GPU payload buffer (uint8,
    // length = payload_cap()).  No copy, no ownership transfer -- the
    // buffer continues to be owned by the engine.  Used as the
    // `Tensor(a!)` mutation alias passed to every producer op call,
    // which gives AOT autograd a real R/W dependency that prevents
    // inductor from reordering successive producer launches.
    at::Tensor payload_tensor() const;

    // ---- Runtime queries / actions used by the safety-net branch in
    //      HookPoint.forward (eager-only path).  Never called during
    //      CUDA-graph capture or replay.

    // Free bytes in the payload ring not currently reserved and not
    // pending drain.  CPU-only read.  Reads 0, never a wrapped value, once
    // reservations no producer publishes have pushed the CPU accounting
    // past the ring.
    uint64_t available_capacity() const;

    // Free task-ring entries not currently reserved and not pending
    // drain.  CPU-only read.  Saturates at 0 like available_capacity().
    uint64_t available_task_slots() const;

    // Per-hook reservation: claim `nbytes` of payload ring + 1 task entry
    // for an upcoming producer kernel launch.  Used by the safety net
    // when force_eager is on and the spec is dynamic-shape.  Advances
    // cpu_payload_head/cpu_task_head atomically.  Reserves nothing and
    // throws when no task entry is free, since the producer would
    // overwrite an unconsumed READY word: the drain's failure when the
    // drain has failed, std::logic_error otherwise.  Callers
    // flush_and_wait() first.
    void reserve_one(uint64_t nbytes);

    // Synchronise the current CUDA stream + force drain to process all
    // outstanding entries.  Blocking; the Python binding releases the
    // GIL.  Used by safety-net branches that need to free ring space or
    // ensure FIFO ordering before consuming the next meta out-of-band.
    void flush_and_wait();

    // ---- Runtime CUDA-graph node toggling (legacy rings) -----------------
    // cudaGraph_t / cudaGraphNode_t / cudaGraphExec_t travel as opaque
    // uint64_t so this header stays CUDA-free (bindings.cpp is compiled with
    // g++).  The enabled (hook_type, layer_no) set is the single source of
    // truth: apply_toggle()/ensure_graph_current() (device node state) and
    // is_hook_enabled()/effective_enabled_mask() (host metadata gate) all
    // read it.  State and CUDA-graph operations live in ToggleRegistry
    // (toggle_registry.h); these methods delegate.

    // While enabled, the producer ops record their kernel node (through
    // cudaStreamGetCaptureInfo) during stream capture.  Default off.
    // Refused on a record ring.
    void enable_toggle_capture(bool enabled);
    bool toggle_capture_enabled() const;
    // Called by the producer ops during capture (C++ only, not bound).
    void register_capture_node(uint64_t graph, int hook_type, int layer_no,
                               uint64_t node);
    // Called by the producer ops when a capture-time recording cannot be
    // validated (non-kernel tail dependency, unreadable capture state, or a
    // producer the toggle does not manage): activation then fails closed.
    void note_capture_anomaly();
    uint64_t capture_anomaly_count() const;
    // Bind a captured graph to its instantiated exec (after capture).
    void bind_graph_exec(uint64_t graph, uint64_t exec);
    // Bumped by every registry mutation (capture flag, node recording,
    // anomaly, bind, clear).  Host-side guard verdicts are cached on it.
    uint64_t toggle_registry_version() const;
    // Replace the enabled (hook_type, layer_no) set and activate the gate.
    void set_enabled_hooks(const std::vector<std::pair<int,int>>& enabled);
    // Apply the enabled set to every bound exec (cudaGraphNodeSetEnabled,
    // diff-based).  Returns the first CUDA error code (0 = success).  Eager:
    // call only when no bound exec has a replay in flight.
    int  apply_toggle();
    // Host metadata gate: true iff the hook is enabled AND registered, or
    // the gate is inactive.
    bool is_hook_enabled(int hook_type, int layer_no) const;
    // Batched is_hook_enabled, one entry per query.
    std::vector<int> effective_enabled_mask(
        const std::vector<std::pair<int,int>>& query) const;
    // Lazy per-graph apply: bring ONE graph up to the current enabled set if
    // it is stale.  Call just before that graph replays.  Nonzero is a CUDA
    // error and must be treated as fatal by the caller (the metadata for the
    // new set has already been pushed).
    int  ensure_graph_current(uint64_t graph);
    // Record an event after a graph's replay for the optional pre-apply wait
    // (set_toggle_pre_apply_sync).  Nonzero must be treated as fatal.
    int  record_replay_event(uint64_t graph);
    void set_toggle_pre_apply_sync(bool enabled);
    uint64_t toggle_node_count() const;
    // Producer nodes recorded for one captured graph (0 when none or when
    // the graph was captured outside the capture window).
    uint64_t toggle_node_count_for_graph(uint64_t graph) const;
    // Process-wide count of producer kernels launched by the ring producer
    // ops (eager or under capture).  A change across a graph capture means
    // the graph captured at least one producer.
    uint64_t producer_launch_count() const;
    uint64_t bound_graph_count() const;          // 0 => apply is a no-op
    uint64_t last_apply_count() const;           // SetEnabled calls in the last apply_toggle
    bool     toggle_registry_uniform() const;    // all graphs share one hook set
    // Read-only replay-time check: the graph has recorded producer nodes AND
    // a bound exec.  A graph captured after the capture window closed is
    // neither, so its producers would run while the metadata gate filters.
    bool     is_graph_ready(uint64_t graph) const;
    // The node registry and the exec bindings cover exactly the same graphs.
    bool     toggle_registry_complete() const;
    // Full reset (re-enables disabled nodes of bound execs first): registry,
    // bindings, enabled set, gate, capture flag and lazy state.
    void clear_toggle_registry();

    // Device-side toggling (toggle_device.h).  Opt in before the capture
    // window opens; returns false when the build, driver or device lacks
    // device graph updates (the mode then stays off).
    bool enable_device_toggle(bool enabled);
    bool toggle_device_mode() const;
    uint64_t toggle_device_node_count() const;
    uint64_t toggle_device_errors() const;
    // Upload a captured graph's exec if it holds device-updatable producers
    // and was not uploaded yet (required before its first launch); a no-op
    // returning 0 otherwise.  Returns the CUDA error.
    int  upload_graph_exec(uint64_t graph, uint64_t exec);

    // Test-only: make ensure_graph_current fail with `code` (0 disables), and
    // query whether a graph's applied version is current.
    void _test_force_apply_error(int code);
    bool _test_applied_current(uint64_t graph) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ring_py
