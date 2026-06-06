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
#include <utility>   // std::pair (node-toggle hook keys)
#include <vector>

#include "ring/tensor_meta.h"   // TensorMeta, TensorMetaFifo

// Forward-declare at::Tensor so SubmitFn can use it without including ATen here.
namespace at { class Tensor; }

namespace ring_py {

// Plain C++ mirror of ring::RingConfig -- no CUDA types.
struct RingConfig {
    uint32_t task_ring_entries          = 1024;
    uint64_t payload_ring_bytes         = 256ULL * 1024 * 1024;
    uint64_t pinned_staging_bytes       = 0;
    uint64_t drain_poll_timeout_us      = 100;
    // Drain flush thresholds
    float    drain_flush_task_ratio     = 0.0f;
    float    drain_flush_payload_ratio  = 0.0f;
    uint64_t drain_flush_entry_threshold = 0;
    uint64_t drain_flush_byte_threshold  = 0;
    uint64_t drain_flush_timeout_us      = 100ULL * 1000;
    // Clone per-request slices
    bool     clone_slices               = false;
    // ClickHouse insert queue limits
    uint64_t insert_queue_max_bytes     = 4096ULL * 1024 * 1024;
    uint64_t insert_queue_max_items     = 65536;
};

// Live ring flow-control counters (debug / reserve-invariant probe).
// payload/task *_head advance on reserve() (prepare_step); *_tail_committed
// advance as the drain thread consumes ACTUAL producer task entries. A
// monotonically growing (head - tail_committed) gap means reserve() over-counts
// vs what producers actually write (e.g. node-toggle reserving for disabled
// hooks) -- the reserve invariant is held iff the ring fully drains to
// head == tail_committed after a flush.
struct RingFlushStats {
    uint64_t cpu_payload_head{0};
    uint64_t cpu_payload_tail_committed{0};
    uint64_t cpu_task_head{0};
    uint64_t cpu_task_tail_committed{0};
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
    // Three variants matching the three torch ops.

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
    // under mgmt_mu_ to pre-allocate ring space for this step's producers.
    // Also resets the internal hook index counter for hook_no_notify.
    static constexpr int STEP_RING_OK      = 0;
    static constexpr int STEP_RING_FLUSHED = 1;
    // Step's total bytes > ring capacity (or n_hooks > task ring entries):
    // even an empty ring can't hold it.  Caller falls back to the per-hook
    // safety net in HookPoint.forward via transport.force_eager = True.
    static constexpr int STEP_OVERSIZED    = 2;

    int prepare_step(uint64_t step_total_bytes, uint32_t num_hooks);

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
    // pending drain.  CPU-only read.
    uint64_t available_capacity() const;

    // Per-hook reservation: claim `nbytes` of payload ring + 1 task entry
    // for an upcoming producer kernel launch.  Used by the safety net
    // when force_eager is on and the spec is dynamic-shape.  Advances
    // cpu_payload_head/cpu_task_head atomically.
    void reserve_one(uint64_t nbytes);

    // Synchronise the current CUDA stream + force drain to process all
    // outstanding entries.  Blocking; the Python binding releases the
    // GIL.  Used by safety-net branches that need to free ring space or
    // ensure FIFO ordering before consuming the next meta out-of-band.
    void flush_and_wait();

    // Snapshot the live ring head/tail counters (reserve-invariant probe).
    RingFlushStats get_stats() const;

    // ---- Runtime node-toggle (Phase B) -----------------------------------
    // cudaGraph_t / cudaGraphNode_t / cudaGraphExec_t are passed as opaque
    // uint64_t so this header stays CUDA-free (it is included by the
    // g++-compiled bindings.cpp). The enabled (hook_type, layer_no) set is the
    // SINGLE SOURCE OF TRUTH: apply_toggle()/ensure_graph_current() (device)
    // and is_hook_enabled()/effective_enabled_mask() (host meta gate) read it.

    // enable_toggle_capture(true) makes the producer op record its kernel node
    // (via cudaStreamGetCaptureInfo) during graph capture; default off.
    void enable_toggle_capture(bool enabled);
    bool toggle_capture_enabled() const;
    // Called from the producer op during capture (C++ only, not bound).
    void register_capture_node(uint64_t graph, int hook_type, int layer_no, uint64_t node);
    // Bind a captured graph to its instantiated exec (post-warmup).
    void bind_graph_exec(uint64_t graph, uint64_t exec);
    // Set the enabled (hook_type, layer_no) set (the single source).
    void set_enabled_hooks(const std::vector<std::pair<int,int>>& enabled);
    // Apply the enabled set to every bound exec via cudaGraphNodeSetEnabled.
    // Returns the first CUDA error code (0 = success). Diff-based.
    int  apply_toggle();
    // Host meta-gate query: true if the hook is enabled AND registered (or
    // toggle inactive -> all on).
    bool is_hook_enabled(int hook_type, int layer_no) const;
    // Batched is_hook_enabled: one entry per query, same semantics. Lets the
    // per-reconfigure effective-set recompute be ONE pybind crossing instead of
    // N round-trips; C++ remains the single source of truth.
    std::vector<int> effective_enabled_mask(
        const std::vector<std::pair<int,int>>& query) const;
    // Phase 4 lazy per-graph apply. ensure_graph_current(): apply the current
    // enabled set to ONE graph if it is stale (call just before that graph
    // replays). record_replay_event(): record a stream event after that graph's
    // replay so a later ensure can wait for it before mutating the exec.
    int  ensure_graph_current(uint64_t graph);
    // Returns the CUDA error from event create/record (0 = ok). Nonzero must be
    // treated as FATAL by the caller (a missing/stale replay event would let a
    // later ensure mutate an executing exec -> UB).
    int  record_replay_event(uint64_t graph);
    // Test-only: force ensure_graph_current to fail with `code` (0 disables);
    // and query whether a graph's applied_version == target_version.
    void _test_force_apply_error(int code);
    bool _test_applied_current(uint64_t graph) const;
    uint64_t toggle_node_count() const;
    uint64_t bound_graph_count() const;          // #bound execs (0 => apply is a no-op)
    uint64_t last_apply_count() const;           // #SetEnabled calls in last apply_toggle
    bool     toggle_registry_uniform() const;    // all graphs share the same hook set
    // Read-only replay-time guard (eager + lazy): a graph is "ready" iff it has
    // recorded producer nodes AND a bound exec. A graph vLLM captured at RUNTIME
    // (new batch_descriptor, after the warmup capture window closed) is neither,
    // so its producers run default-ON while the meta gate filters -> desync. The
    // replay hook calls this and fails loud. NO version bump / event wait / node
    // mutation -- pure validation (distinct from ensure_graph_current).
    bool     is_graph_ready(uint64_t graph) const;
    // True iff the node-registry and exec-binding key sets match EXACTLY (every
    // captured graph is bound, every bound graph has nodes). Checked at
    // activation so a partial/mismatched bind fails loud instead of desyncing.
    bool     toggle_registry_complete() const;
    // Full reset: registry, exec map, enabled set, toggle_active, capture flag,
    // and Phase-4 lazy state (versions + per-graph replay events).
    void clear_toggle_registry();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ring_py
