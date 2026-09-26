// ring/ring_engine_py.cu -- Pimpl implementation of RingEnginePy.
// Compiled with nvcc so it can instantiate ring::RingEngine (needs CUDA).

#include "ring_engine_py.h"
#include "ring/ring_engine.h"
#include "ring/drain_thread.h"
#include "ring/ring_state.h"
#include "ring/ring_config.h"
#include "ring/tensor_meta.h"
#include "ring/ring_torch_op.h"
#include "ring/producer.cuh"
#include "ring/ring_debug.h"
#include "ring/record_stall_budget.h"
#include <ATen/cuda/CUDAContext.h>  // at::cuda::getCurrentCUDAStream
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

// Forward-declare symbols from producer.cu
namespace ring {
void set_ring_null_mode(bool enabled);
}  // namespace ring

namespace ring_py {

namespace {

using FlushClock = std::chrono::steady_clock;

std::string describe(const std::exception_ptr& failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& error) {
        return error.what();
    } catch (...) {
        return "unknown record failure";
    }
}

void check_flush_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    throw std::runtime_error(
        std::string("record flush ") + operation + " failed: " +
        cudaGetErrorString(error));
}

class ScopedFlushEvent {
public:
    ScopedFlushEvent() {
        check_flush_cuda(
            cudaEventCreateWithFlags(&event_, cudaEventDisableTiming),
            "cudaEventCreateWithFlags");
    }

    ~ScopedFlushEvent() noexcept {
        if (event_ != nullptr) cudaEventDestroy(event_);
    }

    ScopedFlushEvent(const ScopedFlushEvent&) = delete;
    ScopedFlushEvent& operator=(const ScopedFlushEvent&) = delete;

    cudaEvent_t get() const { return event_; }

    cudaError_t destroy() noexcept {
        if (event_ == nullptr) return cudaSuccess;
        cudaEvent_t event = event_;
        event_ = nullptr;
        return cudaEventDestroy(event);
    }

private:
    cudaEvent_t event_{nullptr};
};

bool wait_for_stream_prefix_until(
    cudaStream_t stream, FlushClock::time_point deadline) {
    ScopedFlushEvent event;
    check_flush_cuda(cudaEventRecord(event.get(), stream), "cudaEventRecord");

    for (;;) {
        const cudaError_t status = cudaEventQuery(event.get());
        if (status == cudaSuccess) {
            check_flush_cuda(event.destroy(), "cudaEventDestroy");
            return true;
        }
        if (status != cudaErrorNotReady) {
            check_flush_cuda(status, "cudaEventQuery");
        }

        const auto now = FlushClock::now();
        if (now >= deadline) {
            check_flush_cuda(event.destroy(), "cudaEventDestroy");
            return false;
        }
        std::this_thread::sleep_until(
            std::min(deadline, now + std::chrono::microseconds(50)));
    }
}

FlushClock::time_point record_flush_deadline(uint64_t timeout_ms) {
    const auto now = FlushClock::now();
    const auto max_remaining = FlushClock::time_point::max() - now;
    const auto max_milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(max_remaining);
    if (timeout_ms >= static_cast<uint64_t>(max_milliseconds.count())) {
        return FlushClock::time_point::max();
    }
    return now + std::chrono::milliseconds(timeout_ms);
}

// The legacy producer protocol -- a step reservation, then producers whose
// payloads the drain pairs with pushed TensorMetas -- and the record
// protocol share one ring but not its bookkeeping. A legacy reservation on
// a record ring advances the CPU heads with no record publication behind
// it, so the space is never reclaimed and every later reserve_record's
// reclaim sequence is off by the phantom task count. A legacy payload
// carries no record descriptor, so the record consumer pairs it with the
// next record's descriptor, or fails the ring when none is queued. Refuse
// before either happens, as push_step refuses legacy metadata.
void refuse_on_record_ring(bool record_mode, const char* what) {
    if (record_mode) {
        throw std::logic_error(
            std::string(what) + " cannot be used on a record ring");
    }
}

}  // namespace

// ---------------------------------------------------------------------------
struct RingEnginePy::Impl {
    TensorMetaFifo   fifo;
    ring::RingEngine engine;
    uint32_t         current_hook_idx{0};

    // Snapshot of the device-side actual_bytes_counter as of the last
    // prepare_step call.  Used to compute the per-step delta of bytes the
    // producer actually wrote, for reclamation accounting when a step's
    // reservation overshoots its actual writes.  Consumed by future
    // GPU-side-strip flows where the producer's src_bytes is set from a
    // device tensor at execution time and the CPU can't know it upfront.
    uint64_t         last_counter_read{0};

    // Cached torch.Tensor view of the payload buffer.  Built once at
    // engine init; returned by payload_tensor().  Used as the
    // Tensor(a!) mutation alias passed to every producer op call.
    at::Tensor       payload_view;
    bool             record_mode{false};

    // Record rings only.  The step state is written by the reserving
    // thread; the atomics let record_capture_status read from any thread.
    RecordRuntimeOptions         record_options;
    // The sink's admission bound, when a budget needs one (see the ctor),
    // and the time the drain may take past it to move what the ring holds
    // once a budget is spent (ring::record_drain_grace).
    FlushClock::duration         sink_admission_bound{};
    FlushClock::duration         drain_grace{};
    std::atomic<uint64_t>        step_wait_ns{0};
    std::atomic<uint64_t>        reserve_wait_ns{0};
    std::atomic<uint64_t>        max_step_wait_ns{0};
    std::atomic<uint64_t>        stall_budget_exhaustions{0};
    std::atomic<uint64_t>        skipped_steps{0};
    // begin_record_step() has run at least once.
    std::atomic<bool>            step_started{false};
    // This step's budget ran out: its remaining records are being skipped.
    std::atomic<bool>            skipping_step{false};
    // kRaiseAtProducer: the spent budget, raised at the next step boundary
    // and at flush rather than inside the forward that ran out of it.
    mutable std::mutex           step_failure_mu;
    std::exception_ptr           step_failure;

    Impl(ring::RingConfig cfg, SubmitFn sf)
        : engine(std::move(cfg), fifo, std::move(sf))
    {
        const auto& state = engine.ring_state();
        int dev_idx = 0;
        cudaGetDevice(&dev_idx);
        payload_view = at::from_blob(
            state.payload_buf,
            {static_cast<int64_t>(state.payload_cap)},
            at::TensorOptions().dtype(at::kByte).device(at::kCUDA, dev_idx));
    }

    Impl(ring::RingConfig cfg, std::shared_ptr<ring::RecordSinkLease> lease,
         RecordRuntimeOptions options)
        : engine(std::move(cfg), std::move(lease), options.failure_policy),
          record_mode(true), record_options(options)
    {
        if (options.step_stall_budget_ms != 0) {
            // Past the budget the forward still waits for the one envelope
            // the sink is admitting; only a sink that bounds its admission
            // makes that wait, and so the step's stall, bounded.
            const auto sink = engine.record_sink();
            const auto bound = sink ? sink->admission_bound() : std::nullopt;
            if (!bound) {
                throw std::invalid_argument(
                    "step_stall_budget_ms needs a record sink with an "
                    "admission bound (a NativePackSink with overload "
                    "'drop_newest', or 'block' with an admission_timeout_s); "
                    "this sink's admission has no bound, so a stall past the "
                    "budget would not be bounded either");
            }
            sink_admission_bound =
                std::chrono::duration_cast<FlushClock::duration>(*bound);
            drain_grace = std::chrono::duration_cast<FlushClock::duration>(
                ring::record_drain_grace(
                    engine.payload_cap(), engine.staging_cap()));
        }
        const auto& state = engine.ring_state();
        int dev_idx = 0;
        cudaGetDevice(&dev_idx);
        payload_view = at::from_blob(
            state.payload_buf,
            {static_cast<int64_t>(state.payload_cap)},
            at::TensorOptions().dtype(at::kByte).device(at::kCUDA, dev_idx));
    }

    void account_record_wait(FlushClock::time_point started) {
        const uint64_t waited = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                FlushClock::now() - started).count());
        reserve_wait_ns.fetch_add(waited, std::memory_order_relaxed);
        const uint64_t step = step_wait_ns.fetch_add(
            waited, std::memory_order_relaxed) + waited;
        uint64_t worst = max_step_wait_ns.load(std::memory_order_relaxed);
        while (step > worst && !max_step_wait_ns.compare_exchange_weak(
                   worst, step, std::memory_order_relaxed)) {
        }
    }

    // Make room for a record reservation that did not fit: finish the
    // producers already queued, then drain the ring.  The drain is held back
    // by the record worker, and the worker by the sink, so this is where a
    // slow or stuck sink reaches the forward.  The wait is bounded by what
    // is left of this step's stall budget.  Past it the rest of the step is
    // skipped: every record still queued for the sink, from any step, and
    // every record the step still reserves are discarded, which leaves the
    // drain waiting only for the envelope the sink is admitting.
    void wait_for_record_space() {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamSynchronize(stream);
        auto& drain = engine.drain_thread();
        auto& consumer = engine.record_consumer();
        const bool disable = record_options.failure_policy ==
            ring::RecordFailurePolicy::kDisableCapture;
        const auto started = FlushClock::now();
        if (record_options.step_stall_budget_ms == 0) {
            drain.force_flush_and_wait_until(FlushClock::time_point::max());
            account_record_wait(started);
            return;
        }
        // Already skipping this step, or capture is off: the worker discards
        // instead of submitting, so the sink is out of the path.
        if (skipping_step.load(std::memory_order_relaxed) ||
            (disable && consumer.failed())) {
            wait_past_the_budget(started);
            return;
        }
        const auto budget = std::chrono::duration_cast<FlushClock::duration>(
            std::chrono::milliseconds(record_options.step_stall_budget_ms));
        const auto spent = std::chrono::duration_cast<FlushClock::duration>(
            std::chrono::nanoseconds(
                step_wait_ns.load(std::memory_order_relaxed)));
        const auto deadline =
            started + (spent < budget ? budget - spent : FlushClock::duration{});
        const bool drained = drain.force_flush_and_wait_until(deadline);
        account_record_wait(started);
        if (drained) return;

        stall_budget_exhaustions.fetch_add(1, std::memory_order_relaxed);
        skipped_steps.fetch_add(1, std::memory_order_relaxed);
        skipping_step.store(true, std::memory_order_relaxed);
        consumer.begin_discard_window();
        if (!disable) {
            const auto step_ms =
                step_wait_ns.load(std::memory_order_relaxed) / 1'000'000ull;
            std::lock_guard<std::mutex> lock(step_failure_mu);
            if (!step_failure) {
                step_failure = std::make_exception_ptr(std::runtime_error(
                    "record capture stall budget exhausted: reservations "
                    "waited " + std::to_string(step_ms) + " ms in one step "
                    "for the sink to free ring space (step_stall_budget_ms=" +
                    std::to_string(record_options.step_stall_budget_ms) +
                    "); the step's remaining records and every record still "
                    "queued for the sink, from any step, were discarded"));
            }
        }
        // The reservation must still complete: under CUDA-graph replay the
        // producers launch whatever the host decides.
        wait_past_the_budget(FlushClock::now());
    }

    // With the sink out of the path, what is left is the envelope it is
    // admitting (at most its admission bound) and moving what the ring
    // holds out to the worker, which discards it (at most drain_grace).
    // Past both the ring is failed under either policy: the alternative is
    // a forward that never returns.  Either the sink held its admission past
    // its bound or the drain was slower than the grace assumes; the wait
    // cannot tell which, so the error names both.
    void wait_past_the_budget(FlushClock::time_point started) {
        auto& drain = engine.drain_thread();
        const auto deadline = started + sink_admission_bound + drain_grace;
        const bool drained = drain.force_flush_and_wait_until(deadline);
        account_record_wait(started);
        if (drained) return;
        const auto ms = [](FlushClock::duration duration) {
            return std::to_string(std::chrono::duration_cast<
                std::chrono::milliseconds>(duration).count());
        };
        const std::exception_ptr failure =
            std::make_exception_ptr(std::runtime_error(
                "record ring did not drain, with the sink out of the path, "
                "within the sink's admission bound (" +
                ms(sink_admission_bound) + " ms) plus the drain grace (" +
                ms(drain_grace) + " ms: 2 s plus the ring and staging bytes "
                "at 1 GB/s); either the sink held an admission past its bound "
                "or the drain was slower than the grace allows"));
        engine.record_consumer().record_failure(failure);
        std::rethrow_exception(failure);
    }

    // kRaiseAtProducer: the spent budget of an earlier step, if any.
    std::exception_ptr pending_step_failure() const {
        std::lock_guard<std::mutex> lock(step_failure_mu);
        return step_failure;
    }
};

// ---------------------------------------------------------------------------
static ring::RingConfig convert(const RingConfig& c) {
    ring::RingConfig r{};
    r.task_ring_entries           = c.task_ring_entries;
    r.payload_ring_bytes          = c.payload_ring_bytes;
    r.pinned_staging_bytes        = c.pinned_staging_bytes;
    r.drain_poll_timeout_us       = c.drain_poll_timeout_us;
    r.drain_flush.task_ratio      = c.drain_flush_task_ratio;
    r.drain_flush.payload_ratio   = c.drain_flush_payload_ratio;
    r.drain_flush.entry_threshold = c.drain_flush_entry_threshold;
    r.drain_flush.byte_threshold  = c.drain_flush_byte_threshold;
    r.drain_flush.timeout_us      = c.drain_flush_timeout_us;
    r.clone_slices                = c.clone_slices;
    r.insert_queue_max_bytes      = c.insert_queue_max_bytes;
    r.insert_queue_max_items      = c.insert_queue_max_items;
    return r;
}

// ---------------------------------------------------------------------------
RingEnginePy::RingEnginePy(RingConfig cfg, SubmitFn submit_fn) {
    impl_ = std::make_unique<Impl>(convert(cfg), std::move(submit_fn));
}

RingEnginePy::RingEnginePy(
    RingConfig cfg, std::shared_ptr<ring::RecordSink> sink,
    RecordRuntimeOptions options)
    : RingEnginePy(
          std::move(cfg), ring::RecordSinkLease::acquire(std::move(sink)),
          options) {}

RingEnginePy::RingEnginePy(
    RingConfig cfg, std::shared_ptr<ring::RecordSinkLease> lease,
    RecordRuntimeOptions options) {
    impl_ = std::make_unique<Impl>(convert(cfg), std::move(lease), options);
}

RingEnginePy::~RingEnginePy() = default;

void RingEnginePy::init(uint64_t stream_handle) {
    impl_->engine.init(reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::start() {
    ring_diag_reset_host_counters();
    impl_->engine.start();
}

void RingEnginePy::stop() {
    impl_->engine.stop();
#if RING_DEBUG
    ring_diag_print_host_counters();
#endif
}

void RingEnginePy::set_null_mode(bool enabled) {
    // cudaMemcpyToSymbol goes through the legacy default stream, which does
    // NOT synchronize with PyTorch's non-blocking compute streams.  Sync
    // before to drain pending producer kernels that need the old value,
    // and after to ensure the new value is visible before the next launch.
    cudaDeviceSynchronize();
    ring::set_ring_null_mode(enabled);
    cudaDeviceSynchronize();
}



void RingEnginePy::push_step(StepContext* ctx, std::vector<TensorMeta>& metas) {
    if (impl_->record_mode) {
        delete ctx;
        throw std::logic_error("legacy metadata cannot be pushed to a record ring");
    }
    impl_->fifo.push_step(ctx, metas);
}

// ---------------------------------------------------------------------------
// hook_no_notify (3 variants) -- unconditional producer launches.
//
// No condition gating.  Space is guaranteed by the pre-forward capacity
// check in Python.  Each variant maps to one torch op.  Legacy rings only:
// a record ring's producers are the record_no_notify* family.
// ---------------------------------------------------------------------------
void RingEnginePy::hook_no_notify(uint64_t d_ptr, uint64_t nbytes,
                                  uint32_t hook_type,
                                  uint64_t stream_handle)
{
    refuse_on_record_ring(impl_->record_mode, "legacy producer");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    RING_DBG("[hook_no_notify_static] idx=%u nbytes=%lu\n",
            impl_->current_hook_idx, (unsigned long)nbytes);
    impl_->current_hook_idx++;
    ring::launch_producer_static(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes, hook_type, stream);
}

void RingEnginePy::hook_no_notify_prefix(uint64_t d_ptr, uint64_t nbytes_upper,
                                          uint64_t row_count_dev_ptr,
                                          uint64_t row_bytes,
                                          uint32_t hook_type,
                                          uint64_t stream_handle)
{
    refuse_on_record_ring(impl_->record_mode, "legacy producer");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    RING_DBG("[hook_no_notify_prefix] idx=%u nbytes_upper=%lu row_bytes=%lu\n",
            impl_->current_hook_idx, (unsigned long)nbytes_upper,
            (unsigned long)row_bytes);
    impl_->current_hook_idx++;
    ring::launch_producer_prefix(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes_upper,
        reinterpret_cast<const int64_t*>(row_count_dev_ptr),
        row_bytes,
        hook_type, stream);
}

void RingEnginePy::hook_no_notify_chunked(uint64_t d_ptr, uint64_t nbytes_upper,
                                           uint64_t chunk_bytes_dev_ptr,
                                           uint32_t K,
                                           uint32_t hook_type,
                                           uint64_t stream_handle)
{
    refuse_on_record_ring(impl_->record_mode, "legacy producer");
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);
    RING_DBG("[hook_no_notify_chunked] idx=%u nbytes_upper=%lu K=%u\n",
            impl_->current_hook_idx, (unsigned long)nbytes_upper, K);
    impl_->current_hook_idx++;
    ring::launch_producer_chunked(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr),
        nbytes_upper,
        reinterpret_cast<const int64_t*>(chunk_bytes_dev_ptr),
        K,
        hook_type, stream);
}

void RingEnginePy::record_no_notify(
    uint64_t d_ptr, uint64_t nbytes,
    uint64_t emit_gate_ptr, int32_t emit_value,
    uint64_t stream_handle) {
    if (!impl_->record_mode) {
        throw std::logic_error("record producer requires a record ring");
    }
    ring::launch_record_producer_static(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr), nbytes,
        reinterpret_cast<const int32_t*>(emit_gate_ptr), emit_value,
        reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::record_no_notify_prefix(
    uint64_t d_ptr, uint64_t nbytes_upper,
    uint64_t row_count_dev_ptr, uint64_t row_bytes,
    uint64_t emit_gate_ptr, int32_t emit_value,
    uint64_t stream_handle) {
    if (!impl_->record_mode) {
        throw std::logic_error("record producer requires a record ring");
    }
    ring::launch_record_producer_prefix(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr), nbytes_upper,
        reinterpret_cast<const int64_t*>(row_count_dev_ptr), row_bytes,
        reinterpret_cast<const int32_t*>(emit_gate_ptr), emit_value,
        reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::record_no_notify_chunked(
    uint64_t d_ptr, uint64_t nbytes_upper,
    uint64_t chunk_bytes_dev_ptr, uint32_t chunk_count,
    uint64_t emit_gate_ptr, int32_t emit_value,
    uint64_t stream_handle) {
    if (!impl_->record_mode) {
        throw std::logic_error("record producer requires a record ring");
    }
    ring::launch_record_producer_chunked(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr), nbytes_upper,
        reinterpret_cast<const int64_t*>(chunk_bytes_dev_ptr), chunk_count,
        reinterpret_cast<const int32_t*>(emit_gate_ptr), emit_value,
        reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::record_no_notify_seq_prefix_pack(
    uint64_t d_ptr, uint64_t nbytes_upper,
    uint64_t valid_count_dev_ptr, uint64_t valid_prefix_sum_dev_ptr,
    uint32_t batch, uint64_t feature_bytes,
    uint64_t emit_gate_ptr, int32_t emit_value,
    uint64_t stream_handle) {
    if (!impl_->record_mode) {
        throw std::logic_error("record producer requires a record ring");
    }
    ring::launch_record_producer_seq_prefix_pack(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr), nbytes_upper,
        reinterpret_cast<const int64_t*>(valid_count_dev_ptr),
        reinterpret_cast<const int64_t*>(valid_prefix_sum_dev_ptr), batch,
        feature_bytes, reinterpret_cast<const int32_t*>(emit_gate_ptr),
        emit_value, reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::record_no_notify_segmented_pack(
    uint64_t d_ptr, uint64_t nbytes_upper,
    uint64_t segment_start_dev_ptr, uint64_t segment_end_dev_ptr,
    uint32_t segment_count, uint64_t feature_bytes,
    uint64_t emit_gate_ptr, int32_t emit_value,
    uint64_t stream_handle) {
    if (!impl_->record_mode) {
        throw std::logic_error("record producer requires a record ring");
    }
    ring::launch_record_producer_segmented_pack(
        impl_->engine.ring_state(),
        reinterpret_cast<const uint8_t*>(d_ptr), nbytes_upper,
        reinterpret_cast<const int64_t*>(segment_start_dev_ptr),
        reinterpret_cast<const int64_t*>(segment_end_dev_ptr), segment_count,
        feature_bytes, reinterpret_cast<const int32_t*>(emit_gate_ptr),
        emit_value, reinterpret_cast<cudaStream_t>(stream_handle));
}

void RingEnginePy::notify_drain() {
    impl_->engine.drain_thread().notify();
}

// ---------------------------------------------------------------------------
// prepare_step -- single Python->C++ call for pre-forward capacity check.
//
// Fast path (STEP_RING_OK): reads two uint64_t counters, returns immediately.
// No stream resolution, no sync, no flush.
//
// Slow path (STEP_RING_FLUSHED / STEP_OVERSIZED): resolves the current CUDA
// stream via at::cuda::getCurrentCUDAStream(), synchronises it, then asks the
// drain thread to flush all pending entries.
// ---------------------------------------------------------------------------
int RingEnginePy::prepare_step(uint64_t step_total_bytes,
                               uint32_t num_hooks)
{
    // Record rings reserve through reserve_record.
    refuse_on_record_ring(impl_->record_mode, "legacy step reservation");
    impl_->current_hook_idx = 0;
    if (step_total_bytes % ring::PAYLOAD_ALIGN != 0) {
        throw std::invalid_argument(
            "prepare_step total must be a sum of per-tensor aligned "
            "transport sizes");
    }

    // actual_bytes_counter reclamation: DISABLED for now (see below).
    //
    // The counter exists to reclaim ring space when a step's reservation
    // OVER-estimates what the producer actually writes.  That only happens for
    // producers whose written byte count the CPU cannot size up front -- i.e.
    // variable-byte / EP "chunked" producers that reserve an upper bound.  No
    // hook currently uses that path: the vLLM adapter only wires the prefix
    // producer (CPU-known actual_q_len * row_bytes) and the basic producer
    // (CPU-known x.nbytes()), both of which reserve exactly what they write and
    // need no reclamation.  The reclamation consumer was also never landed, so
    // the delta is unused.
    //
    // Reading the counter here is NOT free: it is a host dereference of a
    // cudaMallocManaged page whose preferred location is the GPU and which the
    // producer writes every step, so the read forces a UVM coherence stall
    // (measured ~430 us/step on Llama-8B -- effectively a per-step implicit GPU
    // sync, despite no explicit cudaStreamSynchronize).  Keep it commented out
    // until a chunked-style producer AND a reclamation consumer actually exist;
    // when they do, read the counter OFF the prepare_step critical path (e.g.
    // on the drain thread, which already synchronizes) rather than here.
    //
    // const uint64_t counter_cur = *impl_->engine.ring_state().actual_bytes_counter;
    // const uint64_t counter_delta = counter_cur - impl_->last_counter_read;
    // impl_->last_counter_read = counter_cur;

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
    const uint64_t payload_avail = pcap -
        (drain.cpu_payload_head() - drain.cpu_payload_tail_committed());
    const uint64_t task_avail = tcap -
        (drain.cpu_task_head() - drain.cpu_task_tail_committed());

    if (step_total_bytes <= payload_avail && num_hooks <= task_avail) {
        drain.reserve(step_total_bytes, num_hooks);
        return STEP_RING_OK;  // fast path -- no CUDA or thread interaction
    }

    // Either payload or task ring full from prior steps.  Sync main
    // stream so all producer kernels finish writing, then flush.
    cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(ms);
    drain.force_flush_and_wait();
    drain.reserve(step_total_bytes, num_hooks);
    return STEP_RING_FLUSHED;
}

int RingEnginePy::reserve_record(
    const std::vector<std::pair<uint64_t, bool>>& reservation_items) {
    if (!impl_->record_mode) {
        throw std::logic_error("record reservation requires a record ring");
    }
    uint64_t reservation_bytes = 0;
    std::vector<ring::RecordReservationItem> items;
    items.reserve(reservation_items.size());
    for (const auto& item : reservation_items) {
        if (item.first % ring::PAYLOAD_ALIGN != 0) {
            throw std::invalid_argument(
                "record reservation bytes must be PAYLOAD_ALIGN-aligned");
        }
        if (reservation_bytes >
            std::numeric_limits<uint64_t>::max() - item.first) {
            throw std::overflow_error("record reservation byte total overflow");
        }
        reservation_bytes += item.first;
        items.push_back({item.first, item.second});
    }
    if (items.empty()) return STEP_RING_OK;
    const uint64_t num_tasks = items.size();

    const uint64_t payload_cap = impl_->engine.payload_cap();
    const uint64_t staging_cap = impl_->engine.staging_cap();
    const uint64_t effective_cap = std::min(payload_cap, staging_cap);
    const uint64_t task_cap = impl_->engine.task_cap();
    auto& drain = impl_->engine.drain_thread();
    drain.rethrow_drain_failure();
    drain.rethrow_record_reclaim_failure();
    drain.apply_pending_record_reclaims();
    // A latched failure raises here, before anything is reserved, rather
    // than from the descriptor push after a reservation no producer uses.
    // Under kDisableCapture it never raises: the ring keeps its protocol
    // and the consumer discards.
    if (impl_->record_options.failure_policy ==
        ring::RecordFailurePolicy::kRaiseAtProducer) {
        impl_->engine.record_consumer().rethrow_if_failed();
    }
    if (impl_->record_options.step_stall_budget_ms != 0 &&
        !impl_->step_started.load(std::memory_order_relaxed)) {
        // Without step boundaries the budget would span the runtime's whole
        // life, and once spent every later wait would skip at once.
        throw std::logic_error(
            "step_stall_budget_ms is set, but no record step has begun: call "
            "RecordRuntime.begin_step() once per model step, before its "
            "first reservation");
    }

    if (reservation_bytes > effective_cap || num_tasks > task_cap) {
        impl_->wait_for_record_space();
        drain.rethrow_drain_failure();
        drain.rethrow_record_reclaim_failure();
        drain.apply_pending_record_reclaims();
        if (drain.pending_record_reclaims() != 0) {
            throw std::runtime_error(
                "record flush found incomplete producer reclaims");
        }
        return STEP_OVERSIZED;
    }

    const uint64_t payload_used =
        drain.cpu_payload_head() - drain.cpu_payload_tail_committed();
    const uint64_t task_used =
        drain.cpu_task_head() - drain.cpu_task_tail_committed();
    if (reservation_bytes <= payload_cap - payload_used &&
        num_tasks <= task_cap - task_used) {
        drain.reserve_record(items);
        return STEP_RING_OK;
    }

    impl_->wait_for_record_space();
    drain.rethrow_drain_failure();
    drain.rethrow_record_reclaim_failure();
    drain.apply_pending_record_reclaims();
    if (drain.pending_record_reclaims() != 0) {
        throw std::runtime_error(
            "record flush found incomplete producer reclaims");
    }
    drain.reserve_record(items);
    return STEP_RING_FLUSHED;
}

void RingEnginePy::begin_record_step() {
    if (!impl_->record_mode) {
        throw std::logic_error("record steps require a record ring");
    }
    auto& consumer = impl_->engine.record_consumer();
    if (impl_->record_options.failure_policy ==
        ring::RecordFailurePolicy::kRaiseAtProducer) {
        // A spent budget latches here, at the step boundary: from now on
        // the runtime is failed like after any other refusal.
        if (const std::exception_ptr failure = impl_->pending_step_failure()) {
            consumer.record_failure(failure);
        }
        consumer.rethrow_if_failed();
    }
    impl_->step_wait_ns.store(0, std::memory_order_relaxed);
    impl_->step_started.store(true, std::memory_order_relaxed);
    if (impl_->skipping_step.exchange(false, std::memory_order_relaxed)) {
        consumer.end_discard_window();
    }
    consumer.begin_step();
}

RecordCaptureStatus RingEnginePy::record_capture_status() const {
    if (!impl_->record_mode) {
        throw std::logic_error("record capture status requires a record ring");
    }
    const ring::RecordConsumerSnapshot consumer =
        impl_->engine.record_consumer().snapshot();
    RecordCaptureStatus status;
    status.failure_policy = consumer.policy;
    status.failed = consumer.failed;
    status.failure = consumer.failure;
    status.discarded_descriptors = consumer.discarded_descriptors;
    status.discarded_payloads = consumer.discarded_payloads;
    status.step_stall_budget_ms = impl_->record_options.step_stall_budget_ms;
    status.stall_budget_exhaustions =
        impl_->stall_budget_exhaustions.load(std::memory_order_relaxed);
    status.skipped_steps =
        impl_->skipped_steps.load(std::memory_order_relaxed);
    status.steps_with_discards = consumer.steps_with_discards;
    if (!status.failed) {
        if (const std::exception_ptr failure = impl_->pending_step_failure()) {
            status.failed = true;
            status.failure = describe(failure);
        }
    }
    status.reserve_wait_ns =
        impl_->reserve_wait_ns.load(std::memory_order_relaxed);
    status.max_step_wait_ns =
        impl_->max_step_wait_ns.load(std::memory_order_relaxed);
    return status;
}

void RingEnginePy::push_record_descriptors(
    std::vector<ring::RecordDescriptor> descriptors) {
    if (!impl_->record_mode) {
        throw std::logic_error("record descriptors require a record ring");
    }
    impl_->engine.record_consumer().push_descriptors(std::move(descriptors));
}

void RingEnginePy::submit_record_cpu_direct(
    at::Tensor cpu_tensor, uint64_t tensor_bytes) {
    if (!impl_->record_mode) {
        throw std::logic_error("record CPU submission requires a record ring");
    }
    impl_->engine.drain_thread().submit_cpu_direct(
        std::move(cpu_tensor), tensor_bytes);
}

bool RingEnginePy::flush_records_and_wait(uint64_t timeout_ms) {
    if (!impl_->record_mode) {
        throw std::logic_error("record flush requires a record ring");
    }
    const auto deadline = record_flush_deadline(timeout_ms);
    auto& drain = impl_->engine.drain_thread();
    auto& consumer = impl_->engine.record_consumer();
    drain.rethrow_drain_failure();
    drain.rethrow_record_reclaim_failure();
    // Under kDisableCapture a latched failure is reported only once the
    // drain has delivered what the forward already emitted: the consumer
    // discards it, so ring and staging space come back and the discard
    // counters are complete when the caller hears why capture stopped.
    if (consumer.failed() &&
        impl_->record_options.failure_policy ==
            ring::RecordFailurePolicy::kDisableCapture) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        if (wait_for_stream_prefix_until(stream, deadline)) {
            drain.force_flush_and_wait_until(deadline);
        }
    }
    consumer.rethrow_if_failed();
    if (const std::exception_ptr failure = impl_->pending_step_failure()) {
        std::rethrow_exception(failure);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (!wait_for_stream_prefix_until(stream, deadline)) return false;
    if (!drain.force_flush_and_wait_until(deadline)) return false;
    drain.rethrow_record_reclaim_failure();

    drain.apply_pending_record_reclaims();
    if (drain.pending_record_reclaims() != 0) {
        throw std::runtime_error(
            "record flush found incomplete producer reclaims");
    }

    const auto now = FlushClock::now();
    const auto remaining = now < deadline
        ? std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
        : std::chrono::milliseconds::zero();
    if (!consumer.wait_until_idle(remaining)) return false;
    consumer.finish();

    const auto sink = impl_->engine.record_sink();
    if (sink) {
        try {
            sink->rethrow_if_failed();
            const auto before_sink = FlushClock::now();
            if (before_sink >= deadline) return false;
            const auto sink_timeout =
                std::chrono::duration_cast<ring::RecordSink::Duration>(
                    deadline - before_sink);
            if (!sink->flush_and_wait(sink_timeout)) return false;
            sink->rethrow_if_failed();
        } catch (...) {
            // A sink failure found at the barrier (a pipeline failure, a
            // record lost after admission) is as much a refusal as one at
            // submit: it latches, so capture stops and says why.
            consumer.record_failure(std::current_exception());
            throw;
        }
    }
    if (FlushClock::now() > deadline) return false;
    return true;
}

void RingEnginePy::submit_cpu_direct(at::Tensor cpu_tensor, uint64_t tensor_bytes) {
    // Record rings submit through submit_record_cpu_direct.
    refuse_on_record_ring(impl_->record_mode, "legacy CPU submission");
    impl_->engine.drain_thread().submit_cpu_direct(std::move(cpu_tensor), tensor_bytes);
}

// ---------------------------------------------------------------------------
// Capacity queries (startup only, not per-step)
// ---------------------------------------------------------------------------
uint64_t RingEnginePy::payload_cap() const {
    return impl_->engine.payload_cap();
}

uint64_t RingEnginePy::staging_cap() const {
    return impl_->engine.staging_cap();
}

uint64_t RingEnginePy::task_cap() const {
    return impl_->engine.task_cap();
}

at::Tensor RingEnginePy::payload_tensor() const {
    return impl_->payload_view;
}

// ---------------------------------------------------------------------------
// Runtime queries / actions used by the safety-net branch in
// HookPoint.forward.  All three are called only when force_eager is active
// (eager mode); never run during CUDA-graph capture or replay.
//
// Thread safety of the check-and-reserve pattern used by the safety net:
//
//   if nbytes <= available_capacity():
//       reserve_one(nbytes)
//
// The main thread (this thread) is the only writer of cpu_payload_head_
// (it advances only through reserve / reserve_one calls).  The drain
// thread only ever advances cpu_payload_tail_committed_ forward as it
// frees ring space.  Between the check and the reserve:
//   - tail may move forward (drain freed more): actual available at
//     reserve time is >= what we observed.
//   - head is unchanged (single-threaded writer).
// So the check's "fits" decision remains valid at reserve time.  No extra
// locking around the pair is required.
//
// Within available_capacity(), the two accessor calls happen under
// separate mutex acquires (drain.cpu_payload_head() and
// drain.cpu_payload_tail_committed() each take mgmt_mu_ internally).
// The observed snapshot is non-atomic: if drain advances tail between
// the two reads, available_observed = pcap - head + tail_later, which
// is >= the true available at the time of the head read.  That is, the
// non-atomicity errs on the "over-estimate available" side -- the
// reserve will still succeed because the actual ring state has at least
// as much room as we computed.
// ---------------------------------------------------------------------------

uint64_t RingEnginePy::available_capacity() const {
    auto& drain = impl_->engine.drain_thread();
    const uint64_t pcap = impl_->engine.payload_cap();
    return pcap - (drain.cpu_payload_head() - drain.cpu_payload_tail_committed());
}

// Per-hook reservation: claim nbytes of payload + 1 task entry for an
// upcoming producer kernel launch.  Caller must have checked
// available_capacity() first.  drain.reserve takes mgmt_mu_ internally.
void RingEnginePy::reserve_one(uint64_t nbytes) {
    refuse_on_record_ring(impl_->record_mode, "legacy per-hook reservation");
    impl_->engine.drain_thread().reserve(
        ring::align_up(nbytes, ring::PAYLOAD_ALIGN), 1);
}

// Synchronise the current CUDA stream so all queued producer kernels
// finish writing, then force the drain thread to flush all outstanding
// task entries through the consumer pipeline.  Blocking call; the
// Python binding releases the GIL.
void RingEnginePy::flush_and_wait() {
    cudaStream_t ms = at::cuda::getCurrentCUDAStream().stream();
    cudaStreamSynchronize(ms);
    impl_->engine.drain_thread().force_flush_and_wait();
}

}  // namespace ring_py
