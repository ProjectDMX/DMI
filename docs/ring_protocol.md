# Ring Offload Protocol

Publish/consume protocol, strict vs timeout-drop semantics, and chunking + reassembly contract for the CUDA-graph-safe LLM internal-state offload system.

---

## Overview

The offload pipeline uses two preallocated GPU rings to move LLM internal tensors from device to host without host synchronisation inside a CUDA graph:

| Ring | Role |
|------|------|
| **Task ring** (control ring) | Fixed-size FIFO of `TaskEntry` structs; carries metadata and payload descriptors |
| **Payload ring** (byte ring) | Circular byte buffer; holds the raw tensor bytes |

Both rings are preallocated before graph capture and remain at fixed device addresses throughout.

---

## Data Structures

### TaskEntry (128 bytes, 128-byte aligned)

```
offset  0  ready_seq           uint64  sequence guard (SENTINEL until published)
offset  8  seq_no              uint64  monotonic slot index at publish time
offset 16  logical_task_id     uint64  packed {hook_id, chunk_seq, tensor_idx}
offset 24  chunk_offset_bytes  uint64  byte offset of this chunk in the logical tensor
offset 32  tensor_total_bytes  uint64  total logical tensor bytes (valid on IS_FIRST)
offset 40  payload_off1        uint64  first payload span offset in payload_buf
offset 48  payload_len1        uint64  first payload span length
offset 56  payload_off2        uint64  second payload span offset (0 if single span)
offset 64  payload_len2        uint64  second payload span length (0 if single span)
offset 72  chunk_idx           uint32  chunk index within logical task
offset 76  hook_type           uint32  hook classification
offset 80  hook_id             uint32  hook identifier
offset 84  flags               uint32  TASK_FLAG_IS_FIRST | IS_LAST | IS_DROP
offset 88  reason              uint32  DROP_REASON_* (drop entries only)
offset 92  _pad0               uint32
offset 96  _padding[32]        uint8   explicit padding to 128 bytes
```

### PayloadRingState (logical view)

```
payload_buf[payload_ring_bytes]  — circular byte buffer in device memory
payload_head                     — uint64, producer's next write position (unwrapped)
payload_tail                     — uint64, consumer's oldest live byte (unwrapped)
```

Free bytes = `payload_ring_bytes - (payload_head - payload_tail)`.

Physical offset = `logical_position % payload_ring_bytes`.

### Consumer Progress Signal

```
consumer_heartbeat / last_released_seq  — uint64 in device memory
```

Updated by the consumer whenever `task_tail` advances.  The producer reads this to detect liveness in TIMEOUT_DROP mode.

---

## Publish Protocol (producer → consumer)

The producer is a CUDA custom op inserted as a post-op node in the model graph.

### For each chunk of a logical task:

1. **Check space** — spin (or drop) until:
   - `task_free_slots(head, tail, cap) >= 1`
   - `payload_free_bytes(phead, ptail, pcap) >= chunk_bytes`

2. **Compute payload spans** — call `payload_compute_spans(phead, pcap, chunk_bytes)` to get a `TwoSpan {off1, len1, off2, len2}`.  If `len2 == 0` the reservation is contiguous; otherwise it wraps.

3. **Copy data** — issue `cudaMemcpyAsync` (D2D) into `payload_buf[off1..off1+len1]` and, if `len2 > 0`, into `payload_buf[0..len2]`.

4. **Advance payload_head** — `payload_head += len1 + len2`.

5. **Fill TaskEntry fields** — write `seq_no`, `logical_task_id`, spans, flags, etc.

6. **Publish** — `__threadfence()` then write `ready_seq = seq_no`.

7. **Advance task_head** — `task_head++`.

### DROP marker (TIMEOUT_DROP mode):

- Producer does NOT advance `task_head` or `payload_head` until space is confirmed.
- On timeout, publish a `TaskEntry` with `TASK_FLAG_IS_DROP` set and `payload_len1 = payload_len2 = 0`.
- No payload space is reserved; head pointers are not rolled back (rollback-free by design).

---

## Consume Protocol (consumer → host)

The drain loop runs outside the CUDA graph on a dedicated stream or host thread.

### For each slot:

1. **Spin-wait** — loop reading `volatile entries[tail % cap].ready_seq` until it equals `tail`.

2. **Acquire fence** — `__threadfence()` to ensure entry data fields are visible.

3. **Read entry** — copy relevant fields (`logical_task_id`, spans, flags, etc.) to local registers.

4. **Check IS_DROP** — if set, increment drop counter and skip to step 7.

5. **D2H transfer** — issue async `cudaMemcpyAsync` from `payload_buf[off1..off1+len1]` (and span 2 if `len2 > 0`) to the pinned staging buffer.

6. **Advance payload_tail** — `payload_tail += len1 + len2` (after D2H is launched; D2H may still be in-flight).

7. **Release slot** — write `ready_seq = READY_SEQ_SENTINEL`, then `__threadfence()`.

8. **Advance task_tail** — `task_tail++`.

9. **Update heartbeat** — write `consumer_heartbeat = task_tail` (producer reads this).

---

## Backpressure Modes

### Strict (INFINITE) — default

Producer spins in a device-side loop until both:
- At least one task slot is free: `task_free_slots(...) >= 1`
- At least `chunk_bytes` payload bytes are free: `payload_free_bytes(...) >= chunk_bytes`

**No timeout, no drop.**  Safe for steady-state operation when the consumer is alive and draining faster than the producer fills.

### Timeout-drop (TIMEOUT_DROP)

Producer reads `consumer_heartbeat` at the start of each backpressure loop iteration.  If the value has not changed for `no_progress_timeout_cycles` GPU cycles (measured with `clock64()`), the logical task is abandoned:

1. Do NOT advance `task_head` or `payload_head` (rollback-free: nothing was committed).
2. If `drop_reporting == DROP_TASK`: publish a DROP marker entry (one task slot, zero payload).
3. Increment `dropped_timeout_count`.
4. Return without writing the tensor.

**Key invariant**: the producer never leaves partially-written spans in the ring.  Either the full chunk is committed and published, or nothing is committed.

---

## Task Splitting and Chunking

Task splitting is enabled by default.  One logical tensor → N `TaskEntry` chunks.

### Chunk sizing

`chunk_bytes` (default 64 MiB) caps each chunk.  This ensures the producer never needs more than `chunk_bytes` of payload space at once — even if `tensor_total_bytes >> payload_ring_bytes`.

### Multi-chunk sequencing

For a logical tensor of total size T split into N chunks:
- Chunk 0: `IS_FIRST`, `chunk_offset_bytes = 0`, `tensor_total_bytes = T`
- Chunks 1…N-2: neither IS_FIRST nor IS_LAST
- Chunk N-1: `IS_LAST`, `chunk_offset_bytes = (N-1) * chunk_bytes`

`logical_task_id` is the same across all chunks.  `chunk_idx` is the 0-based chunk number.

### Giant tensor streaming (tensor_total_bytes > payload_ring_bytes)

The producer emits chunk 0, then waits (in strict mode) for the consumer to free payload space before emitting chunk 1, and so on.  This allows tensors of unbounded size to be streamed through a fixed-size ring without deadlock, provided the consumer keeps draining.

---

## Reassembly (host pipeline)

The CPU worker thread reassembles chunks into a contiguous pageable tensor:

1. On `IS_FIRST`: allocate a pageable buffer of `tensor_total_bytes` bytes.
2. For each chunk: `memcpy(dst + chunk_offset_bytes, pinned_src, chunk_len)`.
3. On `IS_LAST`: deliver the completed tensor to the user callback/queue.
4. On DROP: discard any partial reassembly state for that `logical_task_id`.

---

## CUDA Graph Constraints

The producer post-op must be **capture-safe**:

| Requirement | How enforced |
|-------------|--------------|
| No allocation during capture | All buffers (rings, pinned pool) preallocated before first capture |
| No event creation during capture | Events preallocated; event handles are stable pointers |
| No host synchronisation | No `cudaDeviceSynchronize` / `cudaStreamSynchronize` in captured code |
| Stable pointers | Ring buffer device addresses never change after allocation |

The drain loop and host pipeline run **outside** the captured graph on their own streams/threads and are not subject to capture constraints.

---

## Sequence Guard Invariant

`ready_seq = READY_SEQ_SENTINEL` (`0xFFFF…FFFF`) means the slot has never been published (or was just released).

`ready_seq = seq_no` (any other value) means the producer has published the entry with that sequence number and all other fields are valid.

Since `seq_no` is a monotonically increasing 64-bit counter, collision with SENTINEL would require ~5.8×10^11 years at 10^9 slots/second.  No wraparound guard is necessary.
