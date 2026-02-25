#include "future_process.h"
#include "batching_queue.hpp"   // WatermarkBatchingQueue
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace dmx_host {
thread_local int worker_id;
thread_local int contiguous_introduced_copy = 0;

void ProcessFutureStage::ThreadInit(int thread_idx){
    worker_id = thread_idx;
    contiguous_introduced_copy = 0;
}
void ProcessFutureStage::ThreadCleanup() noexcept{
    std::cout << "worker " << worker_id
              << " in future_process introduced "
              << contiguous_introduced_copy
              << " tensor copies due to non-contiguous" << std::endl;
}

template <typename QueueT>
std::optional<std::vector<dmx_host_queue_item>> ProcessFutureStage::ProcessFuture(
    std::vector<dmx_host_queue_item>&& batch,
    QueueT* next_q){
    // In this function, push as soon as possible, do not return to engine for pushing.
    for (const auto& r : batch) {
        const auto& future_row = std::get<FutureProcessRow>(r.core);
        if(future_row.size() != 6){
            throw std::invalid_argument(
                "FutureProcessRow must have exactly 6 cells: "
                "[model_id, shard_rank, request_ids, token_ranges, act_name, tensor_future]");
        }
        std::string model_id = std::get<std::string>(future_row[0]);
        int32_t shard_rank = std::get<int32_t>(future_row[1]);
        const auto& request_ids = std::get<std::vector<std::string>>(future_row[2]);
        const auto& token_ranges =
            std::get<std::vector<std::pair<int32_t, int32_t>>>(future_row[3]);
        std::string act_name = std::get<std::string>(future_row[4]);
        monitoring::BackendFuture backend_future = std::get<monitoring::BackendFuture>(future_row[5]);
        auto parsed_pair = parse_internal_id(act_name);
        int32_t layer_no = parsed_pair.first;
        std::string parsed_act_name = parsed_pair.second;
        at::Tensor tensor = backend_future.result(std::optional<double>(), true);

        if (!tensor.defined()) {
            continue;
        }
        if (tensor.dim() < 1) {
            std::cerr << "[FutureProcess] skip row: tensor dim < 1 for act_name="
                      << parsed_act_name << std::endl;
            continue;
        }
        if (request_ids.size() != token_ranges.size()) {
            std::cerr << "[FutureProcess] skip row: request_ids.size() != token_ranges.size()"
                      << std::endl;
            continue;
        }

        const int64_t bsz = tensor.size(0);
        if (bsz < 0) {
            std::cerr << "[FutureProcess] skip row: tensor batch size < 0" << std::endl;
            continue;
        }
        if (static_cast<size_t>(bsz) != request_ids.size()) {
            std::cerr << "[FutureProcess] skip row: tensor.size(0)=" << bsz
                      << " request_ids.size()=" << request_ids.size()
                      << " act_name=" << parsed_act_name << std::endl;
            continue;
        }

        auto ends_with = [](const std::string& s, const std::string& suf) -> bool {
            return s.size() >= suf.size() &&
                s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
        };
        const bool is_attn =
            ends_with(parsed_act_name, "attn.hook_attn_scores") ||
            ends_with(parsed_act_name, "attn.hook_pattern");

        for (size_t j = 0; j < request_ids.size(); ++j) {
            const auto [start_token_idx, end_token_idx] = token_ranges[j];
            if (start_token_idx >= end_token_idx) {
                continue;  // Empty range for finished/pad-only request.
            }
            if (start_token_idx < 0 || end_token_idx < 0) {
                std::cerr << "[FutureProcess] skip request: negative token range ("
                          << start_token_idx << ", " << end_token_idx << ")" << std::endl;
                continue;
            }

            int32_t eff_token_len = end_token_idx - start_token_idx;
            if (eff_token_len <= 0) {
                continue;
            }
            /**
             * @brief 
             * Assumption: huggingface generate().
             * 1. For prefill, always left-padded.
             * 2. For decoding, append tokens to right, only PAD again after finishes.
             * 3. One batch is all prefill or all decode.
             * ==> So always suffix.
             */

            at::Tensor slice = tensor.select(0, static_cast<int64_t>(j));
            bool did_slice = false;

            // delta token length (batch dim already removed in slice)
            const int64_t token_dim = is_attn ? ((slice.dim() >= 2) ? (slice.dim() - 2) : 0) : 0;
            const int64_t delta_token_len =
                (slice.defined() && slice.dim() > token_dim) ? slice.size(token_dim) : 0;

            if (delta_token_len > static_cast<int64_t>(eff_token_len)) {
                // Always suffix.
                int64_t skip_first = delta_token_len - static_cast<int64_t>(eff_token_len);
                slice = slice.narrow(/*dim=*/token_dim, /*start=*/skip_first,
                                     /*length=*/static_cast<int64_t>(eff_token_len));
                did_slice = true;
            }

            if (is_attn && slice.defined() && slice.dim() >= 2) {
                const int64_t key_dim = slice.dim() - 1;
                const int64_t want_k = static_cast<int64_t>(end_token_idx);
                if (want_k >= 0 && slice.size(key_dim) > want_k) {
                    int64_t skip_first = slice.size(key_dim) - want_k;
                    slice = slice.narrow(/*dim=*/key_dim, /*start=*/skip_first, /*length=*/want_k);
                    did_slice = true;
                }
            }

            if (did_slice && !slice.is_contiguous()) {
                slice = slice.contiguous();
                contiguous_introduced_copy++;
            }

            ClickHouseRow new_row;
            new_row.push_back(model_id);
            new_row.push_back(request_ids[j]);
            new_row.push_back(parsed_act_name);
            new_row.push_back(layer_no);
            new_row.push_back(shard_rank);
            new_row.push_back(start_token_idx);
            new_row.push_back(end_token_idx);
            new_row.push_back(slice);
            next_q->enqueue(dmx_host_queue_item(std::move(new_row), slice.nbytes()));
        }
    }
    return std::nullopt;
}

// Force emission of the specialization that bindings.cpp calls (QueueT == WatermarkBatchingQueue<...>).
template std::optional<std::vector<dmx_host_queue_item> >
ProcessFutureStage::ProcessFuture<WatermarkBatchingQueue<dmx_host_queue_item, uint64_t, false, false, false>
>(std::vector<dmx_host_queue_item>&&,
  WatermarkBatchingQueue<dmx_host_queue_item, uint64_t, false, false, false>*);


}
