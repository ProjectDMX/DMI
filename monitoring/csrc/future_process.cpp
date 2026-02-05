#include "future_process.h"
#include "batching_queue.hpp"   // WatermarkBatchingQueue
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace dmx_host {
thread_local int worker_id;
void ProcessFutureStage::ThreadInit(int thread_idx){
    worker_id = thread_idx;
}
void ProcessFutureStage::ThreadCleanup() noexcept{

}

template <typename QueueT>
std::optional<std::vector<dmx_host_queue_item>> ProcessFutureStage::ProcessFuture(std::vector<dmx_host_queue_item>&& batch, QueueT* next_q){
    //In this function, push as soon as possible, do not return to engine for pushing.
    for (const auto& r : batch) {
        const auto& future_row = std::get<FutureProcessRow>(r.core);
        if(future_row.size() != 5){
            throw std::invalid_argument("FutureProcessRow must have exactly 5 cells: [model_id,request_id,start,act_name,tensor_future]");
        }
        std::string model_id = std::get<std::string>(future_row[0]);
        std::string request_id = std::get<std::string>(future_row[1]);
        int32_t start_token_idx = std::get<int32_t>(future_row[2]);
        std::string act_name = std::get<std::string>(future_row[3]);
        monitoring::BackendFuture backend_future = std::get<monitoring::BackendFuture>(future_row[4]);
        auto parsed_pair = parse_internal_id(act_name);
        int32_t layer_no = parsed_pair.first;
        std::string parsed_act_name = parsed_pair.second;
        at::Tensor tensor = backend_future.result(std::optional<double>(), true);
        ClickHouseRow new_row;
        new_row.push_back(model_id);
        new_row.push_back(request_id);
        new_row.push_back(parsed_act_name);
        new_row.push_back(layer_no);
        new_row.push_back(start_token_idx);
        int64_t delta_len = get_delta_token_len(tensor.sizes().vec(), parsed_act_name);
        int32_t delta_len_32 = 0;
        if (delta_len > std::numeric_limits<int32_t>::max() || 
            delta_len < std::numeric_limits<int32_t>::min()) {
            // Handle error: value is too big!
            throw std::overflow_error("Value too large for int32_t");
        } else {
            delta_len_32 = static_cast<int32_t>(delta_len);
        }
        if (delta_len < 0){
            throw std::overflow_error("Value < 0 for get_delta_token_len");
        }
        int32_t end_token_idx = start_token_idx + delta_len_32;
        new_row.push_back(end_token_idx);
        new_row.push_back(tensor);
        next_q->enqueue(dmx_host_queue_item(std::move(new_row), tensor.nbytes()));
    }
    return std::nullopt;
}

// Force emission of the specialization that bindings.cpp calls (QueueT == WatermarkBatchingQueue<...>).
template std::optional<std::vector<dmx_host_queue_item> >
ProcessFutureStage::ProcessFuture<WatermarkBatchingQueue<dmx_host_queue_item, uint64_t, false, false, false>
>(std::vector<dmx_host_queue_item>&&,
  WatermarkBatchingQueue<dmx_host_queue_item, uint64_t, false, false, false>*);


}
