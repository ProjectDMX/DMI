#ifndef DMX_HOST_ENGINE__
#define DMX_HOST_ENGINE__

#include "dmx_host_utils.h"
#include "pipelined_engine.hpp"

namespace dmx_host{

// DMXHostEngine is a single-stage ClickHouse insert pipeline.
// Pre-assembled ClickHouseRows are submitted via submit_direct().
class DMXHostEngine : public PipelinedEngine<dmx_host_queue_item, uint64_t, 1, QueueOptions<false, false, false>, false,
NoOutputHandler<dmx_host_queue_item> >{
public:
    explicit DMXHostEngine(StageConfig insert_stage):
    PipelinedEngine(std::array<StageConfig, 1>{std::move(insert_stage)}, EngineConfig{}){}

    // Submit a pre-assembled ClickHouseRow directly to the insert stage.
    // Fields must match the order expected by ClickHouseInsertStage:
    //   [0] model_id     (string)
    //   [1] request_id   (string)
    //   [2] act_name     (string)
    //   [3] layer_no     (int32)
    //   [4] shard_rank   (int32)
    //   [5] start_token  (int32)
    //   [6] end_token    (int32)
    //   [7] tensor       (at::Tensor, contiguous CPU)
    void submit_direct(ClickHouseRow row, uint64_t nbytes) {
        std::vector<dmx_host_queue_item> items;
        items.emplace_back(std::move(row), nbytes);
        submit_items(std::move(items));
    }
};

}

#endif
