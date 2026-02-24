#ifndef DMX_HOST_ENGINE__
#define DMX_HOST_ENGINE__

#include "dmx_host_utils.h"
#include "pipelined_engine.hpp"

namespace dmx_host{

class DMXHostEngine : public PipelinedEngine<dmx_host_queue_item, uint64_t, 2, QueueOptions<false, false, false>, false, 
NoOutputHandler<dmx_host_queue_item> >{
public:
    DMXHostEngine(std::array<StageConfig, 2> two_stages): 
    PipelinedEngine (two_stages, EngineConfig{}){
    }

    // submit a batch of runs(batch of batch).
    void submit(std::string model_id, int32_t shard_rank, 
        std::vector< std::vector<std::string> > request_ids,
        std::vector<std::vector<std::pair<int32_t, int32_t> > > token_range_per_request, 
        std::vector< std::map<std::string, monitoring::BackendFuture> > > cache_dicts){
            submit_items(input_handler_v1(model_id, shard_rank, request_ids, token_range_per_request, cache_dicts));
    }
};

}

#endif