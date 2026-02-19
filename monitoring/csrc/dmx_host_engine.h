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
    void submit(std::vector<std::vector<std::string> > keys, std::vector<int32_t> start_token_idxs, 
    std::vector<std::map<std::string, monitoring::BackendFuture> > cache_dicts){
        submit_items(input_handler_v1(keys, start_token_idxs, cache_dicts));
    }
};

}

#endif