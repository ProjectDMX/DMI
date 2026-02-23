#include "dmx_host_utils.h"

namespace dmx_host{
    // ---------------------------------------------------------
    // Helper Functions
    // ---------------------------------------------------------
    std::vector<std::string> split_string(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    // Helper to check if a string starts with a prefix (C++20 has .starts_with())
    bool starts_with(const std::string& str, const std::string& prefix) {
        return str.size() >= prefix.size() && 
            str.compare(0, prefix.size(), prefix) == 0;
    }

    // Helper to check if a string ends with a suffix (C++20 has .ends_with())
    bool ends_with(const std::string& str, const std::string& suffix) {
        return str.size() >= suffix.size() && 
            str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    // ---------------------------------------------------------
    // Main Functions
    // ---------------------------------------------------------

    /**
     * Parses internal ID.
     * Returns: A pair containing {int32_t, string}
     */
    std::pair<int32_t, std::string> parse_internal_id(const std::string& internal_id) {
        if (starts_with(internal_id, "blocks.")) {
            std::vector<std::string> internal_id_list = split_string(internal_id, '.');
            
            // Ensure the split has enough parts to prevent out-of-bounds access
            if (internal_id_list.size() < 2) {
                return {-1, internal_id};
            }

            int32_t block_num = std::stoi(internal_id_list[1]);

            // Reconstruct the string: "blocks" + elements from index 2 onwards
            std::string result_str = "blocks";
            for (size_t i = 2; i < internal_id_list.size(); ++i) {
                result_str += "." + internal_id_list[i];
            }

            return {block_num, result_str};
        } else {
            return {-1, internal_id};
        }
    }

    /**
     * Gets delta token length.
     * shape: Passed as a vector<int64_t> to represent the tuple
     */
    int64_t get_delta_token_len(const std::vector<int64_t>& shape, const std::string& act_name) {
        if (ends_with(act_name, "attn.hook_attn_scores") || 
            ends_with(act_name, "attn.hook_pattern")) {
            // Corresponds to shape[2]
            return shape[2];
        } else {
            // Corresponds to shape[1]
            return shape[1];
        }
    }

    std::vector<dmx_host_queue_item> input_handler_v1(std::vector<std::vector<std::string> > keys, std::vector<int32_t> start_token_idxs, 
    std::vector<std::map<std::string, monitoring::BackendFuture> > cache_dicts){
        std::vector<dmx_host_queue_item> outputs;
        size_t key_cnt = keys.size();
        if(key_cnt != start_token_idxs.size() || cache_dicts.size() != key_cnt){
            throw std::runtime_error("In input_handler_v1, keys.size(), start_token_idxs.size(), cache_dicts.size() don't match.");
        }
        for(size_t i = 0; i < key_cnt; ++i){
            if(keys[i].size() != 2){
                throw std::runtime_error("In input_handler_v1, every key should be (model_id, request_id)");
            }
            size_t tcnt = cache_dicts[i].size();
            for (const auto& [name, t_future] : cache_dicts[i]) {
                FutureProcessRow fr;
                fr.push_back(keys[i][0]);
                fr.push_back(keys[i][1]);
                fr.push_back(start_token_idxs[i]);
                fr.push_back(name);
                fr.push_back(t_future);
                dmx_host_queue_item out_item(fr, t_future.size());
                outputs.push_back(out_item);
            }
        }
        return outputs;
    }
}