#ifndef DMX_HOST_ENGINE__
#define DMX_HOST_ENGINE__

#include "clickhouse_client.h"
#include "dmx_host_utils.h"
#include "pipelined_engine.hpp"

#include <algorithm>
#include <chrono>

namespace dmx_host{

using DMXHostEngineBase = PipelinedEngine<
    dmx_host_queue_item, uint64_t, 1, QueueOptions<false, false, false>, false,
    NoOutputHandler<dmx_host_queue_item>>;

class DMXHostEngine : public DMXHostEngineBase {
public:
    using Base = DMXHostEngineBase;
    using StageConfig = Base::StageConfig;
    using EngineConfig = Base::EngineConfig;
    using Duration = Base::Duration;

    explicit DMXHostEngine(StageConfig insert_stage)
        : DMXHostEngine(Prepare(std::move(insert_stage))) {}

    bool wait_until_ready(Duration timeout) {
        if (timeout.count() < 0.0) {
            throw std::invalid_argument("timeout must be non-negative");
        }
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(timeout);
        while (std::chrono::steady_clock::now() < deadline) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
            const auto slice = std::min(remaining, std::chrono::milliseconds(50));
            if (metrics_->WaitUntilReady(slice)) return true;
            if (!failures().empty()) return false;
        }
        const auto snapshot = metrics_->Snapshot();
        return snapshot.ready_workers == snapshot.expected_workers;
    }

    ClickHouseMetricsSnapshot clickhouse_metrics() const {
        return metrics_->Snapshot();
    }

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

private:
    struct PreparedStage {
        StageConfig stage;
        std::shared_ptr<ClickHouseRuntimeMetrics> metrics;
    };

    static PreparedStage Prepare(StageConfig stage) {
        if (stage.parallelism <= 0) {
            throw std::invalid_argument("parallelism must be positive");
        }
        auto config =
            std::any_cast<ClickHouseClientConfig>(stage.thread_init_config);
        auto metrics =
            std::make_shared<ClickHouseRuntimeMetrics>(stage.parallelism);
        config.runtime_metrics = metrics;
        stage.thread_init_config = std::move(config);
        return PreparedStage{std::move(stage), std::move(metrics)};
    }

    explicit DMXHostEngine(PreparedStage prepared)
        : Base(std::array<StageConfig, 1>{std::move(prepared.stage)}, EngineConfig{}),
          metrics_(std::move(prepared.metrics)) {}

    std::shared_ptr<ClickHouseRuntimeMetrics> metrics_;
};

}

#endif
