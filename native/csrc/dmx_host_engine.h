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

    void validate_record_schema(const RecordSchema& schema) const {
        if (!record_stage_) {
            throw std::logic_error(
                "record runtime requires a schema-driven record stage");
        }
        if (RecordSchemaIdentity(schema) != record_schema_identity_) {
            throw std::invalid_argument(
                "record runtime schema does not match the host record stage");
        }
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
        Base::submit_items(std::move(items));
    }

    void submit_record(GenericRecordRow row, uint64_t nbytes) {
        std::vector<dmx_host_queue_item> items;
        items.emplace_back(std::move(row), nbytes);
        Base::submit_items(std::move(items));
    }

    bool flush_and_wait(Duration timeout) {
        if (timeout.count() < 0.0) {
            throw std::invalid_argument("timeout must be non-negative");
        }
        if (!record_stage_) {
            throw std::logic_error(
                "durable record flush requires a schema-driven record stage");
        }
        return Base::flush_and_wait(timeout);
    }

private:
    struct PreparedStage {
        StageConfig stage;
        std::shared_ptr<ClickHouseRuntimeMetrics> metrics;
        bool record_stage{false};
        std::string record_schema_identity;
    };

    static PreparedStage Prepare(StageConfig stage) {
        if (stage.parallelism <= 0) {
            throw std::invalid_argument("parallelism must be positive");
        }
        auto metrics =
            std::make_shared<ClickHouseRuntimeMetrics>(stage.parallelism);
        bool record_stage = false;
        std::string record_schema_identity;
        if (stage.thread_init_config.type() == typeid(ClickHouseClientConfig)) {
            auto config =
                std::any_cast<ClickHouseClientConfig>(stage.thread_init_config);
            config.runtime_metrics = metrics;
            stage.thread_init_config = std::move(config);
        } else if (stage.thread_init_config.type() ==
                   typeid(ClickHouseRecordStageConfig)) {
            record_stage = true;
            auto config =
                std::any_cast<ClickHouseRecordStageConfig>(
                    stage.thread_init_config);
            record_schema_identity = RecordSchemaIdentity(config.schema);
            config.client.runtime_metrics = metrics;
            stage.thread_init_config = std::move(config);
        } else {
            throw std::invalid_argument(
                "unsupported DMXHostEngine stage initialization config");
        }
        return PreparedStage{
            std::move(stage), std::move(metrics), record_stage,
            std::move(record_schema_identity)};
    }

    explicit DMXHostEngine(PreparedStage prepared)
        : Base(std::array<StageConfig, 1>{std::move(prepared.stage)}, EngineConfig{}),
          metrics_(std::move(prepared.metrics)),
          record_stage_(prepared.record_stage),
          record_schema_identity_(
              std::move(prepared.record_schema_identity)) {}

    std::shared_ptr<ClickHouseRuntimeMetrics> metrics_;
    bool record_stage_{false};
    std::string record_schema_identity_;
};

}

#endif
