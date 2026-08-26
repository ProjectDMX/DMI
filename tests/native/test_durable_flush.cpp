#include "pipelined_engine.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

struct Item {
  std::uint64_t value = 0;
  std::uint64_t size() const { return sizeof(value); }
};

using Engine = dmx_host::PipelinedEngine<Item, std::uint64_t, 1>;

Engine::StageConfig MakeBufferedStage(std::vector<std::uint64_t>* observed,
                                      std::mutex* observed_mu) {
  Engine::StageConfig stage;
  stage.name = "sink";
  stage.parallelism = 2;
  stage.input_queue.min_batch_items.reset();
  stage.input_queue.min_batch_size = 1U << 20;
  stage.input_queue.max_linger.reset();
  stage.process_fn = [observed, observed_mu](
      std::vector<Item> batch, Engine::QueueT*)
      -> std::optional<std::vector<Item>> {
    std::lock_guard<std::mutex> lock(*observed_mu);
    for (const auto& item : batch) observed->push_back(item.value);
    return std::vector<Item>{};
  };
  return stage;
}

void CheckNonterminalDurableFlush() {
  std::vector<std::uint64_t> observed;
  std::mutex observed_mu;
  Engine engine(
      std::array<Engine::StageConfig, 1>{
          MakeBufferedStage(&observed, &observed_mu)});
  engine.start();

  engine.submit_items({Item{1}, Item{2}, Item{3}});
  assert(engine.flush_and_wait(Engine::Duration(1.0)));
  {
    std::lock_guard<std::mutex> lock(observed_mu);
    assert(observed.size() == 3);
  }

  // A successful flush is nonterminal: later submission and another flush
  // must use the same workers and queue.
  engine.submit_items({Item{4}});
  assert(engine.flush_and_wait(Engine::Duration(1.0)));
  {
    std::lock_guard<std::mutex> lock(observed_mu);
    assert(observed.size() == 4);
  }
  assert(engine.stop(true, Engine::Duration(1.0)));
}

void CheckWorkerFailureReachesFlushCaller() {
  Engine::StageConfig stage;
  stage.name = "failing_sink";
  stage.parallelism = 1;
  stage.process_fn = [](std::vector<Item>, Engine::QueueT*)
      -> std::optional<std::vector<Item>> {
    throw std::runtime_error("injected durable sink failure");
  };

  Engine engine(std::array<Engine::StageConfig, 1>{std::move(stage)});
  engine.start();
  engine.submit_items({Item{1}});
  bool raised = false;
  try {
    (void)engine.flush_and_wait(Engine::Duration(1.0));
  } catch (const std::runtime_error&) {
    raised = true;
  }
  assert(raised);
  assert(!engine.failures().empty());
}

void CheckDeadline() {
  Engine::StageConfig stage;
  stage.name = "slow_sink";
  stage.parallelism = 1;
  stage.process_fn = [](std::vector<Item>, Engine::QueueT*)
      -> std::optional<std::vector<Item>> {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    return std::vector<Item>{};
  };

  Engine engine(std::array<Engine::StageConfig, 1>{std::move(stage)});
  engine.start();
  engine.submit_items({Item{1}});
  assert(!engine.flush_and_wait(Engine::Duration(0.001)));
}

}  // namespace

int main() {
  CheckNonterminalDurableFlush();
  CheckWorkerFailureReachesFlushCaller();
  CheckDeadline();
  return 0;
}
