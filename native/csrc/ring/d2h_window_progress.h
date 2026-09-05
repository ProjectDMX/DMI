#pragma once

#include "d2h_window_progress_layout.h"

#include <ATen/ATen.h>
#include <cuda_runtime.h>

namespace ring {

struct D2HWindowProgressSnapshot {
    D2HWindowPackedProgressLayout::Version version{
        D2HWindowPackedProgressLayout::kNoPatternVersion};
    D2HWindowPackedProgressLayout::Counter counter{0};
};

struct D2HWindowProgressState {
    at::Tensor device_packed_progress;
    at::Tensor cpu_visible_packed_progress;
};

class D2HWindowProgressSource {
  public:
    virtual ~D2HWindowProgressSource() = default;
    virtual D2HWindowProgressSnapshot load() const noexcept = 0;
    virtual D2HWindowProgressState state() const = 0;
    virtual void enqueue_reset(D2HWindowPackedProgressLayout::Version version,
                               D2HWindowPackedProgressLayout::Counter counter,
                               cudaStream_t stream) = 0;
};

class PackedVersionCounterProgressSource final : public D2HWindowProgressSource {
  public:
    explicit PackedVersionCounterProgressSource(int owner_device);
    ~PackedVersionCounterProgressSource() override;

    PackedVersionCounterProgressSource(const PackedVersionCounterProgressSource&) =
        delete;
    PackedVersionCounterProgressSource&
    operator=(const PackedVersionCounterProgressSource&) = delete;

    D2HWindowProgressSnapshot load() const noexcept override;
    D2HWindowProgressState state() const override;
    void enqueue_reset(D2HWindowPackedProgressLayout::Version version,
                       D2HWindowPackedProgressLayout::Counter counter,
                       cudaStream_t stream) override;

  private:
    int owner_device_{-1};
    D2HWindowPackedProgressLayout::Word* device_packed_progress_{nullptr};
    D2HWindowPackedProgressLayout::Word* cpu_visible_packed_progress_{nullptr};
    at::Tensor device_view_;
    at::Tensor cpu_visible_view_;
};

}  // namespace ring
