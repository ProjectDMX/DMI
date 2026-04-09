NVCC     ?= $(shell command -v nvcc 2>/dev/null || printf '%s' nvcc)
SM_ARCH  ?= native
CXX_STD  ?= c++17
BUILD    := build
INCLUDES := -I../.. -I../../monitoring/csrc

NVCC_FLAGS := \
    -std=$(CXX_STD) \
    -arch=$(SM_ARCH) \
    -O2 \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    -Xcompiler -Wno-unused-parameter \
    -lineinfo \
    $(INCLUDES)

PRODUCER_CU  := ../../monitoring/csrc/ring/producer.cu
DRAIN_SRC    := ../../monitoring/csrc/ring/drain_thread.cpp

# test_ring_engine needs ATen headers + libtorch for DrainTask (contains at::Tensor).
# Use torch.utils.cmake_prefix_path to locate them.  The Python E2E tests
# (tests/test_e2e_correctness_vs_hf.py) cover the full pipeline including p2p.
TORCH_INC := $(shell python -c "import torch; print(torch.utils.cmake_prefix_path + '/../../include')" 2>/dev/null)
TORCH_LIB := $(shell python -c "import torch; print(torch.utils.cmake_prefix_path + '/../../lib')" 2>/dev/null)

TARGETS := $(BUILD)/test_rings $(BUILD)/test_producer $(BUILD)/test_ring_engine

.PHONY: all test_rings test_producer test_ring_engine run clean

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_rings: test_rings.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/test_producer: test_producer.cu $(PRODUCER_CU) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(BUILD)/test_ring_engine: test_ring_engine.cu $(PRODUCER_CU) $(DRAIN_SRC) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -I$(TORCH_INC) -I$(TORCH_INC)/torch/csrc/api/include -L$(TORCH_LIB) -Xlinker -rpath,$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -lpthread $^ -o $@

test_rings: $(BUILD)/test_rings
	$(BUILD)/test_rings

test_producer: $(BUILD)/test_producer
	$(BUILD)/test_producer

test_ring_engine: $(BUILD)/test_ring_engine
	$(BUILD)/test_ring_engine

run: all
	$(BUILD)/test_rings
	$(BUILD)/test_producer
	$(BUILD)/test_ring_engine

clean:
	rm -rf $(BUILD)
