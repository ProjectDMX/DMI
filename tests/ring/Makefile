NVCC     ?= /usr/local/cuda/bin/nvcc
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
DRAIN_SRCS   := ../../monitoring/csrc/ring/drain_thread.cpp \
                ../../monitoring/csrc/ring/chunk_assembler.cpp

TARGETS := $(BUILD)/test_rings $(BUILD)/test_producer $(BUILD)/test_ring_engine $(BUILD)/test_null_mode

.PHONY: all test_rings test_producer test_ring_engine run clean

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_rings: test_rings.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/test_producer: test_producer.cu $(PRODUCER_CU) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(BUILD)/test_ring_engine: test_ring_engine.cu $(PRODUCER_CU) $(DRAIN_SRCS) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -lpthread $^ -o $@

test_rings: $(BUILD)/test_rings
	$(BUILD)/test_rings

test_producer: $(BUILD)/test_producer
	$(BUILD)/test_producer

$(BUILD)/test_null_mode: test_null_mode.cu $(PRODUCER_CU) $(DRAIN_SRCS) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) -lpthread $^ -o $@

test_ring_engine: $(BUILD)/test_ring_engine
	$(BUILD)/test_ring_engine

test_null_mode: $(BUILD)/test_null_mode
	$(BUILD)/test_null_mode

run: all
	$(BUILD)/test_rings
	$(BUILD)/test_producer
	$(BUILD)/test_ring_engine
	$(BUILD)/test_null_mode

clean:
	rm -rf $(BUILD)
