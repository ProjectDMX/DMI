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

TARGETS := $(BUILD)/test_rings $(BUILD)/test_producer

.PHONY: all test_rings test_producer run clean

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_rings: test_rings.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/test_producer: test_producer.cu | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

test_rings: $(BUILD)/test_rings
	$(BUILD)/test_rings

test_producer: $(BUILD)/test_producer
	$(BUILD)/test_producer

run: all
	$(BUILD)/test_rings
	$(BUILD)/test_producer

clean:
	rm -rf $(BUILD)
