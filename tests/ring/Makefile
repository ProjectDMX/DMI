NVCC     ?= /usr/local/cuda/bin/nvcc
SM_ARCH  ?= native
CXX_STD  ?= c++17
BUILD    := build
TARGET   := $(BUILD)/test_rings

SRCS     := test_rings.cu
INCLUDES := -I../.. -I../../monitoring/csrc

NVCC_FLAGS := \
    -std=$(CXX_STD) \
    -arch=$(SM_ARCH) \
    -O2 \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    -Xcompiler -Wno-unused-parameter \
    $(INCLUDES)

# Enable debug info (optional; comment out for pure perf builds).
NVCC_FLAGS += -lineinfo

.PHONY: all run clean

all: $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET): $(SRCS) | $(BUILD)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: $(TARGET)
	$(TARGET)

clean:
	rm -rf $(BUILD)
