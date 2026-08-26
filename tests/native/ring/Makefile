PYTHON   ?= python
SM_ARCH  ?= native
CXX_STD  ?= c++17
BUILD    := build
INCLUDES := -I../../.. -I../../../native/csrc

CUDA_RESOLVER := $(abspath ../../../native/cuda_toolkit.py)
CUDA_CONFIG_STAMP := $(BUILD)/cuda-toolkit.config
CUDA_RESOLVE_REQUIRED := 1
ifneq ($(strip $(MAKECMDGOALS)),)
ifeq ($(strip $(filter-out clean,$(MAKECMDGOALS))),)
CUDA_RESOLVE_REQUIRED := 0
endif
endif

ifeq ($(CUDA_RESOLVE_REQUIRED),1)
CUDA_CONFIG_RAW := $(strip $(shell \
	$(PYTHON) $(CUDA_RESOLVER) resolve --format record \
		--cudacxx "$(CUDACXX)" \
		--nvcc "$(NVCC)" \
		--cuda-toolkit-root "$(CUDA_TOOLKIT_ROOT)" \
		--cmake-toolkit-root "$(CUDAToolkit_ROOT)" \
		--cuda-home "$(CUDA_HOME)" \
		--cuda-path "$(CUDA_PATH)"))
empty :=
space := $(empty) $(empty)
comma := ,
CUDA_CONFIG := $(subst |,$(space),$(CUDA_CONFIG_RAW))
ifneq ($(words $(CUDA_CONFIG)),9)
$(error CUDA toolkit resolution failed; see the diagnostic above)
endif

CUDA_NVCC := $(word 2,$(CUDA_CONFIG))
CUDA_INCLUDE_FLAGS := -I$(word 3,$(CUDA_CONFIG))
TORCH_INCLUDE_DIRS := $(subst ;,$(space),$(word 7,$(CUDA_CONFIG)))
TORCH_LIBRARY_DIRS := $(subst ;,$(space),$(word 8,$(CUDA_CONFIG)))
TORCH_INCLUDE_FLAGS := $(addprefix -I,$(TORCH_INCLUDE_DIRS))
TORCH_LIB_FLAGS := $(addprefix -L,$(TORCH_LIBRARY_DIRS))
TORCH_RPATH_FLAGS := $(foreach path,$(TORCH_LIBRARY_DIRS),-Xlinker -rpath$(comma)$(path))
TORCH_ABI_FLAG := -D_GLIBCXX_USE_CXX11_ABI=$(word 9,$(CUDA_CONFIG))
endif

NVCC_FLAGS := \
    -std=$(CXX_STD) \
    -arch=$(SM_ARCH) \
    -O2 \
    -Xcompiler -Wall \
    -Xcompiler -Wextra \
    -Xcompiler -Wno-unused-parameter \
    -lineinfo \
    $(TORCH_ABI_FLAG) \
    $(CUDA_INCLUDE_FLAGS) \
    $(INCLUDES)

PRODUCER_CU  := ../../../native/csrc/ring/producer.cu
DRAIN_SRC    := ../../../native/csrc/ring/drain_thread.cpp
P2P_SRC      := ../../../native/csrc/ring/p2p_thread.cpp
RECORD_CONSUMER_SRC := ../../../native/csrc/ring/record_consumer.cpp
RING_ENGINE_CU := ../../../native/csrc/ring/ring_engine.cu
RING_ENGINE_PY_CU := ../../../native/csrc/ring/ring_engine_py.cu

TARGETS := $(BUILD)/test_rings $(BUILD)/test_producer \
           $(BUILD)/test_ring_engine $(BUILD)/test_null_mode \
           $(BUILD)/test_record_consumer

.PHONY: all test_rings test_producer test_ring_engine test_null_mode \
        test_record_consumer run clean FORCE

all: $(TARGETS)

$(BUILD):
	mkdir -p $(BUILD)

FORCE:

$(CUDA_CONFIG_STAMP): FORCE | $(BUILD)
	$(PYTHON) $(CUDA_RESOLVER) write-stamp --output $@ --record "$(CUDA_CONFIG_RAW)"

$(BUILD)/test_rings: test_rings.cu $(CUDA_CONFIG_STAMP) | $(BUILD)
	$(CUDA_NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD)/test_producer: test_producer.cu $(PRODUCER_CU) $(CUDA_CONFIG_STAMP) | $(BUILD)
	$(CUDA_NVCC) $(NVCC_FLAGS) $(filter-out $(CUDA_CONFIG_STAMP),$^) -o $@

$(BUILD)/test_ring_engine: test_ring_engine.cu $(PRODUCER_CU) $(DRAIN_SRC) \
		$(P2P_SRC) $(RECORD_CONSUMER_SRC) $(RING_ENGINE_CU) \
		$(RING_ENGINE_PY_CU) $(CUDA_CONFIG_STAMP) | $(BUILD)
	$(CUDA_NVCC) $(NVCC_FLAGS) $(TORCH_INCLUDE_FLAGS) $(TORCH_LIB_FLAGS) \
		$(filter-out $(CUDA_CONFIG_STAMP),$^) $(TORCH_RPATH_FLAGS) \
		-ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lpthread -o $@

$(BUILD)/test_null_mode: test_null_mode.cu $(PRODUCER_CU) $(CUDA_CONFIG_STAMP) | $(BUILD)
	$(CUDA_NVCC) $(NVCC_FLAGS) $(filter-out $(CUDA_CONFIG_STAMP),$^) -o $@

$(BUILD)/test_record_consumer: test_record_consumer.cpp $(RECORD_CONSUMER_SRC) $(CUDA_CONFIG_STAMP) | $(BUILD)
	$(CUDA_NVCC) $(NVCC_FLAGS) $(TORCH_INCLUDE_FLAGS) $(TORCH_LIB_FLAGS) \
		$(filter-out $(CUDA_CONFIG_STAMP),$^) $(TORCH_RPATH_FLAGS) \
		-ltorch -ltorch_cpu -lc10 -lpthread -o $@

test_rings: $(BUILD)/test_rings
	$(BUILD)/test_rings

test_producer: $(BUILD)/test_producer
	$(BUILD)/test_producer

test_ring_engine: $(BUILD)/test_ring_engine
	$(BUILD)/test_ring_engine

test_null_mode: $(BUILD)/test_null_mode
	$(BUILD)/test_null_mode

test_record_consumer: $(BUILD)/test_record_consumer
	$(BUILD)/test_record_consumer

run: all
	$(BUILD)/test_rings
	$(BUILD)/test_producer
	$(BUILD)/test_ring_engine
	$(BUILD)/test_null_mode
	$(BUILD)/test_record_consumer

clean:
	rm -rf $(BUILD)
