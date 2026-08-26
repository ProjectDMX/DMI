PYTHON ?= python
CXX ?= g++

# Optional toggles
#   LTO=1      -> enable link-time optimization (-flto)
LTO ?= 0

TARGET := _native_backend
HOST_TARGET := _host_backend
SRCS := csrc/bindings.cpp \
        csrc/clickhouse_client.cpp \
        csrc/ring/drain_thread.cpp \
        csrc/ring/p2p_thread.cpp \
        csrc/ring/record_consumer.cpp \
        csrc/ring/ring_torch_op.cpp
HOST_SRCS := csrc/bindings.cpp csrc/clickhouse_client.cpp
HEADERS := csrc/dmx_host_utils.h \
           csrc/clickhouse_client.h csrc/dmx_host_engine.h csrc/record_schema.h \
           csrc/ring/ring_config.h csrc/ring/ring_state.h csrc/ring/ring_alloc.h \
           csrc/ring/task_entry.h csrc/ring/drain_task.h csrc/ring/drain_thread.h \
           csrc/ring/record_descriptor.h csrc/ring/record_consumer.h \
           csrc/ring/p2p_thread.h csrc/ring/pinned_staging.h \
           csrc/ring/ring_engine.h csrc/ring/ring_engine_py.h \
           csrc/ring/ring_torch_op.h csrc/ring/tensor_meta.h
BUILD_DIR := build
HOST_BUILD_DIR := $(BUILD_DIR)/host
OBJS := $(patsubst csrc/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
HOST_OBJS := $(patsubst csrc/%.cpp,$(HOST_BUILD_DIR)/%.o,$(HOST_SRCS))
PROJECT_ROOT := $(abspath $(CURDIR)/..)

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET_SO := $(TARGET)$(EXT_SUFFIX)
LOCAL_TARGET := $(CURDIR)/$(TARGET_SO)
FINAL_TARGET := $(PROJECT_ROOT)/src/dmi/$(TARGET_SO)
HOST_TARGET_SO := $(HOST_TARGET)$(EXT_SUFFIX)
HOST_LOCAL_TARGET := $(CURDIR)/$(HOST_TARGET_SO)
HOST_FINAL_TARGET := $(PROJECT_ROOT)/src/dmi/$(HOST_TARGET_SO)

# Resolve one complete CUDA toolkit. Do not combine a compiler, headers, and
# libraries from unrelated installations: that can produce an extension linked
# against multiple libcudart major versions.
CUDA_RESOLVER := $(CURDIR)/cuda_toolkit.py
CUDA_CONFIG_STAMP := $(BUILD_DIR)/cuda-toolkit.config
CUDA_RESOLVE_REQUIRED := 1
ifneq ($(strip $(MAKECMDGOALS)),)
ifeq ($(strip $(filter-out clean host,$(MAKECMDGOALS))),)
CUDA_RESOLVE_REQUIRED := 0
endif
endif

comma := ,
ifeq ($(CUDA_RESOLVE_REQUIRED),1)
CUDA_INPUT_CUDACXX := $(CUDACXX)
CUDA_INPUT_NVCC := $(NVCC)
CUDA_INPUT_TOOLKIT_ROOT := $(CUDA_TOOLKIT_ROOT)
CUDA_INPUT_CMAKE_TOOLKIT_ROOT := $(CUDAToolkit_ROOT)
CUDA_INPUT_HOME := $(CUDA_HOME)
CUDA_INPUT_PATH := $(CUDA_PATH)

CUDA_CONFIG_RAW := $(strip $(shell \
	$(PYTHON) $(CUDA_RESOLVER) resolve --format record \
		--cudacxx "$(CUDA_INPUT_CUDACXX)" \
		--nvcc "$(CUDA_INPUT_NVCC)" \
		--cuda-toolkit-root "$(CUDA_INPUT_TOOLKIT_ROOT)" \
		--cmake-toolkit-root "$(CUDA_INPUT_CMAKE_TOOLKIT_ROOT)" \
		--cuda-home "$(CUDA_INPUT_HOME)" \
		--cuda-path "$(CUDA_INPUT_PATH)"))
empty :=
space := $(empty) $(empty)
CUDA_CONFIG := $(subst |,$(space),$(CUDA_CONFIG_RAW))
ifneq ($(words $(CUDA_CONFIG)),9)
$(error CUDA toolkit resolution failed; see the diagnostic above)
endif

CUDA_TOOLKIT_ROOT_RESOLVED := $(word 1,$(CUDA_CONFIG))
CUDA_NVCC := $(word 2,$(CUDA_CONFIG))
CUDA_INCLUDE_DIR := $(word 3,$(CUDA_CONFIG))
CUDA_LIBRARY_DIR := $(word 4,$(CUDA_CONFIG))
CUDA_VERSION := $(word 5,$(CUDA_CONFIG))
TORCH_CUDA_VERSION := $(word 6,$(CUDA_CONFIG))
CUDA_EXPECTED_MAJOR := $(word 1,$(subst .,$(space),$(TORCH_CUDA_VERSION)))
TORCH_INCLUDE_DIRS := $(subst ;,$(space),$(word 7,$(CUDA_CONFIG)))
TORCH_LIBRARY_DIRS := $(subst ;,$(space),$(word 8,$(CUDA_CONFIG)))
TORCH_CXX11_ABI := $(word 9,$(CUDA_CONFIG))

CUDA_INCLUDE_FLAGS := -I$(CUDA_INCLUDE_DIR)
CUDA_LIB_FLAGS := -L$(CUDA_LIBRARY_DIR)
TORCH_INCLUDE_FLAGS := $(addprefix -I,$(TORCH_INCLUDE_DIRS))
TORCH_LIB_FLAGS := $(addprefix -L,$(TORCH_LIBRARY_DIRS))
CUDA_RPATH_FLAGS := -Wl$(comma)-rpath$(comma)$(CUDA_LIBRARY_DIR)
TORCH_RPATH_FLAGS := $(foreach path,$(TORCH_LIBRARY_DIRS),-Wl$(comma)-rpath$(comma)$(path))
ABI_FLAG := -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_CXX11_ABI)

$(info DMI CUDA toolkit: $(CUDA_TOOLKIT_ROOT_RESOLVED) (CUDA $(CUDA_VERSION), PyTorch $(TORCH_CUDA_VERSION)))
else ifneq ($(filter host,$(MAKECMDGOALS)),)
TORCH_INCLUDE_DIRS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join(include_paths()))")
TORCH_LIBRARY_DIRS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join(library_paths()))")
TORCH_CXX11_ABI := $(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_INCLUDE_FLAGS := $(addprefix -I,$(TORCH_INCLUDE_DIRS))
TORCH_LIB_FLAGS := $(addprefix -L,$(TORCH_LIBRARY_DIRS))
TORCH_RPATH_FLAGS := $(foreach path,$(TORCH_LIBRARY_DIRS),-Wl$(comma)-rpath$(comma)$(path))
ABI_FLAG := -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_CXX11_ABI)
endif

PYTHON_INCLUDE_FLAGS := $(shell $(PYTHON) -c "import sysconfig; paths = sysconfig.get_paths(); includes = {paths.get('include'), paths.get('platinclude')}; print(' '.join(f'-I{p}' for p in includes if p))")

# Pybind11 ABI flags
PYBIND_FLAGS := $(shell $(PYTHON) -c "import pybind11; print(' '.join([f'-DPYBIND11_COMPILER_TYPE=\\\"{pybind11.get_compiler_type()}\\\"', f'-DPYBIND11_STDLIB=\\\"{pybind11.get_stdlib()}\\\"', f'-DPYBIND11_BUILD_ABI=\\\"{pybind11.get_build_abi()}\\\"']))" 2>/dev/null || echo "")

CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
            -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=$(TARGET) \
            $(ABI_FLAG) $(PYBIND_FLAGS) \
            $(TORCH_INCLUDE_FLAGS) $(PYTHON_INCLUDE_FLAGS) $(CUDA_INCLUDE_FLAGS) -Icsrc \
            -MMD -MP
HOST_CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
                 -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=$(HOST_TARGET) \
                 -DDMI_HOST_ONLY=1 $(ABI_FLAG) $(PYBIND_FLAGS) \
                 $(TORCH_INCLUDE_FLAGS) $(PYTHON_INCLUDE_FLAGS) -Icsrc \
                 -MMD -MP

LDFLAGS := $(TORCH_LIB_FLAGS) $(CUDA_LIB_FLAGS) $(TORCH_RPATH_FLAGS) $(CUDA_RPATH_FLAGS) \
           -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -lcudart -lcuda -lpthread
HOST_LDFLAGS := $(TORCH_LIB_FLAGS) $(TORCH_RPATH_FLAGS) \
                -lc10 -ltorch_cpu -ltorch -ltorch_python -lpthread
ifeq ($(shell uname -s),Darwin)
HOST_LDFLAGS += -Wl,-undefined,dynamic_lookup
endif


# ---- ClickHouse C++ client (fixed in-repo paths) ----
CLICKHOUSE_INCLUDE := $(PROJECT_ROOT)/third_party/clickhouse-cpp
CLICKHOUSE_LIB_DIR := $(CLICKHOUSE_INCLUDE)/build/clickhouse
CLICKHOUSE_LIBS    := -lclickhouse-cpp-lib
CLICKHOUSE_CONTRIB := $(CLICKHOUSE_INCLUDE)/build/contrib

ifneq ($(CLICKHOUSE_INCLUDE),)
CXXFLAGS += -I$(CLICKHOUSE_INCLUDE) -I$(CLICKHOUSE_INCLUDE)/contrib/absl
HOST_CXXFLAGS += -I$(CLICKHOUSE_INCLUDE) -I$(CLICKHOUSE_INCLUDE)/contrib/absl
endif
ifneq ($(CLICKHOUSE_LIB_DIR),)
LDFLAGS += -L$(CLICKHOUSE_LIB_DIR) -Wl,-rpath,$(CLICKHOUSE_LIB_DIR)
HOST_LDFLAGS += -L$(CLICKHOUSE_LIB_DIR) -Wl,-rpath,$(CLICKHOUSE_LIB_DIR)
endif
LDFLAGS += $(CLICKHOUSE_LIBS)
HOST_LDFLAGS += $(CLICKHOUSE_LIBS)
# clickhouse-cpp transitive deps (static libs built by its cmake)
LDFLAGS += -L$(CLICKHOUSE_CONTRIB)/lz4/lz4 -llz4 \
           -L$(CLICKHOUSE_CONTRIB)/cityhash/cityhash -lcityhash \
           -L$(CLICKHOUSE_CONTRIB)/absl/absl -labsl_int128 \
           -L$(CLICKHOUSE_CONTRIB)/zstd/zstd -lzstdstatic
HOST_LDFLAGS += -L$(CLICKHOUSE_CONTRIB)/lz4/lz4 -llz4 \
                -L$(CLICKHOUSE_CONTRIB)/cityhash/cityhash -lcityhash \
                -L$(CLICKHOUSE_CONTRIB)/absl/absl -labsl_int128 \
                -L$(CLICKHOUSE_CONTRIB)/zstd/zstd -lzstdstatic

# ---- CUDA kernel compilation (nvcc) ----------------------------------------
SM_ARCH ?= native

NVCCFLAGS := \
    -std=c++17 \
    -arch=$(SM_ARCH) \
    -O2 \
    -Xcompiler -fPIC \
    $(ABI_FLAG) \
    $(CUDA_INCLUDE_FLAGS) \
    $(TORCH_INCLUDE_FLAGS) \
    $(PYTHON_INCLUDE_FLAGS) \
    -Icsrc

RING_SRCS := csrc/ring/producer.cu \
             csrc/ring/ring_engine.cu \
             csrc/ring/ring_engine_py.cu
RING_OBJS := $(patsubst csrc/%.cu,$(BUILD_DIR)/%.o,$(RING_SRCS))

OBJS += $(RING_OBJS)

.PHONY: all host clean cuda-info FORCE

all: $(FINAL_TARGET)

host: $(HOST_FINAL_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

FORCE:

$(CUDA_CONFIG_STAMP): FORCE | $(BUILD_DIR)
	$(PYTHON) $(CUDA_RESOLVER) write-stamp --output $@ --record "$(CUDA_CONFIG_RAW)"

$(BUILD_DIR)/%.o: csrc/%.cpp $(HEADERS) $(CUDA_CONFIG_STAMP) | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(HOST_BUILD_DIR)/%.o: csrc/%.cpp $(HEADERS) | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(HOST_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/ring/%.o: csrc/ring/%.cu $(CUDA_CONFIG_STAMP) | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CUDA_NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(FINAL_TARGET): $(OBJS)
	$(CXX) -shared $^ -o $(BUILD_DIR)/$(TARGET_SO) $(LDFLAGS)
	$(PYTHON) $(CUDA_RESOLVER) check-link --binary $(BUILD_DIR)/$(TARGET_SO) --expected-major $(CUDA_EXPECTED_MAJOR)
	cp $(BUILD_DIR)/$(TARGET_SO) $(LOCAL_TARGET)
	cp $(BUILD_DIR)/$(TARGET_SO) $(FINAL_TARGET)

$(HOST_FINAL_TARGET): $(HOST_OBJS)
	$(CXX) -shared $^ -o $(HOST_BUILD_DIR)/$(HOST_TARGET_SO) $(HOST_LDFLAGS)
	cp $(HOST_BUILD_DIR)/$(HOST_TARGET_SO) $(HOST_LOCAL_TARGET)
	cp $(HOST_BUILD_DIR)/$(HOST_TARGET_SO) $(HOST_FINAL_TARGET)

cuda-info:
	@echo "CUDA toolkit root: $(CUDA_TOOLKIT_ROOT_RESOLVED)"
	@echo "CUDA compiler: $(CUDA_NVCC)"
	@echo "CUDA version: $(CUDA_VERSION)"
	@echo "PyTorch CUDA version: $(TORCH_CUDA_VERSION)"
	@echo "CUDA include directory: $(CUDA_INCLUDE_DIR)"
	@echo "CUDA library directory: $(CUDA_LIBRARY_DIR)"

clean:
	rm -rf $(BUILD_DIR) $(LOCAL_TARGET) $(FINAL_TARGET) \
		$(HOST_LOCAL_TARGET) $(HOST_FINAL_TARGET)

# Overrides --------------------------------------------------------------
ifeq ($(LTO),1)
CXXFLAGS += -flto
LDFLAGS += -flto
HOST_CXXFLAGS += -flto
HOST_LDFLAGS += -flto
endif

# Optional NVTX runtime linking (default off). The code compiles without the
# library thanks to nvtx_shim.h providing no-op stubs if headers/libs are absent.
NVTX ?= 0
ifeq ($(NVTX),1)
LDFLAGS += -lnvToolsExt
endif

# Automatic header dependency tracking.
# -MMD -MP (g++) and -MMD -MF (nvcc) above generate .d files alongside .o files.
# Including them here makes make rebuild any .o whose headers have changed.
-include $(wildcard $(BUILD_DIR)/*.d $(BUILD_DIR)/ring/*.d $(HOST_BUILD_DIR)/*.d)
