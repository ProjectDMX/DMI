PYTHON ?= python
CXX ?= g++

# Optional toggles
#   LTO=1      -> enable link-time optimization (-flto)
LTO ?= 0

FULL_TARGET := monitoring_native_backend
HOST_TARGET := monitoring_host_backend
HOST_SRCS := csrc/bindings.cpp \
             csrc/clickhouse_client.cpp
FULL_SRCS := $(HOST_SRCS) \
             csrc/ring/drain_thread.cpp \
             csrc/ring/p2p_thread.cpp \
             csrc/ring/ring_torch_op.cpp
HOST_HEADERS := csrc/dmx_host_utils.h \
                csrc/clickhouse_client.h csrc/dmx_host_engine.h \
                csrc/pipelined_engine.hpp
FULL_HEADERS := $(HOST_HEADERS) \
                csrc/ring/ring_config.h csrc/ring/ring_state.h csrc/ring/ring_alloc.h \
                csrc/ring/task_entry.h csrc/ring/drain_task.h csrc/ring/drain_thread.h \
                csrc/ring/p2p_thread.h csrc/ring/pinned_staging.h \
                csrc/ring/ring_engine.h csrc/ring/ring_engine_py.h \
                csrc/ring/ring_torch_op.h csrc/ring/tensor_meta.h
BUILD_DIR := build
FULL_OBJS := $(patsubst csrc/%.cpp,$(BUILD_DIR)/full/%.o,$(FULL_SRCS))
HOST_OBJS := $(patsubst csrc/%.cpp,$(BUILD_DIR)/host/%.o,$(HOST_SRCS))
PROJECT_ROOT := $(abspath $(CURDIR)/..)

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
FULL_TARGET_SO := $(FULL_TARGET)$(EXT_SUFFIX)
HOST_TARGET_SO := $(HOST_TARGET)$(EXT_SUFFIX)
FULL_LOCAL_TARGET := $(CURDIR)/$(FULL_TARGET_SO)
HOST_LOCAL_TARGET := $(CURDIR)/$(HOST_TARGET_SO)
FULL_FINAL_TARGET := $(PROJECT_ROOT)/$(FULL_TARGET_SO)
HOST_FINAL_TARGET := $(PROJECT_ROOT)/$(HOST_TARGET_SO)

TORCH_INCLUDE_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join(f'-I{path}' for path in include_paths()))")

PYTHON_INCLUDE_FLAGS := $(shell $(PYTHON) -c "import sysconfig; paths = sysconfig.get_paths(); includes = {paths.get('include'), paths.get('platinclude')}; print(' '.join(f'-I{p}' for p in includes if p))")

CUDA_INCLUDE_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import CUDA_HOME; import os; cuda_inc = os.path.join(CUDA_HOME, 'include') if CUDA_HOME else ''; print(f'-I{cuda_inc}' if cuda_inc and os.path.exists(cuda_inc) else '')")

TORCH_LIB_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join(f'-L{path}' for path in library_paths()))")

TORCH_RPATH_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join(f'-Wl,-rpath,{path}' for path in library_paths()))")

CUDA_LIB_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import CUDA_HOME; import os; print(next((f'-L{os.path.join(CUDA_HOME, c)}' for c in ('lib64','lib') if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, c))), ''))")

CUDA_RPATH_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import CUDA_HOME; import os; print(next((f'-Wl,-rpath,{os.path.join(CUDA_HOME, c)}' for c in ('lib64','lib') if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, c))), ''))")

ABI_FLAG := $(shell $(PYTHON) -c "import torch; print(f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}')")

# Pybind11 ABI flags
PYBIND_FLAGS := $(shell $(PYTHON) -c "import pybind11; print(' '.join([f'-DPYBIND11_COMPILER_TYPE=\\\"{pybind11.get_compiler_type()}\\\"', f'-DPYBIND11_STDLIB=\\\"{pybind11.get_stdlib()}\\\"', f'-DPYBIND11_BUILD_ABI=\\\"{pybind11.get_build_abi()}\\\"']))" 2>/dev/null || echo "")

COMMON_CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
                   -DTORCH_API_INCLUDE_EXTENSION_H \
                   $(ABI_FLAG) $(PYBIND_FLAGS) \
                   $(TORCH_INCLUDE_FLAGS) $(PYTHON_INCLUDE_FLAGS) -Icsrc \
                   -MMD -MP
CXXFLAGS = $(COMMON_CXXFLAGS) -DTORCH_EXTENSION_NAME=$(FULL_TARGET) $(CUDA_INCLUDE_FLAGS)
HOST_CXXFLAGS = $(COMMON_CXXFLAGS) -DTORCH_EXTENSION_NAME=$(HOST_TARGET) -DDMI_HOST_ONLY=1

COMMON_LDFLAGS := $(TORCH_LIB_FLAGS) $(TORCH_RPATH_FLAGS) \
                  -lc10 -ltorch_cpu -ltorch -ltorch_python -lpthread
ifeq ($(shell uname -s),Darwin)
COMMON_LDFLAGS += -Wl,-undefined,dynamic_lookup
endif
LDFLAGS = $(COMMON_LDFLAGS) $(CUDA_LIB_FLAGS) $(CUDA_RPATH_FLAGS) \
          -lc10_cuda -ltorch_cuda -lcudart -lcuda
HOST_LDFLAGS = $(COMMON_LDFLAGS)


# ---- ClickHouse C++ client (fixed in-repo paths) ----
CLICKHOUSE_INCLUDE := $(abspath $(CURDIR)/../libs/clickhouse-cpp)
CLICKHOUSE_LIB_DIR := $(abspath $(CURDIR)/../libs/clickhouse-cpp/build/clickhouse)
CLICKHOUSE_LIBS    := -lclickhouse-cpp-lib
CLICKHOUSE_CONTRIB := $(abspath $(CURDIR)/../libs/clickhouse-cpp/build/contrib)

ifneq ($(CLICKHOUSE_INCLUDE),)
COMMON_CXXFLAGS += -I$(CLICKHOUSE_INCLUDE) -I$(CLICKHOUSE_INCLUDE)/contrib/absl
endif
ifneq ($(CLICKHOUSE_LIB_DIR),)
COMMON_LDFLAGS += -L$(CLICKHOUSE_LIB_DIR) -Wl,-rpath,$(CLICKHOUSE_LIB_DIR)
endif
COMMON_LDFLAGS += $(CLICKHOUSE_LIBS)
# clickhouse-cpp transitive deps (static libs built by its cmake)
COMMON_LDFLAGS += -L$(CLICKHOUSE_CONTRIB)/lz4/lz4 -llz4 \
                  -L$(CLICKHOUSE_CONTRIB)/cityhash/cityhash -lcityhash \
                  -L$(CLICKHOUSE_CONTRIB)/absl/absl -labsl_int128 \
                  -L$(CLICKHOUSE_CONTRIB)/zstd/zstd -lzstdstatic

CONDA_PREFIX ?= $(shell $(PYTHON) -c "import os; print(os.environ.get('CONDA_PREFIX',''))")
ifneq ($(CONDA_PREFIX),)
COMMON_CXXFLAGS += -I$(CONDA_PREFIX)/include
COMMON_LDFLAGS  += -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
endif


# venv/virtualenv support (adds include/lib paths for the active interpreter)
VENV_PREFIX ?= $(shell $(PYTHON) -c "import sys; in_venv = (getattr(sys,'base_prefix',sys.prefix) != sys.prefix) or hasattr(sys,'real_prefix'); print(sys.prefix if in_venv else '')")
ifneq ($(VENV_PREFIX),)
COMMON_CXXFLAGS += -I$(VENV_PREFIX)/include
ifneq ($(wildcard $(VENV_PREFIX)/lib64),)
COMMON_LDFLAGS  += -L$(VENV_PREFIX)/lib64 -Wl,-rpath,$(VENV_PREFIX)/lib64
endif
ifneq ($(wildcard $(VENV_PREFIX)/lib),)
COMMON_LDFLAGS  += -L$(VENV_PREFIX)/lib -Wl,-rpath,$(VENV_PREFIX)/lib
endif
endif

# ---- CUDA kernel compilation (nvcc) ----------------------------------------
NVCC    ?= /usr/local/cuda/bin/nvcc
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
RING_OBJS := $(patsubst csrc/%.cu,$(BUILD_DIR)/full/%.o,$(RING_SRCS))

FULL_OBJS += $(RING_OBJS)

.PHONY: all host clean

all: $(FULL_FINAL_TARGET)

host: $(HOST_FINAL_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/full/%.o: csrc/%.cpp $(FULL_HEADERS) | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/host/%.o: csrc/%.cpp $(HOST_HEADERS) | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(HOST_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/full/ring/%.o: csrc/ring/%.cu | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(FULL_FINAL_TARGET): $(FULL_OBJS)
	$(CXX) -shared $^ -o $(BUILD_DIR)/full/$(FULL_TARGET_SO) $(LDFLAGS)
	cp $(BUILD_DIR)/full/$(FULL_TARGET_SO) $(FULL_LOCAL_TARGET)
	cp $(BUILD_DIR)/full/$(FULL_TARGET_SO) $(FULL_FINAL_TARGET)

$(HOST_FINAL_TARGET): $(HOST_OBJS)
	$(CXX) -shared $^ -o $(BUILD_DIR)/host/$(HOST_TARGET_SO) $(HOST_LDFLAGS)
	cp $(BUILD_DIR)/host/$(HOST_TARGET_SO) $(HOST_LOCAL_TARGET)
	cp $(BUILD_DIR)/host/$(HOST_TARGET_SO) $(HOST_FINAL_TARGET)

clean:
	rm -rf $(BUILD_DIR) \
	       $(FULL_LOCAL_TARGET) $(FULL_FINAL_TARGET) \
	       $(HOST_LOCAL_TARGET) $(HOST_FINAL_TARGET)

# Overrides --------------------------------------------------------------
ifeq ($(LTO),1)
COMMON_CXXFLAGS += -flto
COMMON_LDFLAGS += -flto
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
-include $(wildcard $(BUILD_DIR)/full/*.d $(BUILD_DIR)/full/ring/*.d \
                   $(BUILD_DIR)/host/*.d)
