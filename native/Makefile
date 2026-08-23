PYTHON ?= python
CXX ?= g++

# Optional toggles
#   LTO=1      -> enable link-time optimization (-flto)
LTO ?= 0

TARGET := _native_backend
SRCS := csrc/bindings.cpp \
        csrc/clickhouse_client.cpp \
        csrc/ring/drain_thread.cpp \
        csrc/ring/p2p_thread.cpp \
        csrc/ring/ring_torch_op.cpp
HEADERS := csrc/dmx_host_utils.h \
           csrc/clickhouse_client.h csrc/dmx_host_engine.h \
           csrc/ring/ring_config.h csrc/ring/ring_state.h csrc/ring/ring_alloc.h \
           csrc/ring/task_entry.h csrc/ring/drain_task.h csrc/ring/drain_thread.h \
           csrc/ring/p2p_thread.h csrc/ring/pinned_staging.h \
           csrc/ring/ring_engine.h csrc/ring/ring_engine_py.h \
           csrc/ring/ring_torch_op.h csrc/ring/tensor_meta.h
BUILD_DIR := build
OBJS := $(patsubst csrc/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
PROJECT_ROOT := $(abspath $(CURDIR)/..)

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET_SO := $(TARGET)$(EXT_SUFFIX)
LOCAL_TARGET := $(CURDIR)/$(TARGET_SO)
FINAL_TARGET := $(PROJECT_ROOT)/src/dmi/$(TARGET_SO)

TORCH_INCLUDE_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join(f'-I{path}' for path in include_paths()))")

PYTHON_INCLUDE_FLAGS := $(shell $(PYTHON) -c "import sysconfig; paths = sysconfig.get_paths(); includes = {paths.get('include'), paths.get('platinclude')}; print(' '.join(f'-I{p}' for p in includes if p))")

CUDA_INCLUDE_DIRS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import CUDA_HOME; import glob, os, site, sys; roots=list(dict.fromkeys(p for p in (CUDA_HOME, sys.prefix) if p)); sites=[*site.getsitepackages(), site.getusersitepackages()]; paths=[os.path.join(root, 'include') for root in roots]; paths += [path for root in roots for path in glob.glob(os.path.join(root, 'targets', '*', 'include'))]; paths += [path for base in sites for path in glob.glob(os.path.join(base, 'nvidia', '*', 'include'))]; print(' '.join(dict.fromkeys(path for path in paths if os.path.isdir(path))))")
CUDA_INCLUDE_FLAGS := $(addprefix -I,$(CUDA_INCLUDE_DIRS))

TORCH_LIB_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join(f'-L{path}' for path in library_paths()))")

TORCH_RPATH_FLAGS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join(f'-Wl,-rpath,{path}' for path in library_paths()))")

CUDA_LIBRARY_DIRS := $(shell $(PYTHON) -c "from torch.utils.cpp_extension import CUDA_HOME; import glob, os, site, sys, sysconfig; roots=list(dict.fromkeys(p for p in (CUDA_HOME, sys.prefix) if p)); sites=[*site.getsitepackages(), site.getusersitepackages()]; multiarch=sysconfig.get_config_var('MULTIARCH'); paths=[os.path.join(root, leaf) for root in roots for leaf in ('lib64', 'lib')]; paths += [path for root in roots for path in glob.glob(os.path.join(root, 'targets', '*', 'lib*'))]; paths += [path for base in sites for path in glob.glob(os.path.join(base, 'nvidia', '*', 'lib'))]; paths += [path for path in (os.path.join('/usr/lib', multiarch) if multiarch else '', '/usr/lib64', '/usr/lib', '/usr/local/nvidia/lib64') if path]; print(' '.join(dict.fromkeys(path for path in paths if os.path.isdir(path))))")
CUDA_LIB_FLAGS := $(addprefix -L,$(CUDA_LIBRARY_DIRS))

comma := ,
CUDA_RPATH_FLAGS := $(foreach path,$(CUDA_LIBRARY_DIRS),-Wl$(comma)-rpath$(comma)$(path))

ABI_FLAG := $(shell $(PYTHON) -c "import torch; print(f'-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}')")

# Pybind11 ABI flags
PYBIND_FLAGS := $(shell $(PYTHON) -c "import pybind11; print(' '.join([f'-DPYBIND11_COMPILER_TYPE=\\\"{pybind11.get_compiler_type()}\\\"', f'-DPYBIND11_STDLIB=\\\"{pybind11.get_stdlib()}\\\"', f'-DPYBIND11_BUILD_ABI=\\\"{pybind11.get_build_abi()}\\\"']))" 2>/dev/null || echo "")

CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
            -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=$(TARGET) \
            $(ABI_FLAG) $(PYBIND_FLAGS) \
            $(TORCH_INCLUDE_FLAGS) $(PYTHON_INCLUDE_FLAGS) $(CUDA_INCLUDE_FLAGS) -Icsrc \
            -MMD -MP

LDFLAGS := $(TORCH_LIB_FLAGS) $(CUDA_LIB_FLAGS) $(TORCH_RPATH_FLAGS) $(CUDA_RPATH_FLAGS) \
           -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch -ltorch_python -lcudart -lcuda -lpthread


# ---- ClickHouse C++ client (fixed in-repo paths) ----
CLICKHOUSE_INCLUDE := $(PROJECT_ROOT)/third_party/clickhouse-cpp
CLICKHOUSE_LIB_DIR := $(CLICKHOUSE_INCLUDE)/build/clickhouse
CLICKHOUSE_LIBS    := -lclickhouse-cpp-lib
CLICKHOUSE_CONTRIB := $(CLICKHOUSE_INCLUDE)/build/contrib

ifneq ($(CLICKHOUSE_INCLUDE),)
CXXFLAGS += -I$(CLICKHOUSE_INCLUDE) -I$(CLICKHOUSE_INCLUDE)/contrib/absl
endif
ifneq ($(CLICKHOUSE_LIB_DIR),)
LDFLAGS += -L$(CLICKHOUSE_LIB_DIR) -Wl,-rpath,$(CLICKHOUSE_LIB_DIR)
endif
LDFLAGS += $(CLICKHOUSE_LIBS)
# clickhouse-cpp transitive deps (static libs built by its cmake)
LDFLAGS += -L$(CLICKHOUSE_CONTRIB)/lz4/lz4 -llz4 \
           -L$(CLICKHOUSE_CONTRIB)/cityhash/cityhash -lcityhash \
           -L$(CLICKHOUSE_CONTRIB)/absl/absl -labsl_int128 \
           -L$(CLICKHOUSE_CONTRIB)/zstd/zstd -lzstdstatic

CONDA_PREFIX ?= $(shell $(PYTHON) -c "import os; print(os.environ.get('CONDA_PREFIX',''))")
ifneq ($(CONDA_PREFIX),)
CXXFLAGS += -I$(CONDA_PREFIX)/include
LDFLAGS  += -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
endif


# venv/virtualenv support (adds include/lib paths for the active interpreter)
VENV_PREFIX ?= $(shell $(PYTHON) -c "import sys; in_venv = (getattr(sys,'base_prefix',sys.prefix) != sys.prefix) or hasattr(sys,'real_prefix'); print(sys.prefix if in_venv else '')")
ifneq ($(VENV_PREFIX),)
CXXFLAGS += -I$(VENV_PREFIX)/include
ifneq ($(wildcard $(VENV_PREFIX)/lib64),)
LDFLAGS  += -L$(VENV_PREFIX)/lib64 -Wl,-rpath,$(VENV_PREFIX)/lib64
endif
ifneq ($(wildcard $(VENV_PREFIX)/lib),)
LDFLAGS  += -L$(VENV_PREFIX)/lib -Wl,-rpath,$(VENV_PREFIX)/lib
endif
endif

# ---- CUDA kernel compilation (nvcc) ----------------------------------------
NVCC    ?= $(shell $(PYTHON) -c "from torch.utils.cpp_extension import CUDA_HOME; import os, shutil, sys; paths=[shutil.which('nvcc'), os.path.join(CUDA_HOME, 'bin', 'nvcc') if CUDA_HOME else None, os.path.join(sys.prefix, 'bin', 'nvcc'), '/usr/local/cuda/bin/nvcc']; print(next((path for path in paths if path and (shutil.which(path) or os.path.isfile(path))), 'nvcc'))")
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

.PHONY: all build clean

all: $(FINAL_TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: csrc/%.cpp $(HEADERS) | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/ring/%.o: csrc/ring/%.cu | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(FINAL_TARGET): $(OBJS)
	$(CXX) -shared $^ -o $(BUILD_DIR)/$(TARGET_SO) $(LDFLAGS)
	cp $(BUILD_DIR)/$(TARGET_SO) $(LOCAL_TARGET)
	cp $(BUILD_DIR)/$(TARGET_SO) $(FINAL_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(LOCAL_TARGET) $(FINAL_TARGET)

# Overrides --------------------------------------------------------------
ifeq ($(LTO),1)
CXXFLAGS += -flto
LDFLAGS += -flto
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
-include $(wildcard $(BUILD_DIR)/*.d $(BUILD_DIR)/ring/*.d)
