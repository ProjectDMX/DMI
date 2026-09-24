PYTHON ?= python
CXX ?= g++

# Optional toggles
#   LTO=1      -> enable link-time optimization (-flto)
LTO ?= 0

TARGET := _native_backend
HOST_TARGET := _host_backend
SRCS := csrc/bindings.cpp \
        csrc/clickhouse_record_sink.cpp \
        csrc/reference_python_capture_sink.cpp \
        csrc/clickhouse_client.cpp \
        csrc/ring/drain_thread.cpp \
        csrc/ring/p2p_thread.cpp \
        csrc/ring/record_consumer.cpp \
        csrc/ring/ring_torch_op.cpp
HOST_SRCS := csrc/bindings.cpp csrc/clickhouse_client.cpp \
             csrc/pack/pack_builder.cpp
HEADERS := csrc/dmx_host_utils.h \
           csrc/clickhouse_client.h csrc/clickhouse_record_sink.h \
           csrc/reference_python_capture_sink.h \
           csrc/dmx_host_engine.h csrc/record_schema.h \
           csrc/ring/ring_config.h csrc/ring/ring_state.h csrc/ring/ring_alloc.h \
           csrc/ring/publication_word.h csrc/ring/drain_task.h csrc/ring/drain_thread.h \
           csrc/ring/record_descriptor.h csrc/ring/record_consumer.h \
           csrc/ring/record_sink.h \
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
TORCH_EXT_STAMP := $(BUILD_DIR)/torch-extension.config
CUDA_RESOLVE_REQUIRED := 1
ifneq ($(strip $(MAKECMDGOALS)),)
ifeq ($(strip $(filter-out clean host,$(MAKECMDGOALS))),)
CUDA_RESOLVE_REQUIRED := 0
endif
# Pure-CPU drivers build without a CUDA toolkit (or torch): the test suite
# drives them over stdin/stdout. When EVERY requested goal is CPU-only (or
# clean/host), skip CUDA resolution entirely.
CPU_ONLY_GOALS := build/conformance_main build/conformance_sign build/conformance_store build/conformance_spool build/conformance_sink build/bench_builder build/bench_sink build/conformance_catalog build/_dmi_native_sink build/_dmi_native_store
ifneq ($(strip $(MAKECMDGOALS)),)
ifeq ($(strip $(filter-out clean host cpu-goals capture check-libcurl $(TORCH_EXT_STAMP) $(CPU_ONLY_GOALS),$(MAKECMDGOALS))),)
CUDA_RESOLVE_REQUIRED := 0
endif
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

CXXFLAGS := -std=c++20 -O3 -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
            -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=$(TARGET) \
            $(ABI_FLAG) $(PYBIND_FLAGS) \
            $(TORCH_INCLUDE_FLAGS) $(PYTHON_INCLUDE_FLAGS) $(CUDA_INCLUDE_FLAGS) -Icsrc \
            -MMD -MP
HOST_CXXFLAGS := -std=c++20 -O3 -fPIC -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
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
    -std=c++20 \
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

.PHONY: all host clean cuda-info cpu-goals capture check-libcurl FORCE

# The default build includes the capture storage extensions, because `clean`
# removes them with the rest of build/ and the documented rebuild is plain
# `make`: left out, `make clean && make` deleted them and never rebuilt them.
# They need libcurl's dev package, which the full backend does not, so a host
# without it fails here with the package named. Unconditional rather than
# "when libcurl is found": skipping silently is the stranding again, reported
# only at runtime by the loader. A build that wants the backend alone says so
# with CAPTURE=0. Only 0 and 1 are accepted: any other value -- CAPTURE=yes,
# or an unrelated CAPTURE someone exported -- used to mean 0, and dropped
# the extensions just as silently.
CAPTURE ?= 1
ifeq ($(strip $(CAPTURE)),1)
all: $(FINAL_TARGET) capture
else ifeq ($(strip $(CAPTURE)),0)
all: $(FINAL_TARGET)
else
$(error CAPTURE must be 0 or 1, not '$(CAPTURE)' (from the $(origin CAPTURE)): 1 builds the capture storage extensions with the default build, 0 leaves them out)
endif

# Every target that builds without CUDA, as ONE goal. CI's compile gate
# asks for this rather than listing the binaries: a list in the workflow
# goes stale the moment CPU_ONLY_GOALS gains a target -- which it did,
# when the catalog driver landed and the hardcoded list did not know --
# and that is exactly the drift a compile gate exists to prevent. Adding
# a target to CPU_ONLY_GOALS now puts it under the gate automatically.
cpu-goals: $(CPU_ONLY_GOALS)

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

# A1 conformance driver: dependency-free stdin/stdout harness the Python suite
# uses to drive the native pack builder and compare bytes against the
# reference. No torch/pybind needed; runs in any shell or CI job.
build/conformance_main: csrc/pack/pack_builder.cpp csrc/pack/conformance_main.cpp csrc/pack/pack_builder.h csrc/common/json.cpp csrc/common/json.h
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
	    -o $@ csrc/pack/pack_builder.cpp csrc/pack/conformance_main.cpp csrc/common/json.cpp \
	    -Icsrc/pack -Icsrc/common -lcrypto

# A1 throughput gate: PackBuilder on the bench corpus shape (10k x 64 KiB).
build/bench_builder: csrc/pack/pack_builder.cpp csrc/pack/bench_builder.cpp csrc/pack/pack_builder.h
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O3 -march=native -o $@ \
	    csrc/pack/pack_builder.cpp csrc/pack/bench_builder.cpp \
	    -Icsrc/pack -lcrypto

# A5a throughput gate: PackSink end-to-end (queue → packers → spool),
# same corpus shape as bench_capture_pipeline. Usage:
#   build/bench_sink [single|interleaved] [workers...] [spooldir=PREFIX]
build/bench_sink: csrc/sink/pack_sink.cpp csrc/sink/object_key.cpp csrc/sink/record_row.cpp csrc/sink/bench_sink.cpp csrc/pack/pack_builder.cpp csrc/store/spool.cpp csrc/common/json.cpp csrc/sink/pack_sink.h csrc/sink/object_key.h csrc/sink/record_row.h
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ \
	    csrc/sink/pack_sink.cpp csrc/sink/object_key.cpp csrc/sink/record_row.cpp csrc/sink/bench_sink.cpp \
	    csrc/pack/pack_builder.cpp csrc/store/spool.cpp csrc/common/json.cpp \
	    -Icsrc/sink -Icsrc/pack -Icsrc/store -Icsrc/common -lcrypto -lpthread

# ---- libcurl (the S3 client, the catalog's HTTP client) --------------------
# Needed by build/conformance_store, build/conformance_catalog and
# _dmi_native_store. CURL_INCDIR/CURL_LIBDIR resolve in this order (each
# of the two on its own):
#   1. set on the command line or in the environment -- CI passes the system
#      paths, and this always wins;
#   2. a CURL_SYSROOT set on the command line or in the environment: the
#      packages extracted there, for a host without root. The dev package's
#      libcurl.so links to a file in the runtime package, so both go in:
#        apt-get download libcurl4-openssl-dev libcurl4
#        for deb in libcurl4*.deb; do dpkg-deb -x "$deb" <sysroot>; done
#      Ahead of pkg-config, because naming a sysroot is asking for it: behind
#      it, a host that also has a libcurl.pc ignored the sysroot silently;
#   3. `pkg-config libcurl`, which knows them wherever the dev package
#      (libcurl4-openssl-dev on Debian/Ubuntu) is installed;
#   4. the default sysroot, CURL_SYSROOT_DEFAULT, laid out as in 2 -- only
#      when that directory exists and belongs to the user running make: an
#      unowned directory under /tmp must not supply headers and a library to
#      the build ahead of the system's;
#   5. none at all: no -I/-L, and the compiler's default paths apply.
# At runtime the store loads the OS's libcurl.so.4.
PKG_CONFIG ?= pkg-config
CURL_SYSROOT_EXPLICIT := $(if $(filter command line environment,$(origin CURL_SYSROOT)),$(strip $(CURL_SYSROOT)))
ifneq ($(filter undefined,$(origin CURL_INCDIR) $(origin CURL_LIBDIR)),)
ifeq ($(CURL_SYSROOT_EXPLICIT),)
CURL_PKG_CONFIG_FOUND := $(shell $(PKG_CONFIG) --exists libcurl 2>/dev/null && echo yes)
endif
endif
ifeq ($(CURL_PKG_CONFIG_FOUND),yes)
ifeq ($(origin CURL_INCDIR),undefined)
CURL_INCDIR := $(shell $(PKG_CONFIG) --variable=includedir libcurl)
endif
ifeq ($(origin CURL_LIBDIR),undefined)
CURL_LIBDIR := $(shell $(PKG_CONFIG) --variable=libdir libcurl)
endif
endif
CURL_SYSROOT_DEFAULT ?= /tmp/opencode/sysroot
ifeq ($(origin CURL_SYSROOT),undefined)
CURL_SYSROOT := $(shell test -d '$(CURL_SYSROOT_DEFAULT)' && test -O '$(CURL_SYSROOT_DEFAULT)' && echo '$(CURL_SYSROOT_DEFAULT)')
endif
ifneq ($(CURL_SYSROOT),)
CURL_INCDIR ?= $(CURL_SYSROOT)/usr/include/x86_64-linux-gnu
CURL_LIBDIR ?= $(CURL_SYSROOT)/usr/lib/x86_64-linux-gnu
endif
CURL_CPPFLAGS = $(if $(CURL_INCDIR),-I$(CURL_INCDIR))
CURL_LDFLAGS = $(if $(CURL_LIBDIR),-L$(CURL_LIBDIR))
CURL_SEARCHED = $(if $(CURL_INCDIR)$(CURL_LIBDIR),CURL_INCDIR=$(CURL_INCDIR)$(comma) CURL_LIBDIR=$(CURL_LIBDIR),the compiler's default paths)

# Compile and link one libcurl call before any target that needs it, so a
# missing dev package is one line naming it rather than a page of compiler
# output from whichever source includes <curl/curl.h> first. An order-only
# prerequisite of those targets: it runs, but never makes them out of date.
# A dangling libcurl.so is named separately: it is the dev package extracted
# without its runtime package, and ld would skip it for libcurl.a.
check-libcurl:
	@if [ -n "$(CURL_LIBDIR)" ] && [ -L "$(CURL_LIBDIR)/libcurl.so" ] && [ ! -e "$(CURL_LIBDIR)/libcurl.so" ]; then \
	    echo "native/Makefile: $(CURL_LIBDIR)/libcurl.so links to a missing file." \
	         "Extract the runtime package libcurl4 into the same sysroot as" \
	         "libcurl4-openssl-dev -- see docs/install.md." >&2; \
	    exit 1; \
	fi
	@printf '#include <curl/curl.h>\nint main() { return curl_global_init(CURL_GLOBAL_DEFAULT); }\n' \
	    | $(CXX) -x c++ - -o /dev/null $(CURL_CPPFLAGS) $(CURL_LDFLAGS) -lcurl 2>/dev/null \
	    || { echo "native/Makefile: cannot build against libcurl" \
	              "($(CURL_SEARCHED))." \
	              "Install libcurl4-openssl-dev (Debian/Ubuntu; libcurl-devel on" \
	              "Fedora/RHEL), or pass CURL_INCDIR/CURL_LIBDIR -- see docs/install.md." \
	              "To build without the capture storage extensions, which the" \
	              "in-memory path does not need, pass CAPTURE=0." >&2; \
	         exit 1; }

# A2a SigV4 conformance driver: no torch/pybind/network; runs in any CI job.
build/conformance_sign: csrc/store/s3_sign.cpp csrc/store/conformance_sign.cpp csrc/store/s3_sign.h csrc/common/json.cpp csrc/common/json.h
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ \
	    csrc/store/s3_sign.cpp csrc/store/conformance_sign.cpp csrc/common/json.cpp \
	    -Icsrc/store -Icsrc/common -lcrypto

# A2 store conformance driver + fault-matrix target. Needs libcurl (above).
build/conformance_store: csrc/store/s3_sign.cpp csrc/store/s3_client.cpp csrc/store/spool.cpp csrc/store/uploader.cpp csrc/store/conformance_store.cpp csrc/store/s3_sign.h csrc/store/s3_client.h csrc/store/spool.h csrc/store/uploader.h csrc/common/json.cpp csrc/common/json.h csrc/common/curl_init.cpp csrc/common/curl_init.h | check-libcurl
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ \
	    csrc/store/s3_sign.cpp csrc/store/s3_client.cpp csrc/store/spool.cpp csrc/store/uploader.cpp csrc/store/conformance_store.cpp csrc/common/json.cpp csrc/common/curl_init.cpp \
	    -Icsrc/store -Icsrc/common $(CURL_CPPFLAGS) $(CURL_LDFLAGS) -lcrypto -lcurl -lpthread

# B1 catalog conformance driver: lease coordinator + version allocator over
# ClickHouse's HTTP interface. No torch/pybind; needs libcurl like the store.
build/conformance_catalog: csrc/catalog/clickhouse_client.cpp csrc/catalog/lease_coordinator.cpp csrc/catalog/version_allocator.cpp csrc/catalog/catalog_writer.cpp csrc/catalog/pack_index.cpp csrc/catalog/indexer.cpp csrc/catalog/schema.cpp csrc/catalog/reader.cpp csrc/catalog/hydration.cpp csrc/catalog/conformance_catalog.cpp csrc/common/json.cpp csrc/pack/pack_builder.cpp csrc/store/s3_sign.cpp csrc/store/s3_client.cpp csrc/catalog/clickhouse_client.h csrc/catalog/lease_coordinator.h csrc/catalog/version_allocator.h csrc/catalog/catalog_writer.h csrc/catalog/sql_escape.h csrc/catalog/pack_index.h csrc/catalog/indexer.h csrc/catalog/schema.h csrc/catalog/reader.h csrc/catalog/hydration.h csrc/common/json.h csrc/store/s3_client.h csrc/common/curl_init.cpp csrc/common/curl_init.h | check-libcurl
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ \
	    csrc/catalog/clickhouse_client.cpp csrc/catalog/lease_coordinator.cpp \
	    csrc/catalog/version_allocator.cpp csrc/catalog/catalog_writer.cpp \
	    csrc/catalog/pack_index.cpp csrc/catalog/indexer.cpp csrc/catalog/schema.cpp csrc/catalog/reader.cpp csrc/catalog/hydration.cpp \
	    csrc/catalog/conformance_catalog.cpp \
	    csrc/common/json.cpp csrc/common/curl_init.cpp csrc/pack/pack_builder.cpp \
	    csrc/store/s3_sign.cpp csrc/store/s3_client.cpp \
	    -Icsrc/catalog -Icsrc/common -Icsrc/pack -Icsrc/store \
	    $(CURL_CPPFLAGS) $(CURL_LDFLAGS) -lcrypto -lcurl -lpthread

# A3a spool conformance driver: no network, no torch; runs in any CI job.
build/conformance_spool: csrc/store/spool.cpp csrc/store/conformance_spool.cpp csrc/store/spool.h csrc/common/json.cpp csrc/common/json.h
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ \
	    csrc/store/spool.cpp csrc/store/conformance_spool.cpp csrc/common/json.cpp \
	    -Icsrc/store -Icsrc/common -lcrypto

# ---- Capture storage extensions (`make capture`, part of `all`) -------------
# _dmi_native_sink (the pack writer) and _dmi_native_store (the storage
# service and reader). Each is linked in build/ and installed next to
# _native_backend in src/dmi/, the first directory the loader
# (dmi/transport/native.py) searches, so `clean` removes both names and the
# default build puts both back. The build/ file stays because the torch
# adapter suites import it from there directly. The src/dmi/ name is a
# relative symlink to it, not a copy: the loader prefers src/dmi/, so a copy
# went stale -- and shadowed the fresh build -- whenever something rebuilt
# build/ alone, such as a branch whose `make build/_dmi_native_sink` writes
# only there. A copy left by an earlier build is replaced even when it is
# newer than build/. The targets are the files
# themselves, so an up-to-date extension is not relinked on every `make`;
# build/_dmi_native_sink and build/_dmi_native_store remain as the names CI
# and the loaders' error messages use. PYTHON must be the venv interpreter:
#   make capture PYTHON=/path/to/venv/bin/python
#
# No torch RUNPATH on the sink, unlike _host_backend: every route to it
# imports torch first (dmi.storage.capture does, before the loader runs),
# which is what maps libtorch into the process. Verified: a bare
# `import _dmi_native_sink` without torch fails on libtorch_cpu.so, and the
# loader route succeeds.
SINK_EXT := _dmi_native_sink$(EXT_SUFFIX)
STORE_EXT := _dmi_native_store$(EXT_SUFFIX)
SINK_BUILT := $(BUILD_DIR)/$(SINK_EXT)
STORE_BUILT := $(BUILD_DIR)/$(STORE_EXT)
SINK_FINAL_TARGET := $(PROJECT_ROOT)/src/dmi/$(SINK_EXT)
STORE_FINAL_TARGET := $(PROJECT_ROOT)/src/dmi/$(STORE_EXT)
# Headers by directory rather than by name: a file target is only as fresh
# as the prerequisites it lists, and a missed header means a stale .so.
SINK_HEADERS := $(wildcard csrc/sink/*.h csrc/pack/*.h csrc/store/*.h csrc/common/*.h csrc/ring/*.h)
STORE_HEADERS := $(wildcard csrc/catalog/*.h csrc/store/*.h csrc/pack/*.h csrc/common/*.h)

# What the extensions are built against beyond their sources: torch
# (libtorch for the sink, pybind11's headers for both), the C++ ABI it
# expects and the Python it runs in. A torch upgrade in the venv changes no
# source file, so without this record a file-named target stays "up to date"
# and keeps the old build. Rewritten only when the record changes, like
# $(CUDA_CONFIG_STAMP), so an unchanged venv relinks nothing.
TORCH_EXT_RECORD_PY := import sys, sysconfig, torch; \
    from torch.utils.cpp_extension import include_paths, library_paths; \
    print("|".join([torch.__version__, str(torch.version.git_version), \
        str(int(torch._C._GLIBCXX_USE_CXX11_ABI)), ";".join(include_paths()), \
        ";".join(library_paths()), sysconfig.get_paths()["include"], \
        sysconfig.get_config_var("EXT_SUFFIX")]))

$(TORCH_EXT_STAMP): FORCE | $(BUILD_DIR)
	record=$$($(PYTHON) -c '$(TORCH_EXT_RECORD_PY)') && \
	    $(PYTHON) $(CUDA_RESOLVER) write-stamp --output $@ --record "$$record"

.PHONY: build/_dmi_native_sink build/_dmi_native_store
capture: $(SINK_FINAL_TARGET) $(STORE_FINAL_TARGET)
build/_dmi_native_sink: $(SINK_FINAL_TARGET)
build/_dmi_native_store: $(STORE_FINAL_TARGET)

# FORCE when the installed name is a regular file, i.e. a copy.
installed_copy = $(if $(shell test -e '$(1)' && ! test -L '$(1)' && echo copy),FORCE)
define install_link
ln -sfn "$$($(PYTHON) -c 'import os, sys; print(os.path.relpath(*map(os.path.realpath, sys.argv[1:])))' '$(abspath $<)' '$(@D)')" $@
endef

$(SINK_FINAL_TARGET): $(SINK_BUILT) $(call installed_copy,$(SINK_FINAL_TARGET))
	$(install_link)

$(STORE_FINAL_TARGET): $(STORE_BUILT) $(call installed_copy,$(STORE_FINAL_TARGET))
	$(install_link)

# A5b torch-CPU adapter module: NativePackSink as an importable extension
# with a synthetic-envelope entry point for tests. No CUDA, no ClickHouse,
# no ring engine — torch CPU + libcrypto + spool/pack sources only.
# C++20 because ATen demands it and its floor MOVES: torch 2.13 refuses
# below C++17, torch 2.14 below C++20, and the CI pin `torch>=2.8,<3`
# resolves to whatever is newest on the day. C++20 is the floor for every
# target that includes torch's headers -- this sink, the backend and host
# objects (CXXFLAGS, HOST_CXXFLAGS) and the ring's nvcc compiles (NVCCFLAGS).
# C++17 stays only on the torch-free drivers.
$(SINK_BUILT): csrc/sink/native_pack_sink.cpp csrc/sink/record_row.cpp csrc/sink/pack_sink.cpp csrc/sink/object_key.cpp csrc/sink/bindings_sink.cpp csrc/pack/pack_builder.cpp csrc/store/spool.cpp csrc/common/json.cpp $(SINK_HEADERS) $(TORCH_EXT_STAMP)
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++20 -O2 -fPIC -shared -DTORCH_EXTENSION_NAME=_dmi_native_sink \
	    $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I'+p for p in include_paths()))") \
	    $(shell $(PYTHON) -c "import sysconfig; paths = sysconfig.get_paths(); print('-I'+paths.get('include'))") \
	    -Icsrc -Icsrc/sink -Icsrc/pack -Icsrc/store -Icsrc/common \
	    -D_GLIBCXX_USE_CXX11_ABI=$(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))") \
	    -o $@ \
	    csrc/sink/native_pack_sink.cpp csrc/sink/record_row.cpp \
	    csrc/sink/pack_sink.cpp csrc/sink/object_key.cpp \
	    csrc/sink/bindings_sink.cpp csrc/pack/pack_builder.cpp \
	    csrc/store/spool.cpp csrc/common/json.cpp \
	    $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L'+p for p in library_paths()))") \
	    -ltorch_cpu -ltorch -lc10 -ltorch_python -lcrypto -lpthread
# The native capture storage path's Python module: StorageService (spool ->
# object store -> catalog) and CaptureReader. Pure C++ + libcurl + libcrypto.
# pybind11 is header-only; its headers come from torch's include dir, but
# nothing from torch is linked, and no ring types are registered, so this
# loads beside _native_backend and _dmi_native_sink.
STORE_SOURCES := csrc/catalog/storage_service.cpp csrc/catalog/bindings_store.cpp \
    csrc/catalog/clickhouse_client.cpp csrc/catalog/lease_coordinator.cpp \
    csrc/catalog/version_allocator.cpp csrc/catalog/catalog_writer.cpp \
    csrc/catalog/pack_index.cpp csrc/catalog/indexer.cpp csrc/catalog/schema.cpp \
    csrc/catalog/reader.cpp csrc/catalog/hydration.cpp \
    csrc/store/s3_sign.cpp csrc/store/s3_client.cpp csrc/store/uploader.cpp \
    csrc/store/spool.cpp csrc/pack/pack_builder.cpp \
    csrc/common/json.cpp csrc/common/curl_init.cpp
$(STORE_BUILT): $(STORE_SOURCES) $(STORE_HEADERS) $(TORCH_EXT_STAMP) | check-libcurl
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -fPIC -shared -Wall -Wextra \
	    -I$(shell $(PYTHON) -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include'))") \
	    $(shell $(PYTHON) -c "import sysconfig; paths = sysconfig.get_paths(); print('-I'+paths.get('include'))") \
	    -Icsrc -Icsrc/catalog -Icsrc/common -Icsrc/pack -Icsrc/store $(CURL_CPPFLAGS) \
	    -o $@ $(STORE_SOURCES) \
	    $(CURL_LDFLAGS) -lcurl -lcrypto -lpthread
# A3b sink conformance driver: queue + assembler + spool staging in one
# process, driven over stdin/stdout. Needs only libcrypto + pthreads.
build/conformance_sink: csrc/sink/pack_sink.cpp csrc/sink/object_key.cpp csrc/sink/record_row.cpp csrc/sink/conformance_sink.cpp csrc/pack/pack_builder.cpp csrc/store/spool.cpp csrc/common/json.cpp csrc/sink/pack_sink.h csrc/sink/object_key.h csrc/sink/record_row.h
	mkdir -p $(BUILD_DIR)
	$(CXX) -std=c++17 -O2 -Wall -Wextra -o $@ \
	    csrc/sink/pack_sink.cpp csrc/sink/object_key.cpp csrc/sink/record_row.cpp csrc/sink/conformance_sink.cpp \
	    csrc/pack/pack_builder.cpp csrc/store/spool.cpp csrc/common/json.cpp \
	    -Icsrc/sink -Icsrc/pack -Icsrc/store -Icsrc/common -lcrypto -lpthread

clean:
	rm -rf $(BUILD_DIR) $(LOCAL_TARGET) $(FINAL_TARGET) \
		$(HOST_LOCAL_TARGET) $(HOST_FINAL_TARGET) \
		$(SINK_FINAL_TARGET) $(STORE_FINAL_TARGET)

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
