PYTHON ?= python3
CXX ?= g++

# Output extension suffix, e.g. .cpython-311-x86_64-linux-gnu.so
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or sysconfig.get_config_var('SO'))")

TARGET := clickhouse_client$(EXT_SUFFIX)
BUILD_DIR := build

PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

# ---- ClickHouse-cpp locations (override as needed) ----
CLICKHOUSE_INCLUDE ?= $(abspath $(CURDIR)/../clickhouse-cpp)
CLICKHOUSE_LIBDIR  ?= $(abspath $(CURDIR)/../clickhouse-cpp/build/clickhouse)
CLICKHOUSE_LIBS    ?= -lclickhouse-cpp-lib

# Add rpath so Python can load dependent .so without LD_LIBRARY_PATH (optional)
RPATH ?= -Wl,-rpath,$(CLICKHOUSE_LIBDIR)

CXXFLAGS += -O3 -Wall -Wextra -std=c++17 -fPIC -pthread $(PYBIND11_INCLUDES) -I$(CLICKHOUSE_INCLUDE) -I$(CLICKHOUSE_INCLUDE)/contrib/absl
LDFLAGS  += -shared -L$(CLICKHOUSE_LIBDIR) $(RPATH) $(CLICKHOUSE_LIBS)

SRCS := clickhouse_client.cpp
OBJS := $(BUILD_DIR)/clickhouse_client.o

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/clickhouse_client.o: clickhouse_client.cpp clickhouse_client.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c clickhouse_client.cpp -o $@

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
