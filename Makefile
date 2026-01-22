.PHONY: all monitoring dmx_host clickhouse-cpp deps

MONITORING_DIR ?= $(CURDIR)/monitoring
DMX_HOST_DIR ?= $(CURDIR)/dmx_host
CLICKHOUSE_DIR ?= $(CURDIR)/clickhouse-cpp
CLICKHOUSE_BUILD ?= $(CLICKHOUSE_DIR)/build

all: deps monitoring dmx_host

monitoring:
	$(MAKE) -C $(MONITORING_DIR)

dmx_host:
	$(MAKE) -C $(DMX_HOST_DIR)

deps: clickhouse-cpp

clickhouse-cpp:
	cmake -S $(CLICKHOUSE_DIR) -B $(CLICKHOUSE_BUILD) -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DWITH_OPENSSL=ON
	$(MAKE) -C $(CLICKHOUSE_BUILD)
