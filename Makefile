.PHONY: all monitoring host test-cpu test-host clean

MONITORING_DIR ?= $(CURDIR)/monitoring
PYTHON ?= python

all: monitoring

monitoring:
	$(MAKE) -C $(MONITORING_DIR)

host:
	$(MAKE) -C $(MONITORING_DIR) host

test-cpu:
	$(PYTHON) -m pytest -m cpu -q

test-host: host
	$(PYTHON) -m pytest tests/test_cpu_native_build.py \
		tests/test_clickhouse_host_benchmark.py tests/test_test_harness.py -q
	$(PYTHON) -c "from monitoring import _native_engine as n; host = n._load_named_extension(n._HOST_EXTENSION_NAME); assert host.DMXHostEngine.__module__ == 'monitoring_host_backend'"
	$(PYTHON) -c "import sys; from monitoring.integration_api.v1 import DMXHostEngine; assert DMXHostEngine.__module__ in {'monitoring_host_backend', 'monitoring_native_backend'}; assert 'monitoring.ring_transport' not in sys.modules"

clean:
	$(MAKE) -C $(MONITORING_DIR) clean
