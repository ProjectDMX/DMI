PYTHON ?= python

.PHONY: all native host clean check check-compile test test-all test-cpu test-host test-package

NATIVE_DIR ?= $(CURDIR)/native

all: native

native:
	$(MAKE) -C $(NATIVE_DIR)

host:
	$(MAKE) -C $(NATIVE_DIR) host

clean:
	$(MAKE) -C $(NATIVE_DIR) clean

test: test-cpu

test-cpu:
	$(PYTHON) -m pytest -m cpu -q

test-host: host
	$(PYTHON) -m pytest tests/test_cpu_native_build.py \
		tests/test_clickhouse_host_benchmark.py tests/test_test_harness.py -q
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(CURDIR)/src'); from dmi.transport import native; host = native._load_named_extension(native._HOST_EXTENSION_NAME); assert host.DMXHostEngine.__module__ == '_host_backend'"
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(CURDIR)/src'); from dmi.api.v1 import DMXHostEngine; assert DMXHostEngine.__module__ in {'_host_backend', '_native_backend'}; assert 'dmi.transport.ring' not in sys.modules"

test-all:
	$(PYTHON) -m pytest -q

# Internal Python package-layout regression; this does not qualify a release artifact.
test-package:
	$(PYTHON) tests/tools/check_package.py

check-compile:
	$(PYTHON) -m compileall -q src/dmi tests benchmarks examples

check: check-compile test-cpu test-package
