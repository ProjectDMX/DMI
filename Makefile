PYTHON ?= python

.PHONY: all native clean check check-compile test test-all test-cpu test-package

NATIVE_DIR ?= $(CURDIR)/native

all: native

native:
	$(MAKE) -C $(NATIVE_DIR)

clean:
	$(MAKE) -C $(NATIVE_DIR) clean

test: test-cpu

test-cpu:
	$(PYTHON) -m pytest -m cpu -q

test-all:
	$(PYTHON) -m pytest -q

# Internal Python package-layout regression; this does not qualify a release artifact.
test-package:
	$(PYTHON) tests/tools/check_package.py

check-compile:
	$(PYTHON) -m compileall -q src/dmi tests benchmarks examples

check: check-compile test-cpu test-package
