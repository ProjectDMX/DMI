.PHONY: all native clean

NATIVE_DIR ?= $(CURDIR)/native

all: native

native:
	$(MAKE) -C $(NATIVE_DIR)

clean:
	$(MAKE) -C $(NATIVE_DIR) clean
