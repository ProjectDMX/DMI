.PHONY: all monitoring

MONITORING_DIR ?= $(CURDIR)/monitoring

all: monitoring

monitoring:
	$(MAKE) -C $(MONITORING_DIR)