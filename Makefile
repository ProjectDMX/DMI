.PHONY: all monitoring clean

MONITORING_DIR ?= $(CURDIR)/monitoring

all: monitoring

monitoring:
	$(MAKE) -C $(MONITORING_DIR)

clean:
	$(MAKE) -C $(MONITORING_DIR) clean