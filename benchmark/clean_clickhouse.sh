#!/bin/bash
set -e

echo "=== Stopping ClickHouse ==="
sudo systemctl stop clickhouse-server
sleep 2

echo "=== Clearing /var/lib/clickhouse/ ==="
sudo rm -rf /var/lib/clickhouse/*

echo "=== Verifying ==="
if [ -z "$(ls -A /var/lib/clickhouse/)" ]; then
    echo "Data directory is clean."
else
    echo "WARNING: directory not empty!"
    ls -la /var/lib/clickhouse/
fi

echo "=== Restarting ClickHouse ==="
sudo systemctl start clickhouse-server
sleep 2

echo "=== Status ==="
sudo systemctl status clickhouse-server --no-pager | head -5
echo ""
echo "Done."