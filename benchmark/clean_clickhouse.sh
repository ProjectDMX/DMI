#!/bin/bash
set -e

CLICKHOUSE_DATA_DIR="${CLICKHOUSE_DATA_DIR:-$(clickhouse extract-from-config --key=path 2>/dev/null || true)}"

if [ -z "${CLICKHOUSE_DATA_DIR}" ]; then
    echo "ERROR: unable to detect ClickHouse data dir automatically."
    echo "Set CLICKHOUSE_DATA_DIR to your ClickHouse data directory and retry."
    exit 1
fi

case "${CLICKHOUSE_DATA_DIR}" in
    "."|"..")
        echo "ERROR: refusing to operate on unsafe CLICKHOUSE_DATA_DIR=${CLICKHOUSE_DATA_DIR}"
        exit 1
        ;;
esac

RESOLVED_CLICKHOUSE_DATA_DIR="$(realpath "${CLICKHOUSE_DATA_DIR}" 2>/dev/null || true)"
if [ -z "${RESOLVED_CLICKHOUSE_DATA_DIR}" ]; then
    echo "ERROR: unable to resolve CLICKHOUSE_DATA_DIR to an absolute path: ${CLICKHOUSE_DATA_DIR}"
    exit 1
fi

case "${RESOLVED_CLICKHOUSE_DATA_DIR}" in
    ""|"/"|"."|"..")
        echo "ERROR: refusing to operate on unsafe CLICKHOUSE_DATA_DIR=${RESOLVED_CLICKHOUSE_DATA_DIR}"
        exit 1
        ;;
    /*)
        ;;
    *)
        echo "ERROR: CLICKHOUSE_DATA_DIR must resolve to an absolute path: ${RESOLVED_CLICKHOUSE_DATA_DIR}"
        exit 1
        ;;
esac

CLICKHOUSE_DATA_DIR="${RESOLVED_CLICKHOUSE_DATA_DIR}"

if [ ! -d "${CLICKHOUSE_DATA_DIR}" ]; then
    echo "ERROR: CLICKHOUSE_DATA_DIR does not exist or is not a directory: ${CLICKHOUSE_DATA_DIR}"
    exit 1
fi

echo "=== Stopping ClickHouse ==="
sudo systemctl stop clickhouse-server
sleep 2

echo "=== Clearing ${CLICKHOUSE_DATA_DIR}/ ==="
sudo rm -rf "${CLICKHOUSE_DATA_DIR}"/*

echo "=== Verifying ==="
if [ -z "$(ls -A "${CLICKHOUSE_DATA_DIR}/")" ]; then
    echo "Data directory is clean."
else
    echo "WARNING: directory not empty!"
    ls -la "${CLICKHOUSE_DATA_DIR}/"
fi

echo "=== Restarting ClickHouse ==="
sudo systemctl start clickhouse-server
sleep 2

echo "=== Status ==="
sudo systemctl status clickhouse-server --no-pager | head -5
echo ""
echo "Done."
