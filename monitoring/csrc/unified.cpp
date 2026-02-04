// Single Translation Unit build to maximize inlining across components.
// Compile this file alone to include all implementation parts.

#include "native_engine.cpp"
#include "engine_core.cpp"
#include "api_submit.cpp"
#include "hooks.cpp"
#include "slice.cpp"
// dmx_host pipeline bits
#include "dmx_host_utils.cpp"
#include "future_process.cpp"
#include "clickhouse_client.cpp"
#include "bindings.cpp"

