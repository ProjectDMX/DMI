// Single Translation Unit build to maximize inlining across components.
// Compile this file alone to include all implementation parts.

#include "native_engine.cpp"
#include "engine_core.cpp"
#include "api_submit.cpp"
#include "hooks.cpp"
#include "slice.cpp"
#include "graph_native_delegate.cpp"
#include "graph_shadow_parser.cpp"
#include "bindings.cpp"
