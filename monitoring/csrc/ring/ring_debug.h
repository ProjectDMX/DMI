// ring/ring_debug.h -- Compile-time debug logging for ring transport.
//
// Set RING_DEBUG to 1 at compile time to enable verbose logging.
// When 0 (default), all RING_DBG calls are compiled out completely.
//
// Build with debug:  make -C monitoring/ CXXFLAGS+=-DRING_DEBUG=1 NVCCFLAGS+=-DRING_DEBUG=1

#pragma once
#include <cstdio>

#ifndef RING_DEBUG
#define RING_DEBUG 0
#endif

#if RING_DEBUG
#define RING_DBG(fmt, ...) \
    do { \
        fprintf(stderr, fmt, ##__VA_ARGS__); \
        fflush(stderr); \
    } while (0)
#else
#define RING_DBG(fmt, ...) ((void)0)
#endif

namespace ring {
inline constexpr bool ring_debug_enabled() { return RING_DEBUG != 0; }
}  // namespace ring
