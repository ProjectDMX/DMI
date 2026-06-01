// Lightweight NVTX shim: if nvToolsExt.h is available, enable NVTX ranges and
// thread naming; otherwise provide no-op stubs so instrumentation compiles away.

#ifndef MONITORING_NVTX_SHIM_H
#define MONITORING_NVTX_SHIM_H

#include <stdint.h>

#if defined(MON_NVTX_DISABLE)
// Forced off (e.g. CUDA 13 ships nvToolsExt.h headers but no libnvToolsExt.so,
// so __has_include would be true yet nvtxRangePushA is unresolvable). Build with
// NVTX=0 to take this no-op path.
#  define MON_NVTX_ENABLED 0
#elif defined(__has_include)
#  if __has_include(<nvToolsExt.h>)
#    include <nvToolsExt.h>
#    define MON_NVTX_ENABLED 1
#  else
#    define MON_NVTX_ENABLED 0
#  endif
#else
// Fallback: assume present on CUDA-enabled systems
#  include <nvToolsExt.h>
#  define MON_NVTX_ENABLED 1
#endif

#include <pthread.h>

#if MON_NVTX_ENABLED
#  include <sys/syscall.h>
#  include <unistd.h>
static inline void mon_nvtx_name_thread(const char* name) {
  // Name thread for both NVTX and pthread
  nvtxNameOsThreadA((unsigned int)syscall(SYS_gettid), name);
  (void)pthread_setname_np(pthread_self(), name);
}
static inline void mon_nvtx_push(const char* name) { nvtxRangePushA(name); }
static inline void mon_nvtx_pop(void) { nvtxRangePop(); }
#else
static inline void mon_nvtx_name_thread(const char* name) {
  // Still set pthread name for OS/thread timeline visibility
  (void)pthread_setname_np(pthread_self(), name);
  (void)name;
}
static inline void mon_nvtx_push(const char* name) { (void)name; }
static inline void mon_nvtx_pop(void) {}
#endif

#endif  // MONITORING_NVTX_SHIM_H

