// One process-lifetime libcurl initialization, shared by every client.
//
// libcurl's global init/cleanup pair is PROCESS-global and documented as
// unsafe to call while any other thread is inside the library. A client
// that initialized in its constructor and cleaned up in its destructor
// therefore tore the library down under whatever else was using it: the
// uploader runs up to `max_workers` concurrent `curl_easy_perform` calls,
// and destroying an unrelated client on the main thread dropped the
// refcount to zero beneath them — after which those workers' implicit
// re-initialization inside `curl_easy_init` raced.
//
// So: initialize once, never clean up. The OS reclaims libcurl's
// allocations at exit, which is what a process-lifetime dependency is for;
// there is no correct moment for a library-wide teardown in a process that
// still has threads. Every client calls this from its constructor rather
// than relying on the implicit init inside `curl_easy_init`, which carries
// the same thread-safety caveat.
//
// Header-level curl dependency deliberately avoided: <curl/curl.h> needs
// CURL_INCDIR on the compile line, and this declaration should be usable
// from anywhere in the tree.

#ifndef DMI_COMMON_CURL_INIT_H_
#define DMI_COMMON_CURL_INIT_H_

namespace dmi_common {

// Runs curl_global_init(CURL_GLOBAL_DEFAULT) exactly once per process.
// Safe to call from any thread, any number of times.
void EnsureCurlGlobalInit();

}  // namespace dmi_common

#endif  // DMI_COMMON_CURL_INIT_H_
