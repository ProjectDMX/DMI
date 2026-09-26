#include "curl_init.h"

#include <mutex>
#include <stdexcept>
#include <string>

#include <curl/curl.h>

namespace dmi_common {

void EnsureCurlGlobalInit() {
  static std::once_flag once;
  static CURLcode status = CURLE_OK;
  std::call_once(once, [] { status = curl_global_init(CURL_GLOBAL_DEFAULT); });
  // call_once only means "ran"; it says nothing about whether the init
  // worked. Discarding the code turned a failed process-global init into
  // misleading downstream handle/transport errors, so the failure is
  // propagated to the constructing client instead.
  if (status != CURLE_OK) {
    throw std::runtime_error(
        std::string("curl_global_init failed: ") +
        curl_easy_strerror(status));
  }
}

}  // namespace dmi_common
