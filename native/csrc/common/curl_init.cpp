#include "curl_init.h"

#include <mutex>

#include <curl/curl.h>

namespace dmi_common {

void EnsureCurlGlobalInit() {
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

}  // namespace dmi_common
