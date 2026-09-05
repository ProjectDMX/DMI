// sha256 in-process with best-of-5, like the standalone
#include <openssl/sha.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
using Clock = std::chrono::steady_clock;
static double gib_per_s(uint64_t b, double s){return (double)b/s/1073741824.0;}
int main(){
  const size_t N=10'000, P=64*1024; const uint64_t logical=(uint64_t)N*P;
  std::vector<std::vector<uint8_t>> payloads(N, std::vector<uint8_t>(P,0xAB));
  unsigned char digest[SHA256_DIGEST_LENGTH];
  double best=1e9;
  for(int t=0;t<5;++t){
    auto started=Clock::now();
    SHA256_CTX ctx; SHA256_Init(&ctx);
    for(const auto& p: payloads) SHA256_Update(&ctx,p.data(),p.size());
    SHA256_Final(digest,&ctx);
    best=std::min(best,std::chrono::duration<double>(Clock::now()-started).count());
  }
  printf("sha256_seal best-of-5: %8.3f GiB/s (%02x%02x)\n",gib_per_s(logical,best),digest[0],digest[1]);
}
