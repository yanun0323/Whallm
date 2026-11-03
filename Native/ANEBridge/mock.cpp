#include <cstddef>
#include <cstdint>
#include <cstdlib>

static int created = 0, destroyed = 0;

extern "C" {
void* whallm_ane_projection_create(const uint16_t*, size_t, int, int, int) {
  const char* fail_after = getenv("WHALLM_MOCK_ANE_CREATE_FAIL_AFTER");
  if (fail_after && created >= atoi(fail_after)) return nullptr;
  ++created;
  return malloc(1);
}
int whallm_ane_projection_evaluate(void*, const uint16_t*, size_t, uint16_t* output, size_t length) {
  if (getenv("WHALLM_MOCK_ANE_EVAL_FAIL")) return -1;
  if (getenv("WHALLM_MOCK_ANE_NONFINITE") && length) output[0] = 0x7c00;
  return 0;
}
void whallm_ane_projection_destroy(void* handle) {
  free(handle);
  ++destroyed;
}
const char* whallm_ane_last_error() { return "injected ANE failure"; }
int whallm_mock_ane_live() { return created - destroyed; }
}
