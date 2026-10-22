/* Fault injection is test-local: no environment-controlled production hooks. */
#include <errno.h>
#include <stdlib.h>
static size_t allocation, fail_at;
static int fail(void) { return fail_at && ++allocation == fail_at; }
static void *test_malloc(size_t n) { return fail() ? NULL : malloc(n); }
static void *test_calloc(size_t n, size_t size) {
  return fail() ? NULL : calloc(n, size);
}
static void *test_realloc(void *p, size_t size) {
  return fail() ? NULL : realloc(p, size);
}
static int test_aligned(void **p, size_t align, size_t size) {
  return fail() ? ENOMEM : posix_memalign(p, align, size);
}
#define malloc test_malloc
#define calloc test_calloc
#define realloc test_realloc
#define posix_memalign test_aligned
#include "../runtime.c"
#undef malloc
#undef calloc
#undef realloc
#undef posix_memalign
#include <assert.h>

static void allocation_failures(void) {
  size_t shape[] = {2, 3};
  int completed = 0;
  for (size_t failure = 1; failure <= 32; ++failure) {
    allocation = 0;
    fail_at = failure;
    uint64_t s = gauss_session(-1);
    if (!s) {
      assert(gauss_error() == G_OOM);
      assert(!g_live);
      g_registry_trim();
      continue;
    }
    uint64_t scope = gauss_scope(s, false), a = 0, b = 0, copy = 0;
    if (scope)
      a = gauss_full(scope, 1, 2, shape, 2);
    if (a)
      b = gauss_map(scope, a, a, 1, -1);
    if (b)
      copy = gauss_to_device(scope, b);
    if (!copy)
      assert(gauss_error() == G_OOM);
    else
      ++completed;
    fail_at = 0;
    assert(gauss_close_session(s));
    assert(!g_live && !g_slots);
  }
  assert(completed);
}

static int wait_failure(void *context) {
  (void)context;
  return 719;
}
static int wait_success(void *context) {
  (void)context;
  return 0;
}
static int release_failure(void *context, void *p) {
  (void)context;
  free(p);
  return 719;
}
static void backend_failures(void) {
  /* Model a deferred CUDA error without poisoning the physical GPU context. */
  GSession *s = calloc(1, sizeof(*s));
  GCuda fake = {0};
  fake.wait = wait_failure;
  fake.release = release_failure;
  s->cuda = &fake;
  s->device = 0;
  s->id = g_register(1, s);
  uint64_t scope = gauss_scope(s->id, false);
  assert(scope);
  assert(!gauss_finish(s->id) && gauss_error() == G_BACKEND);
  assert(!gauss_scope(s->id, false) && gauss_error() == G_BACKEND);
  size_t shape[] = {2};
  assert(!gauss_full(scope, 0, 1, shape, 1) && gauss_error() == G_BACKEND);
  fake.wait = wait_success;
  g_error = G_AXIS; /* an unrelated validation error must not mask poison */
  assert(!gauss_finish(s->id) && gauss_error() == G_BACKEND);
  assert(!gauss_close_scope(scope) && gauss_error() == G_BACKEND);
  g_unregister((uint32_t)s->id - 1);
  free(s);
  g_registry_trim();
  assert(!g_live);
}
int main(void) {
  allocation_failures();
  backend_failures();
  puts("allocation rollback and deferred backend failures: ok");
}
