#include "../runtime.c"
#include <assert.h>
#include <time.h>
static uint64_t checked(uint64_t id) {
  if (!id) {
    fprintf(stderr, "Gauss error %u\n", gauss_error());
    abort();
  }
  return id;
}
static void matrix(int device) {
  uint64_t session = checked(gauss_session(device)),
           scope = checked(gauss_scope(session, false));
  size_t sa[] = {2, 3}, sb[] = {3, 2};
  float values[4];
  uint64_t a = checked(gauss_full(scope, 0, 2, sa, 2)),
           b = checked(gauss_full(scope, 0, 2, sb, 1));
  uint64_t c = checked(gauss_matmul(scope, a, b));
  assert(gauss_read(c, 0, values, 4));
  for (size_t i = 0; i < 4; ++i)
    assert(values[i] == 6);
  uint64_t r = checked(gauss_reduce(scope, c, -1, 0, false));
  float sum;
  assert(gauss_read(r, 0, &sum, 1));
  assert(sum == 24);
  assert(gauss_close_scope(scope));
  assert(gauss_bytes(session) == 0);
  assert(!gauss_valid(a));
  assert(gauss_close_session(session));
}
static void lifetime(void) {
  uint64_t session = checked(gauss_session(-1)),
           root = checked(gauss_scope(session, false));
  size_t shape[] = {4};
  for (int i = 0; i < 1000; ++i) {
    uint64_t child = checked(gauss_scope(root, true));
    uint64_t a = checked(gauss_full(child, 1, 1, shape, 3));
    assert(gauss_close_scope(child));
    assert(!gauss_valid(a));
    assert(gauss_bytes(session) == 0);
  }
  uint64_t child = checked(gauss_scope(root, true)),
           a = checked(gauss_full(child, 1, 1, shape, 3));
  uint64_t b = checked(gauss_export(child, a));
  assert(gauss_close_scope(child));
  assert(!gauss_valid(a));
  double values[4];
  assert(gauss_read(b, 1, values, 4));
  assert(values[0] == 3);
  assert(gauss_close_session(session));
  assert(!gauss_valid(b));
  assert(!g_live);
  g_registry_trim();
}
static void deep_scopes(void) {
  uint64_t session = checked(gauss_session(-1));
  uint64_t parent = checked(gauss_scope(session, false));
  for (size_t depth = 0; depth < 10000; ++depth)
    parent = checked(gauss_scope(parent, true));
  size_t shape[] = {1};
  uint64_t leaf = checked(gauss_full(parent, 0, 1, shape, 1));
  assert(gauss_close_session(session));
  assert(!gauss_valid(leaf) && !g_live);
}
int main(int argc, char **argv) {
  (void)argv;
  matrix(-1);
  lifetime();
  deep_scopes();
  if (argc > 1)
    matrix(0);
  puts("native lifetime and matrix smoke: ok");
  return 0;
}
