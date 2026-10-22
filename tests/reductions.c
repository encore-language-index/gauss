/* Fixed-order reduction, real worker dispatch and no small-input wakeups. */
#include "../runtime.c"
#include <assert.h>
static uint64_t bits(double value) {
  uint64_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}
static size_t exercise(size_t threads, uint64_t *results, bool compare) {
  char setting[24];
  snprintf(setting, sizeof(setting), "%zu", threads);
  assert(!setenv("GAUSS_NUM_THREADS", setting, 1));
  uint64_t session = gauss_session(-1);
  assert(session);
  GSession *s = g_lookup(session, 1);
  assert(s->pool.width == threads);
  const size_t sizes[] = {0,
                          1,
                          1024,
                          G_REDUCE_THRESHOLD - 1,
                          G_REDUCE_THRESHOLD,
                          G_REDUCE_THRESHOLD + 1,
                          1048576};
  size_t used = 0;
  for (int dtype = 0; dtype < 2; ++dtype) {
    for (size_t k = 0; k < sizeof(sizes) / sizeof(*sizes); ++k) {
      size_t n = sizes[k];
      uint64_t scope = gauss_scope(session, false);
      assert(scope);
      size_t shape[] = {n, 2}, axes[] = {1, 0};
      uint64_t base = gauss_full(scope, dtype, 2, shape, 0);
      assert(base);
      GTensor *t = g_lookup(base, 3);
      const double values[] = {1048576, .25, -1048576, .5};
      for (size_t i = 0; i < n * 2; ++i)
        g_write(t->storage->data, dtype, i, values[i % 4]);
      uint64_t sliced = gauss_slice(scope, base, 1, 1, 2, 1);
      uint64_t transposed = gauss_view(scope, base, 1, 2, axes);
      size_t broadcast_shape[] = {n, 7};
      uint64_t scalar = gauss_full(scope, dtype, 0, NULL, .25);
      uint64_t broadcast = gauss_view(scope, scalar, 2, 2, broadcast_shape);
      assert(sliced && transposed && broadcast);
      uint64_t inputs[] = {base, sliced, transposed, broadcast};
      for (size_t layout = 0; layout < 4; ++layout) {
        GTensor *input = g_lookup(inputs[layout], 3);
        for (int op = 0; op < 4; ++op) {
          if (!input->layout.count && op >= 2) {
            assert(!gauss_reduce(scope, inputs[layout], -1, op, false));
            assert(gauss_error() == G_SHAPE);
            continue;
          }
          size_t epoch = s->pool.epoch;
          uint64_t result = gauss_reduce(scope, inputs[layout], -1, op, true);
          assert(result);
          assert(s->pool.epoch ==
                 epoch + (threads > 1 &&
                          input->layout.count >= G_REDUCE_THRESHOLD));
          GTensor *out = g_lookup(result, 3);
          double actual = g_read(out->storage->data, dtype, 0);
          uint64_t value = isnan(actual) ? bits(NAN) : bits(actual);
          if (compare)
            assert(results[used] == value);
          else
            results[used] = value;
          ++used;
          /* Independent scalar reference, exact binary fractions here. */
          GTask serial = {out, input, NULL, input->layout, {0}, op, -1, true};
          g_reduce_worker(&serial, 0, 1);
          double expected = g_read(out->storage->data, dtype, 0);
          assert((isnan(actual) && isnan(expected)) || actual == expected);
        }
      }
      assert(gauss_close_scope(scope));
      assert(!gauss_bytes(session));
    }
    /* NaN and opposite infinities in distinct partials; cleanup remains safe.
     */
    for (int special = 0; special < 2; ++special) {
      uint64_t scope = gauss_scope(session, false);
      size_t n = G_REDUCE_THRESHOLD + 1;
      uint64_t a = gauss_full(scope, dtype, 1, &n, 1);
      assert(a);
      GTensor *t = g_lookup(a, 3);
      g_write(t->storage->data, dtype, 1, special ? INFINITY : NAN);
      g_write(t->storage->data, dtype, n - 1, -INFINITY);
      for (int op = 0; op < 4; ++op) {
        uint64_t r = gauss_reduce(scope, a, -1, op, false);
        assert(r);
        GTensor *out = g_lookup(r, 3);
        double v = g_read(out->storage->data, dtype, 0);
        if (!special || op < 2)
          assert(isnan(v));
        else
          assert(v == (op == 2 ? -INFINITY : INFINITY));
      }
      assert(gauss_close_scope(scope));
    }
  }
  assert(gauss_close_session(session));
  return used;
}
int main(void) {
  uint64_t expected[256];
  size_t count = exercise(1, expected, false);
  assert(exercise(4, expected, true) == count);
  assert(exercise(16, expected, true) == count);
  printf("CPU reductions: %zu cases x 3 worker counts, specials and dispatch "
         "passed\n",
         count);
}
