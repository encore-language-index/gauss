#ifndef GAUSS_RUNTIME_H
#define GAUSS_RUNTIME_H
#include "api.h"
#include "pool.h"
#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

enum {
  G_OK,
  G_CLOSED,
  G_SHAPE,
  G_DTYPE,
  G_DEVICE,
  G_OOM,
  G_BACKEND,
  G_AXIS,
  G_LAYOUT,
  G_OVERFLOW,
  G_SCOPE,
  G_DEPENDENCY
};
static _Thread_local int g_error;
static _Atomic uint32_t g_generation = 1;
typedef struct {
  uint32_t gen;
  int kind;
  void *ptr;
} GSlot;
static _Thread_local GSlot *g_slots;
static _Thread_local size_t g_capacity, g_live, g_hint;
typedef struct GSession GSession;
typedef struct GScope GScope;
typedef struct {
  GSession *session;
  void *data;
  size_t refs, bytes;
} GStorage;
typedef struct {
  GScope *scope;
  GStorage *storage;
  GLayout layout;
} GTensor;
struct GScope {
  GSession *session;
  GScope *parent;
  uint64_t id;
  bool closing;
};
typedef void (*GSgemm)(int, int, int, int, int, int, float, const float *, int,
                       const float *, int, float, float *, int);
typedef void (*GDgemm)(int, int, int, int, int, int, double, const double *,
                       int, const double *, int, double, double *, int);
struct GSession {
  uint64_t id;
  int device, failed;
  size_t bytes;
  GPool pool;
  void *library, *context;
  const GCuda *cuda;
  GSgemm sgemm;
  GDgemm dgemm;
};
static uint64_t g_fail(int code) {
  g_error = code;
  return 0;
}
uint32_t gauss_error(void) { return (uint32_t)g_error; }
static uint64_t g_register(int kind, void *ptr) {
  size_t i = g_hint;
  while (i < g_capacity && g_slots[i].ptr)
    ++i;
  if (i == g_capacity) {
    size_t cap = g_capacity ? g_capacity * 2 : 64;
    if (cap > UINT32_MAX)
      return g_fail(G_OOM);
    GSlot *next = realloc(g_slots, cap * sizeof(*next));
    if (!next)
      return g_fail(G_OOM);
    memset(next + g_capacity, 0, (cap - g_capacity) * sizeof(*next));
    g_slots = next;
    g_capacity = cap;
  }
  uint32_t generation = atomic_fetch_add(&g_generation, 1);
  /* Never reuse a generation after wrap-around (even across threads). */
  if (!generation || generation == UINT32_MAX)
    abort();
  g_slots[i] = (GSlot){generation, kind, ptr};
  g_hint = i + 1;
  ++g_live;
  return ((uint64_t)generation << 32) | (uint64_t)(i + 1);
}
static void *g_lookup(uint64_t id, int kind) {
  size_t slot = (size_t)(uint32_t)id;
  if (!slot || --slot >= g_capacity || g_slots[slot].gen != id >> 32 ||
      g_slots[slot].kind != kind || !g_slots[slot].ptr) {
    g_fail(G_CLOSED);
    return NULL;
  }
  return g_slots[slot].ptr;
}
static void g_unregister(size_t slot) {
  g_slots[slot].ptr = NULL;
  if (slot < g_hint)
    g_hint = slot;
  --g_live;
}
static void g_registry_trim(void) {
  if (!g_live) {
    free(g_slots);
    g_slots = NULL;
    g_capacity = 0;
    g_hint = 0;
  }
}
static bool g_mul(size_t a, size_t b, size_t *out) {
  if (b && a > SIZE_MAX / b) {
    g_fail(G_OVERFLOW);
    return false;
  }
  *out = a * b;
  return true;
}
static size_t g_item(int dtype) {
  return dtype == 0 ? sizeof(float) : sizeof(double);
}
static bool g_layout(GLayout *l, int dtype, size_t rank, const size_t *shape) {
  if (dtype != 0 && dtype != 1) {
    g_fail(G_DTYPE);
    return false;
  }
  if (rank > G_RANK || (rank && !shape)) {
    g_fail(G_SHAPE);
    return false;
  }
  memset(l, 0, sizeof(*l));
  l->dtype = dtype;
  l->rank = rank;
  size_t count = 1;
  for (size_t i = rank; i-- > 0;) {
    l->shape[i] = shape[i];
    l->stride[i] = count;
    if (!g_mul(count, shape[i], &count))
      return false;
  }
  size_t bytes;
  if (!g_mul(count, g_item(dtype), &bytes))
    return false;
  l->count = count;
  return true;
}
static bool g_contiguous(GLayout l) {
  size_t stride = 1;
  for (size_t i = l.rank; i-- > 0;) {
    if (l.shape[i] > 1 && l.stride[i] != stride)
      return false;
    stride *= l.shape[i];
  }
  return true;
}
static size_t g_index(GLayout l, size_t i) {
  size_t offset = l.offset;
  for (size_t k = l.rank; k-- > 0;) {
    offset += (i % l.shape[k]) * l.stride[k];
    i /= l.shape[k];
  }
  return offset;
}
static double g_read(const void *p, int dtype, size_t i) {
  return dtype ? ((const double *)p)[i] : ((const float *)p)[i];
}
static void g_write(void *p, int dtype, size_t i, double v) {
  if (dtype)
    ((double *)p)[i] = v;
  else
    ((float *)p)[i] = (float)v;
}
static bool g_cuda_ok(GSession *s, int code) {
  if (!code)
    return true;
  s->failed = code;
  g_fail(G_BACKEND);
  return false;
}
static bool g_wait(GSession *s) {
  bool ok = s->device < 0 || g_cuda_ok(s, s->cuda->wait(s->context));
  if (s->failed) {
    g_fail(G_BACKEND);
    return false;
  }
  return ok;
}
static void g_storage_drop(GStorage *b) {
  if (--b->refs)
    return;
  GSession *s = b->session;
  if (s->device < 0)
    free(b->data);
  else
    g_cuda_ok(s, s->cuda->release(s->context, b->data));
  s->bytes -= b->bytes;
  free(b);
}
static bool g_ancestor(GScope *parent, GScope *scope) {
  for (; scope; scope = scope->parent)
    if (parent == scope)
      return true;
  return false;
}
static GTensor *g_input(GScope *scope, uint64_t id) {
  GTensor *t = g_lookup(id, 3);
  if (!t)
    return NULL;
  if (t->scope->session != scope->session) {
    g_fail(G_DEVICE);
    return NULL;
  }
  if (!g_ancestor(t->scope, scope)) {
    g_fail(G_SCOPE);
    return NULL;
  }
  return t;
}
static uint64_t g_tensor(GScope *scope, GLayout layout, GStorage *storage) {
  if (scope->session->failed)
    return g_fail(G_BACKEND);
  GTensor *t = calloc(1, sizeof(*t));
  if (!t)
    return g_fail(G_OOM);
  if (!storage) {
    GSession *s = scope->session;
    size_t bytes;
    if (!g_mul(layout.count, g_item(layout.dtype), &bytes) ||
        bytes > SIZE_MAX - s->bytes) {
      free(t);
      return g_fail(G_OVERFLOW);
    }
    storage = calloc(1, sizeof(*storage));
    if (!storage) {
      free(t);
      return g_fail(G_OOM);
    }
    storage->session = s;
    storage->bytes = bytes;
    if (s->device < 0) {
      if (bytes && posix_memalign(&storage->data, 64, bytes)) {
        free(storage);
        free(t);
        return g_fail(G_OOM);
      }
    } else if (!g_cuda_ok(
                   s, s->cuda->allocate(s->context, bytes, &storage->data))) {
      free(storage);
      free(t);
      return 0;
    }
    s->bytes += bytes;
  }
  ++storage->refs;
  t->scope = scope;
  t->layout = layout;
  t->storage = storage;
  uint64_t id = g_register(3, t);
  if (!id) {
    g_storage_drop(storage);
    free(t);
  }
  return id;
}
static void g_discard(uint64_t id) {
  GTensor *t = g_lookup(id, 3);
  if (!t)
    return;
  g_storage_drop(t->storage);
  free(t);
  g_unregister((uint32_t)id - 1);
}
uint64_t gauss_session(int32_t device) {
  g_error = 0;
  if (device < -1)
    return g_fail(G_DEVICE);
  GSession *s = calloc(1, sizeof(*s));
  if (!s)
    return g_fail(G_OOM);
  s->device = device;
  if (device >= 0) {
    const char *path = getenv("GAUSS_CUDA_LIBRARY");
    s->library =
        dlopen(path ? path : "libgauss_cuda.so", RTLD_NOW | RTLD_LOCAL);
    const GCuda *(*api)(void) =
        s->library ? (const GCuda *(*)(void))dlsym(s->library, "gauss_cuda_api")
                   : NULL;
    s->cuda = api ? api() : NULL;
    if (!s->cuda || s->cuda->abi != G_PLUGIN_ABI) {
      if (s->library)
        dlclose(s->library);
      free(s);
      return g_fail(G_DEPENDENCY);
    }
    if (s->cuda->open(device, &s->context)) {
      dlclose(s->library);
      free(s);
      return g_fail(G_BACKEND);
    }
  } else {
    const char *path = getenv("GAUSS_BLAS_LIBRARY");
    s->library = dlopen(path ? path : "libopenblas.so", RTLD_NOW | RTLD_LOCAL);
    if (!s->library && !path)
      s->library = dlopen("libopenblas.so.0", RTLD_NOW | RTLD_LOCAL);
    s->sgemm = s->library ? (GSgemm)dlsym(s->library, "cblas_sgemm") : NULL;
    s->dgemm = s->library ? (GDgemm)dlsym(s->library, "cblas_dgemm") : NULL;
    const char *(*config)(void) =
        s->library
            ? (const char *(*)(void))dlsym(s->library, "openblas_get_config")
            : NULL;
    if (!s->sgemm || !s->dgemm || !config || strstr(config(), "USE64BITINT")) {
      if (s->library)
        dlclose(s->library);
      free(s);
      return g_fail(G_DEPENDENCY);
    }
    g_pool_init(&s->pool);
  }
  s->id = g_register(1, s);
  if (!s->id) {
    if (device < 0)
      g_pool_close(&s->pool);
    else
      s->cuda->close(s->context);
    dlclose(s->library);
    free(s);
    return 0;
  }
  return s->id;
}
uint64_t gauss_scope(uint64_t owner, bool child) {
  GScope *parent = child ? g_lookup(owner, 2) : NULL;
  GSession *s = child ? (parent ? parent->session : NULL) : g_lookup(owner, 1);
  if (!s)
    return 0;
  if (s->failed)
    return g_fail(G_BACKEND);
  GScope *scope = calloc(1, sizeof(*scope));
  if (!scope)
    return g_fail(G_OOM);
  scope->parent = parent;
  scope->session = s;
  scope->id = g_register(2, scope);
  if (!scope->id) {
    free(scope);
    return 0;
  }
  return scope->id;
}
static void g_scope_destroy(GScope *scope) {
  /* Mark before freeing any parent pointer. Cleanup must not recurse on the
   * native stack: users can construct arbitrary depth with child() in a loop.
   */
  for (size_t i = 0; i < g_capacity; ++i)
    if (g_slots[i].ptr && g_slots[i].kind == 2) {
      GScope *child = g_slots[i].ptr;
      child->closing = g_ancestor(scope, child);
    }
  for (size_t i = 0; i < g_capacity; ++i)
    if (g_slots[i].ptr && g_slots[i].kind == 3) {
      GTensor *t = g_slots[i].ptr;
      if (t->scope->closing) {
        g_storage_drop(t->storage);
        free(t);
        g_unregister(i);
      }
    }
  for (size_t i = 0; i < g_capacity; ++i)
    if (g_slots[i].ptr && g_slots[i].kind == 2) {
      GScope *child = g_slots[i].ptr;
      if (child->closing) {
        free(child);
        g_unregister(i);
      }
    }
}
bool gauss_close_scope(uint64_t id) {
  GScope *scope = g_lookup(id, 2);
  if (!scope)
    return false;
  GSession *s = scope->session;
  bool ok = g_wait(s);
  g_scope_destroy(scope);
  if (s->failed)
    g_fail(G_BACKEND);
  return ok && !s->failed;
}
bool gauss_finish(uint64_t id) {
  GSession *s = g_lookup(id, 1);
  return s && g_wait(s) && !s->failed;
}
bool gauss_close_session(uint64_t id) {
  GSession *s = g_lookup(id, 1);
  if (!s)
    return false;
  bool ok = g_wait(s);
  for (size_t i = 0; i < g_capacity; ++i)
    if (g_slots[i].ptr && g_slots[i].kind == 2) {
      GScope *scope = g_slots[i].ptr;
      if (scope->session == s)
        g_scope_destroy(scope);
    }
  if (s->device < 0)
    g_pool_close(&s->pool);
  else
    ok = g_cuda_ok(s, s->cuda->close(s->context)) && ok;
  ok = ok && !s->failed;
  dlclose(s->library);
  free(s);
  g_unregister((uint32_t)id - 1);
  g_registry_trim();
  if (!ok)
    g_fail(G_BACKEND);
  return ok;
}
/* Automatic cleanup is intentionally fail-fast: ContextManager cannot return
 * Result. Call finish()/close() explicitly to handle errors instead. */
bool gauss_exit(uint64_t id, bool session) {
  if (!g_lookup(id, session ? 1 : 2))
    return true; /* already explicitly closed */
  bool ok = session ? gauss_close_session(id) : gauss_close_scope(id);
  if (!ok) {
    fputs("Gauss: unhandled backend error during scope cleanup\n", stderr);
    abort();
  }
  return true;
}
size_t gauss_bytes(uint64_t id) {
  GSession *s = g_lookup(id, 1);
  return s ? s->bytes : 0;
}
uint64_t gauss_export(uint64_t scope_id, uint64_t tensor_id) {
  GScope *scope = g_lookup(scope_id, 2);
  GTensor *t = g_lookup(tensor_id, 3);
  if (!scope || !t)
    return 0;
  if (t->scope != scope || !scope->parent)
    return g_fail(G_SCOPE);
  return g_tensor(scope->parent, t->layout, t->storage);
}
uint64_t gauss_full(uint64_t scope_id, int32_t dtype, size_t rank,
                    const size_t *shape, double value) {
  GScope *scope = g_lookup(scope_id, 2);
  GLayout l;
  if (!scope || !g_layout(&l, dtype, rank, shape))
    return 0;
  uint64_t id = g_tensor(scope, l, NULL);
  if (!id)
    return 0;
  GTensor *t = g_lookup(id, 3);
  GSession *s = scope->session;
  if (s->device < 0) {
    for (size_t i = 0; i < l.count; ++i)
      g_write(t->storage->data, dtype, i, value);
  } else if (!g_cuda_ok(
                 s, s->cuda->fill(s->context, t->storage->data, l, value))) {
    g_discard(id);
    return 0;
  }
  return id;
}
uint64_t gauss_from(uint64_t scope_id, int32_t dtype, size_t rank,
                    const size_t *shape, const void *data, size_t count) {
  GScope *scope = g_lookup(scope_id, 2);
  GLayout l;
  if (!scope || !g_layout(&l, dtype, rank, shape))
    return 0;
  if (count != l.count || (count && !data))
    return g_fail(G_SHAPE);
  uint64_t id = g_tensor(scope, l, NULL);
  if (!id)
    return 0;
  GTensor *t = g_lookup(id, 3);
  GSession *s = scope->session;
  size_t bytes = count * g_item(dtype);
  if (s->device < 0) {
    if (bytes)
      memcpy(t->storage->data, data, bytes);
  } else if (!g_cuda_ok(s, s->cuda->upload(s->context, t->storage->data, data,
                                           bytes))) {
    g_discard(id);
    return 0;
  }
  return id;
}
bool gauss_info(uint64_t id, size_t *rank, size_t *count, int32_t *dtype,
                size_t *shape) {
  GTensor *t = g_lookup(id, 3);
  if (!t)
    return false;
  *rank = t->layout.rank;
  *count = t->layout.count;
  *dtype = t->layout.dtype;
  if (shape)
    memcpy(shape, t->layout.shape, t->layout.rank * sizeof(size_t));
  return true;
}
bool gauss_valid(uint64_t id) { return g_lookup(id, 3) != NULL; }
size_t gauss_meta(uint64_t id, int32_t key, size_t index) {
  GTensor *t = g_lookup(id, 3);
  if (!t)
    return 0;
  if (key == 0)
    return t->layout.rank;
  if (key == 1)
    return t->layout.count;
  if (key == 2)
    return (size_t)t->layout.dtype;
  if (index >= t->layout.rank) {
    g_fail(G_AXIS);
    return 0;
  }
  return key == 3 ? t->layout.shape[index] : t->layout.stride[index];
}
static bool g_broadcast(GLayout in, size_t rank, const size_t *shape,
                        GLayout *out) {
  if (in.rank > rank || rank > G_RANK) {
    g_fail(G_SHAPE);
    return false;
  }
  if (!g_layout(out, in.dtype, rank, shape))
    return false;
  out->offset = in.offset;
  for (size_t i = 0; i < rank; ++i) {
    if (i < rank - in.rank)
      out->stride[i] = 0;
    else {
      size_t j = i - (rank - in.rank);
      if (in.shape[j] != shape[i] && in.shape[j] != 1) {
        g_fail(G_SHAPE);
        return false;
      }
      out->stride[i] = in.shape[j] == 1 ? 0 : in.stride[j];
    }
  }
  return true;
}
uint64_t gauss_view(uint64_t scope_id, uint64_t tensor_id, int32_t op,
                    size_t rank, const size_t *args) {
  GScope *scope = g_lookup(scope_id, 2);
  if (!scope)
    return 0;
  GTensor *t = g_input(scope, tensor_id);
  if (!t)
    return 0;
  GLayout l = t->layout;
  if (rank > G_RANK || (rank && !args))
    return g_fail(G_SHAPE);
  if (op == 0) {
    if (!g_contiguous(l))
      return g_fail(G_LAYOUT);
    if (!g_layout(&l, t->layout.dtype, rank, args))
      return 0;
    if (l.count != t->layout.count)
      return g_fail(G_SHAPE);
    l.offset = t->layout.offset;
  } else if (op == 1) {
    if (rank != l.rank)
      return g_fail(G_SHAPE);
    bool used[G_RANK] = {0};
    for (size_t i = 0; i < rank; ++i) {
      if (args[i] >= rank || used[args[i]])
        return g_fail(G_AXIS);
      used[args[i]] = true;
      l.shape[i] = t->layout.shape[args[i]];
      l.stride[i] = t->layout.stride[args[i]];
    }
  } else if (op == 2) {
    if (!g_broadcast(t->layout, rank, args, &l))
      return 0;
  } else
    return g_fail(G_LAYOUT);
  return g_tensor(scope, l, t->storage);
}
uint64_t gauss_slice(uint64_t scope_id, uint64_t tensor_id, size_t axis,
                     size_t start, size_t end, size_t step) {
  GScope *scope = g_lookup(scope_id, 2);
  if (!scope)
    return 0;
  GTensor *t = g_input(scope, tensor_id);
  if (!t)
    return 0;
  GLayout l = t->layout;
  if (axis >= l.rank)
    return g_fail(G_AXIS);
  if (!step || start > end || end > l.shape[axis])
    return g_fail(G_SHAPE);
  size_t stride;
  if (!g_mul(l.stride[axis], step, &stride))
    return 0;
  l.offset += start * l.stride[axis];
  l.stride[axis] = stride;
  l.shape[axis] = (end - start) / step + ((end - start) % step != 0);
  l.count = 1;
  for (size_t i = 0; i < l.rank; ++i)
    l.count *= l.shape[i];
  return g_tensor(scope, l, t->storage);
}
typedef struct {
  GTensor *out, *a, *b;
  GLayout al, bl;
  int op, axis, keep;
} GTask;
static void g_map_worker(void *arg, size_t begin, size_t end) {
  GTask *q = arg;
  bool ac = g_contiguous(q->al), bc = !q->b || g_contiguous(q->bl);
  for (size_t i = begin; i < end; ++i) {
    double a = g_read(q->a->storage->data, q->al.dtype,
                      ac ? q->al.offset + i : g_index(q->al, i));
    double b = q->b ? g_read(q->b->storage->data, q->bl.dtype,
                             bc ? q->bl.offset + i : g_index(q->bl, i))
                    : 0;
    double v = a;
    switch (q->op) {
    case 1:
      v = a + b;
      break;
    case 2:
      v = a - b;
      break;
    case 3:
      v = a * b;
      break;
    case 4:
      v = a / b;
      break;
    case 5:
      v = -a;
      break;
    case 6:
      v = fabs(a);
      break;
    case 7:
      v = exp(a);
      break;
    case 8:
      v = log(a);
      break;
    case 9:
      v = sqrt(a);
      break;
    }
    g_write(q->out->storage->data, q->out->layout.dtype, i, v);
  }
}
uint64_t gauss_map(uint64_t scope_id, uint64_t aid, uint64_t bid, int32_t op,
                   int32_t dtype) {
  GScope *scope = g_lookup(scope_id, 2);
  if (!scope)
    return 0;
  GTensor *a = g_input(scope, aid), *b = bid ? g_input(scope, bid) : NULL;
  if (!a || (bid && !b))
    return 0;
  if (op < 0 || op > 9 || ((op >= 1 && op <= 4) != (b != NULL)))
    return g_fail(G_LAYOUT);
  if (dtype < 0)
    dtype = a->layout.dtype;
  if ((b && (b->layout.dtype != a->layout.dtype || dtype != a->layout.dtype)) ||
      (op != 0 && dtype != a->layout.dtype))
    return g_fail(G_DTYPE);
  size_t rank = a->layout.rank;
  if (b && b->layout.rank > rank)
    rank = b->layout.rank;
  size_t shape[G_RANK];
  for (size_t i = 0; i < rank; ++i) {
    size_t x = i < rank - a->layout.rank
                   ? 1
                   : a->layout.shape[i - (rank - a->layout.rank)];
    size_t y = !b || i < rank - b->layout.rank
                   ? 1
                   : b->layout.shape[i - (rank - b->layout.rank)];
    if (x != y && x != 1 && y != 1)
      return g_fail(G_SHAPE);
    shape[i] = x == 1 ? y : x;
  }
  GLayout l, al, bl = {0};
  if (!g_layout(&l, dtype, rank, shape) ||
      !g_broadcast(a->layout, rank, shape, &al) ||
      (b && !g_broadcast(b->layout, rank, shape, &bl)))
    return 0;
  uint64_t id = g_tensor(scope, l, NULL);
  if (!id)
    return 0;
  GTensor *out = g_lookup(id, 3);
  GSession *s = scope->session;
  if (s->device < 0) {
    GTask task = {out, a, b, al, bl, op, 0, 0};
    g_parallel(&s->pool, l.count, g_map_worker, &task);
  } else if (!g_cuda_ok(s, s->cuda->map(s->context, out->storage->data, l,
                                        a->storage->data, al,
                                        b ? b->storage->data : NULL, bl, op))) {
    g_discard(id);
    return 0;
  }
  return id;
}
/* Fixed partitions, independent of worker count/scheduling. Bounded stack
 * storage avoids an allocation failure path for partial results. Small inputs
 * keep the existing serial path and never wake sleeping workers. */
#define G_REDUCE_THRESHOLD 262144
#define G_REDUCE_PARTS 64
typedef struct {
  GTensor *input;
  int op;
  double partial[G_REDUCE_PARTS];
} GGlobalReduce;
static void g_global_reduce_worker(void *arg, size_t begin, size_t end) {
  GGlobalReduce *q = arg;
  GLayout l = q->input->layout;
  const void *data = q->input->storage->data;
  bool contiguous = g_contiguous(l);
  for (size_t part = begin; part < end; ++part) {
    /* Quotient/remainder form avoids count * part overflow. */
    size_t width = l.count / G_REDUCE_PARTS, rem = l.count % G_REDUCE_PARTS;
    size_t start = width * part + (part < rem ? part : rem);
    size_t stop = start + width + (part < rem);
    double total = q->op == 2 ? INFINITY : q->op == 3 ? -INFINITY : 0;
    for (size_t i = start; i < stop; ++i) {
      double v = g_read(data, l.dtype, contiguous ? l.offset + i : g_index(l, i));
      if (q->op < 2)
        total += v;
      else if (isnan(v) || isnan(total))
        total = NAN;
      else
        total = q->op == 2 ? fmin(total, v) : fmax(total, v);
    }
    q->partial[part] = total;
  }
}
static void g_global_reduce(GPool *pool, GTensor *out, GTensor *input, int op) {
  GGlobalReduce q = {input, op, {0}};
  g_parallel_jobs(pool, G_REDUCE_PARTS, g_global_reduce_worker, &q);
  double total = q.partial[0];
  for (size_t i = 1; i < G_REDUCE_PARTS; ++i) {
    double v = q.partial[i];
    if (op < 2)
      total += v;
    else if (isnan(v) || isnan(total))
      total = NAN;
    else
      total = op == 2 ? fmin(total, v) : fmax(total, v);
  }
  if (op == 1)
    total /= (double)input->layout.count;
  g_write(out->storage->data, out->layout.dtype, 0, total);
}
static void g_reduce_worker(void *arg, size_t begin, size_t end) {
  GTask *q = arg;
  GLayout al = q->al, ol = q->out->layout;
  bool contiguous = g_contiguous(al);
  size_t n = q->axis < 0 ? al.count : al.shape[q->axis];
  for (size_t i = begin; i < end; ++i) {
    size_t base = al.offset, index = i;
    if (q->axis >= 0)
      for (size_t k = al.rank; k-- > 0;)
        if ((int)k != q->axis) {
          base += (index % al.shape[k]) * al.stride[k];
          index /= al.shape[k];
        }
    double total = q->op == 2 ? INFINITY : q->op == 3 ? -INFINITY : 0;
    for (size_t j = 0; j < n; ++j) {
      double v =
          g_read(q->a->storage->data, al.dtype,
                 q->axis < 0 ? (contiguous ? al.offset + j : g_index(al, j))
                             : base + j * al.stride[q->axis]);
      if (q->op < 2)
        total += v;
      else if (isnan(v) || isnan(total))
        total = NAN;
      else
        total = q->op == 2 ? fmin(total, v) : fmax(total, v);
    }
    if (q->op == 1)
      total = n ? total / (double)n : NAN;
    g_write(q->out->storage->data, ol.dtype, i, total);
  }
}
uint64_t gauss_reduce(uint64_t scope_id, uint64_t aid, int32_t axis, int32_t op,
                      bool keep) {
  GScope *scope = g_lookup(scope_id, 2);
  if (!scope)
    return 0;
  GTensor *a = g_input(scope, aid);
  if (!a)
    return 0;
  if (axis < -1 || axis >= (int)a->layout.rank || op < 0 || op > 3)
    return g_fail(G_AXIS);
  size_t n = axis < 0 ? a->layout.count : a->layout.shape[axis];
  if (!n && op >= 2)
    return g_fail(G_SHAPE);
  size_t shape[G_RANK] = {0}, rank = 0;
  for (size_t i = 0; i < a->layout.rank; ++i) {
    if (axis < 0 || (int)i == axis) {
      if (keep)
        shape[rank++] = 1;
    } else
      shape[rank++] = a->layout.shape[i];
  }
  GLayout l;
  if (!g_layout(&l, a->layout.dtype, rank, shape))
    return 0;
  uint64_t id = g_tensor(scope, l, NULL);
  if (!id)
    return 0;
  GTensor *out = g_lookup(id, 3);
  GSession *s = scope->session;
  if (s->device < 0) {
    GTask q = {out, a, NULL, a->layout, {0}, op, axis, keep};
    if (axis < 0 && n >= G_REDUCE_THRESHOLD)
      g_global_reduce(&s->pool, out, a, op);
    else
      g_parallel(&s->pool, l.count, g_reduce_worker, &q);
  } else if (!g_cuda_ok(s, s->cuda->reduce(s->context, out->storage->data, l,
                                           a->storage->data, a->layout, axis,
                                           op, keep))) {
    g_discard(id);
    return 0;
  }
  return id;
}
/* BLAS accepts row/column-major strided matrices. Arbitrary inner strides must
 * be made contiguous explicitly; views in the batch dimensions need no copy. */
static bool g_matrix(GLayout l, int *trans, int *ld) {
  size_t r = l.rank - 2, c = l.rank - 1;
  if (l.stride[c] == 1 && l.stride[r] >= l.shape[c]) {
    *trans = 0;
    if (l.stride[r] > INT_MAX)
      return false;
    *ld = (int)l.stride[r];
    return true;
  }
  if (l.stride[r] == 1 && l.stride[c] >= l.shape[r]) {
    *trans = 1;
    if (l.stride[c] > INT_MAX)
      return false;
    *ld = (int)l.stride[c];
    return true;
  }
  return false;
}
uint64_t gauss_matmul(uint64_t scope_id, uint64_t aid, uint64_t bid) {
  GScope *scope = g_lookup(scope_id, 2);
  if (!scope)
    return 0;
  GTensor *a = g_input(scope, aid), *b = g_input(scope, bid);
  if (!a || !b)
    return 0;
  GLayout al = a->layout, bl = b->layout;
  if (al.dtype != bl.dtype)
    return g_fail(G_DTYPE);
  if (al.rank < 2 || bl.rank < 2 ||
      al.shape[al.rank - 1] != bl.shape[bl.rank - 2])
    return g_fail(G_SHAPE);
  size_t m = al.shape[al.rank - 2], k = al.shape[al.rank - 1],
         n = bl.shape[bl.rank - 1];
  if (m > INT_MAX || n > INT_MAX || k > INT_MAX)
    return g_fail(G_OVERFLOW);
  size_t rank = al.rank > bl.rank ? al.rank : bl.rank, shape[G_RANK];
  for (size_t i = 0; i < rank - 2; ++i) {
    size_t x = i < rank - al.rank ? 1 : al.shape[i - (rank - al.rank)],
           y = i < rank - bl.rank ? 1 : bl.shape[i - (rank - bl.rank)];
    if (x != y && x != 1 && y != 1)
      return g_fail(G_SHAPE);
    shape[i] = x == 1 ? y : x;
  }
  shape[rank - 2] = m;
  shape[rank - 1] = n;
  GLayout l;
  if (!g_layout(&l, al.dtype, rank, shape))
    return 0;
  if (!k)
    return gauss_full(scope_id, al.dtype, rank, shape, 0);
  int ta, tb, lda, ldb;
  if (!g_matrix(al, &ta, &lda) || !g_matrix(bl, &tb, &ldb))
    return g_fail(G_LAYOUT);
  uint64_t id = g_tensor(scope, l, NULL);
  if (!id)
    return 0;
  GTensor *out = g_lookup(id, 3);
  GSession *s = scope->session;
  size_t batches = (m && n) ? l.count / m / n : 0;
  for (size_t batch = 0; batch < batches; ++batch) {
    size_t index = batch, ao = al.offset, bo = bl.offset;
    for (size_t i = rank - 2; i-- > 0;) {
      size_t v = index % shape[i];
      index /= shape[i];
      if (i >= rank - al.rank) {
        size_t j = i - (rank - al.rank);
        if (al.shape[j] != 1)
          ao += v * al.stride[j];
      }
      if (i >= rank - bl.rank) {
        size_t j = i - (rank - bl.rank);
        if (bl.shape[j] != 1)
          bo += v * bl.stride[j];
      }
    }
    const char *ap = (const char *)a->storage->data + ao * g_item(al.dtype),
               *bp = (const char *)b->storage->data + bo * g_item(bl.dtype);
    char *cp = (char *)out->storage->data + batch * m * n * g_item(l.dtype);
    if (s->device >= 0) {
      if (!g_cuda_ok(s, s->cuda->gemm(s->context, cp, ap, bp, l.dtype, (int)m,
                                      (int)n, (int)k, ta, tb, lda, ldb))) {
        g_discard(id);
        return 0;
      }
    } else if (l.dtype)
      s->dgemm(101, ta ? 112 : 111, tb ? 112 : 111, (int)m, (int)n, (int)k, 1,
               (const double *)ap, lda, (const double *)bp, ldb, 0,
               (double *)cp, (int)n);
    else
      s->sgemm(101, ta ? 112 : 111, tb ? 112 : 111, (int)m, (int)n, (int)k, 1,
               (const float *)ap, lda, (const float *)bp, ldb, 0, (float *)cp,
               (int)n);
  }
  return id;
}
bool gauss_read(uint64_t id, int32_t dtype, void *data, size_t count) {
  GTensor *t = g_lookup(id, 3);
  if (!t)
    return false;
  if (t->scope->session->failed)
    return g_fail(G_BACKEND);
  if (dtype != t->layout.dtype) {
    g_fail(G_DTYPE);
    return false;
  }
  if (count != t->layout.count || (count && !data)) {
    g_fail(G_SHAPE);
    return false;
  }
  GSession *s = t->scope->session;
  if (s->device < 0) {
    if (g_contiguous(t->layout)) {
      if (count)
        memcpy(data,
               (const char *)t->storage->data +
                   t->layout.offset * g_item(dtype),
               count * g_item(dtype));
      return true;
    }
    for (size_t i = 0; i < count; ++i)
      g_write(data, dtype, i,
              g_read(t->storage->data, dtype, g_index(t->layout, i)));
    return true;
  }
  if (!count)
    return g_wait(s);
  if (g_contiguous(t->layout))
    return g_cuda_ok(s, s->cuda->download(s->context, data,
                                          (char *)t->storage->data +
                                              t->layout.offset * g_item(dtype),
                                          count * g_item(dtype)));
  uint64_t packed = gauss_map(t->scope->id, id, 0, 0, dtype);
  if (!packed)
    return false;
  bool ok = gauss_read(packed, dtype, data, count);
  g_discard(packed);
  return ok;
}
uint64_t gauss_to_device(uint64_t scope_id, uint64_t id) {
  GScope *scope = g_lookup(scope_id, 2);
  GTensor *t = g_lookup(id, 3);
  if (!scope || !t)
    return 0;
  GLayout l = t->layout;
  size_t bytes = l.count * g_item(l.dtype);
  void *host = bytes ? malloc(bytes) : NULL;
  if (bytes && !host)
    return g_fail(G_OOM);
  if (!gauss_read(id, l.dtype, host, l.count)) {
    free(host);
    return 0;
  }
  uint64_t result =
      gauss_from(scope_id, l.dtype, l.rank, l.shape, host, l.count);
  free(host);
  return result;
}
uint64_t gauss_from_f32(uint64_t s, size_t rank, const size_t *shape,
                        const float *p, size_t n) {
  return gauss_from(s, 0, rank, shape, p, n);
}
uint64_t gauss_from_f64(uint64_t s, size_t rank, const size_t *shape,
                        const double *p, size_t n) {
  return gauss_from(s, 1, rank, shape, p, n);
}
bool gauss_read_f32(uint64_t t, float *p, size_t n) {
  return gauss_read(t, 0, p, n);
}
bool gauss_read_f64(uint64_t t, double *p, size_t n) {
  return gauss_read(t, 1, p, n);
}
#endif
