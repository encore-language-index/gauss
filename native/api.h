#ifndef GAUSS_API_H
#define GAUSS_API_H
#include <stddef.h>
#include <stdint.h>
#define G_RANK 32
#define G_PLUGIN_ABI 1
/* All descriptors are borrowed for the duration of a call. CUDA kernels copy
 * metadata by value; no Encore address may escape a native call. */
typedef struct {
  size_t rank, count, offset, shape[G_RANK], stride[G_RANK];
  int dtype;
} GLayout;
typedef struct {
  int abi;
  int (*open)(int device, void **context);
  int (*close)(void *context);
  int (*wait)(void *context);
  int (*allocate)(void *context, size_t bytes, void **ptr);
  int (*release)(void *context, void *ptr);
  int (*upload)(void *context, void *dst, const void *src, size_t bytes);
  int (*download)(void *context, void *dst, const void *src, size_t bytes);
  int (*fill)(void *context, void *dst, GLayout layout, double value);
  int (*map)(void *context, void *dst, GLayout out, const void *a, GLayout al,
             const void *b, GLayout bl, int op);
  int (*reduce)(void *context, void *dst, GLayout out, const void *a,
                GLayout al, int axis, int op, int keep);
  int (*gemm)(void *context, void *dst, const void *a, const void *b, int dtype,
              int m, int n, int k, int ta, int tb, int lda, int ldb);
} GCuda;
#endif
