#include "api.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
struct Upload {
  void *ptr;
  Upload *next;
};
struct Context {
  int device;
  cudaStream_t stream;
  cublasHandle_t blas;
  Upload *uploads;
};
static int select(Context *c) { return (int)cudaSetDevice(c->device); }
static int wait_context(void *ptr) {
  Context *c = (Context *)ptr;
  int rc = select(c);
  if (rc)
    return rc;
  rc = (int)cudaStreamSynchronize(c->stream);
  while (c->uploads) {
    Upload *u = c->uploads;
    c->uploads = u->next;
    cudaFreeHost(u->ptr);
    free(u);
  }
  return rc;
}
static int open_context(int device, void **out) {
  *out = nullptr;
  int rc = (int)cudaSetDevice(device);
  if (rc)
    return rc;
  Context *c = (Context *)calloc(1, sizeof(Context));
  if (!c)
    return -1;
  c->device = device;
  rc = (int)cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking);
  if (!rc)
    rc = (int)cublasCreate(&c->blas);
  if (!rc)
    rc = (int)cublasSetStream(c->blas, c->stream);
  if (!rc)
    rc = (int)cublasSetMathMode(c->blas, CUBLAS_PEDANTIC_MATH);
  if (rc) {
    if (c->blas)
      cublasDestroy(c->blas);
    if (c->stream)
      cudaStreamDestroy(c->stream);
    free(c);
    return rc;
  }
  *out = c;
  return 0;
}
static int close_context(void *ptr) {
  Context *c = (Context *)ptr;
  int rc = wait_context(c);
  int b = (int)cublasDestroy(c->blas), s = (int)cudaStreamDestroy(c->stream);
  free(c);
  return rc ? rc : b ? b : s;
}
static int allocate(void *ptr, size_t bytes, void **out) {
  Context *c = (Context *)ptr;
  *out = nullptr;
  int rc = select(c);
  if (rc || !bytes)
    return rc;
  return (int)cudaMallocAsync(out, bytes, c->stream);
}
static int release(void *ptr, void *data) {
  Context *c = (Context *)ptr;
  int rc = select(c);
  if (rc || !data)
    return rc;
  return (int)cudaFreeAsync(data, c->stream);
}
static int upload(void *ptr, void *dst, const void *src, size_t bytes) {
  Context *c = (Context *)ptr;
  int rc = select(c);
  if (rc || !bytes)
    return rc;
  Upload *u = (Upload *)calloc(1, sizeof(Upload));
  if (!u)
    return -1;
  rc = (int)cudaMallocHost(&u->ptr, bytes);
  if (rc) {
    free(u);
    return rc;
  }
  memcpy(u->ptr, src, bytes);
  rc = (int)cudaMemcpyAsync(dst, u->ptr, bytes, cudaMemcpyHostToDevice,
                            c->stream);
  if (rc) {
    cudaFreeHost(u->ptr);
    free(u);
    return rc;
  }
  u->next = c->uploads;
  c->uploads = u;
  return 0;
}
static int download(void *ptr, void *dst, const void *src, size_t bytes) {
  Context *c = (Context *)ptr;
  int rc = select(c);
  if (rc)
    return rc;
  if (bytes) {
    rc = (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                              c->stream);
    if (rc)
      return rc;
  }
  return wait_context(c);
}
__device__ static size_t index_of(GLayout l, size_t i) {
  size_t offset = l.offset;
  for (size_t k = l.rank; k-- > 0;) {
    offset += (i % l.shape[k]) * l.stride[k];
    i /= l.shape[k];
  }
  return offset;
}
static bool contiguous(GLayout l) {
  size_t stride = 1;
  for (size_t k = l.rank; k-- > 0;) {
    if (l.shape[k] > 1 && l.stride[k] != stride)
      return false;
    stride *= l.shape[k];
  }
  return true;
}
__device__ static double read_value(const void *data, int dtype, size_t i) {
  return dtype ? ((const double *)data)[i] : ((const float *)data)[i];
}
__device__ static void write_value(void *data, int dtype, size_t i, double x) {
  if (dtype)
    ((double *)data)[i] = x;
  else
    ((float *)data)[i] = (float)x;
}
__global__ static void fill_kernel(void *dst, GLayout l, double value) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < l.count;
       i += (size_t)blockDim.x * gridDim.x)
    write_value(dst, l.dtype, i, value);
}
__global__ static void map_kernel(void *dst, GLayout out, const void *a,
                                  GLayout al, const void *b, GLayout bl, int op,
                                  bool ac, bool bc) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < out.count;
       i += (size_t)blockDim.x * gridDim.x) {
    double x = read_value(a, al.dtype, ac ? al.offset + i : index_of(al, i)),
           y = b ? read_value(b, bl.dtype, bc ? bl.offset + i : index_of(bl, i))
                 : 0,
           v = x;
    switch (op) {
    case 1:
      v = x + y;
      break;
    case 2:
      v = x - y;
      break;
    case 3:
      v = x * y;
      break;
    case 4:
      v = x / y;
      break;
    case 5:
      v = -x;
      break;
    case 6:
      v = fabs(x);
      break;
    case 7:
      v = exp(x);
      break;
    case 8:
      v = log(x);
      break;
    case 9:
      v = sqrt(x);
      break;
    }
    write_value(dst, out.dtype, i, v);
  }
}
/* One cooperative block per output; grid-stride over outputs. No atomic sums,
 * so repeated executions use the same reduction order on a given device. */
__global__ static void reduce_kernel(void *dst, GLayout out, const void *a,
                                     GLayout al, int axis, int op, bool ac,
                                     size_t mean_count) {
  __shared__ double values[256];
  size_t n = axis < 0 ? al.count : al.shape[axis];
  for (size_t i = blockIdx.x; i < out.count; i += gridDim.x) {
    size_t base = al.offset, index = i;
    if (axis >= 0)
      for (size_t k = al.rank; k-- > 0;)
        if ((int)k != axis) {
          base += (index % al.shape[k]) * al.stride[k];
          index /= al.shape[k];
        }
    double value = op == 2 ? INFINITY : op == 3 ? -INFINITY : 0;
    for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
      double x = read_value(a, al.dtype,
                            axis < 0 ? (ac ? al.offset + j : index_of(al, j))
                                     : base + j * al.stride[axis]);
      if (op < 2)
        value += x;
      else if (isnan(x) || isnan(value))
        value = NAN;
      else
        value = op == 2 ? fmin(value, x) : fmax(value, x);
    }
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
      if (threadIdx.x < (unsigned)stride) {
        double x = values[threadIdx.x + stride], v = values[threadIdx.x];
        values[threadIdx.x] = op < 2                   ? v + x
                              : (isnan(x) || isnan(v)) ? NAN
                              : op == 2                ? fmin(v, x)
                                                       : fmax(v, x);
      }
      __syncthreads();
    }
    if (!threadIdx.x)
      write_value(dst, out.dtype, i,
                  op == 1 ? (mean_count ? values[0] / (double)mean_count : NAN)
                          : values[0]);
    __syncthreads();
  }
}
/* Split a whole-tensor reduction across SMs. Partial sums stay f64 even when
 * the input/output are f32; the final kernel performs the sole dtype cast.
 * No atomics: scheduling cannot change the arithmetic order. */
__global__ static void partial_reduce(double *out, const void *a, GLayout al,
                                      int op, bool ac) {
  __shared__ double values[256];
  double value = op == 2 ? INFINITY : op == 3 ? -INFINITY : 0;
  for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < al.count;
       j += (size_t)gridDim.x * blockDim.x) {
    double x = read_value(a, al.dtype, ac ? al.offset + j : index_of(al, j));
    value = op < 2                       ? value + x
            : (isnan(x) || isnan(value)) ? NAN
            : op == 2                    ? fmin(value, x)
                                         : fmax(value, x);
  }
  values[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      double x = values[threadIdx.x + stride], v = values[threadIdx.x];
      values[threadIdx.x] = op < 2                   ? v + x
                            : (isnan(x) || isnan(v)) ? NAN
                            : op == 2                ? fmin(v, x)
                                                     : fmax(v, x);
    }
    __syncthreads();
  }
  if (!threadIdx.x)
    out[blockIdx.x] = values[0];
}
static unsigned blocks(size_t count) {
  size_t n = count / 256 + (count % 256 != 0);
  return (unsigned)(n > 65535 ? 65535 : n);
}
static int fill(void *p, void *dst, GLayout l, double value) {
  Context *c = (Context *)p;
  int rc = select(c);
  if (rc || !l.count)
    return rc;
  fill_kernel<<<blocks(l.count), 256, 0, c->stream>>>(dst, l, value);
  return (int)cudaGetLastError();
}
static int map(void *p, void *dst, GLayout l, const void *a, GLayout al,
               const void *b, GLayout bl, int op) {
  Context *c = (Context *)p;
  int rc = select(c);
  if (rc || !l.count)
    return rc;
  map_kernel<<<blocks(l.count), 256, 0, c->stream>>>(
      dst, l, a, al, b, bl, op, contiguous(al), contiguous(bl));
  return (int)cudaGetLastError();
}
static int reduce(void *p, void *dst, GLayout l, const void *a, GLayout al,
                  int axis, int op, int) {
  Context *c = (Context *)p;
  int rc = select(c);
  if (rc || !l.count)
    return rc;
  if (axis < 0 && al.count >= 4096) {
    unsigned grid = blocks(al.count);
    if (grid > 1024)
      grid = 1024;
    double *partial = nullptr;
    rc = (int)cudaMallocAsync(&partial, grid * sizeof(double), c->stream);
    if (rc)
      return rc;
    partial_reduce<<<grid, 256, 0, c->stream>>>(partial, a, al, op,
                                                contiguous(al));
    rc = (int)cudaGetLastError();
    GLayout pl = {};
    pl.rank = 1;
    pl.count = grid;
    pl.shape[0] = grid;
    pl.stride[0] = 1;
    pl.dtype = 1;
    if (!rc) {
      reduce_kernel<<<1, 256, 0, c->stream>>>(dst, l, partial, pl, -1, op, true,
                                              al.count);
      rc = (int)cudaGetLastError();
    }
    int freed = (int)cudaFreeAsync(partial, c->stream);
    return rc ? rc : freed;
  }
  unsigned grid = (unsigned)(l.count > 65535 ? 65535 : l.count);
  reduce_kernel<<<grid, 256, 0, c->stream>>>(
      dst, l, a, al, axis, op, contiguous(al),
      axis < 0 ? al.count : al.shape[axis]);
  return (int)cudaGetLastError();
}
static int gemm(void *p, void *dst, const void *a, const void *b, int dtype,
                int m, int n, int k, int ta, int tb, int lda, int ldb) {
  Context *c = (Context *)p;
  int rc = select(c);
  if (rc)
    return rc;
  /* Row-major C=A B is column-major C^T=B^T A^T. No transpose copy. */
  cublasOperation_t at = ta ? CUBLAS_OP_T : CUBLAS_OP_N,
                    bt = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (dtype) {
    double one = 1, zero = 0;
    return (int)cublasDgemm(c->blas, bt, at, n, m, k, &one, (const double *)b,
                            ldb, (const double *)a, lda, &zero, (double *)dst,
                            n);
  }
  float one = 1, zero = 0;
  return (int)cublasGemmEx(c->blas, bt, at, n, m, k, &one, b, CUDA_R_32F, ldb,
                           a, CUDA_R_32F, lda, &zero, dst, CUDA_R_32F, n,
                           CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
}
extern "C" const GCuda *gauss_cuda_api(void) {
  static const GCuda api = {G_PLUGIN_ABI, open_context, close_context,
                            wait_context, allocate,     release,
                            upload,       download,     fill,
                            map,          reduce,       gemm};
  return &api;
}
