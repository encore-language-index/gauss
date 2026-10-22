#ifndef GAUSS_POOL_H
#define GAUSS_POOL_H
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
/* A bounded, sleeping pool per session. The caller participates. GEMM is never
 * called from these workers: OpenBLAS owns its own parallel region. */
typedef void (*GWork)(void *, size_t, size_t);
typedef struct GPool GPool;
typedef struct {
  GPool *pool;
  size_t index;
} GWorker;
struct GPool {
  pthread_mutex_t mutex;
  pthread_cond_t ready, done;
  pthread_t threads[15];
  GWorker workers[15];
  size_t width, epoch, pending, count;
  int stop;
  GWork work;
  void *arg;
};
static void *g_worker(void *arg) {
  GWorker *w = arg;
  GPool *p = w->pool;
  size_t epoch = 0;
  pthread_mutex_lock(&p->mutex);
  for (;;) {
    while (!p->stop && epoch == p->epoch)
      pthread_cond_wait(&p->ready, &p->mutex);
    if (p->stop)
      break;
    epoch = p->epoch;
    size_t begin = p->count * w->index / p->width;
    size_t end = p->count * (w->index + 1) / p->width;
    pthread_mutex_unlock(&p->mutex);
    p->work(p->arg, begin, end);
    pthread_mutex_lock(&p->mutex);
    if (--p->pending == 0)
      pthread_cond_signal(&p->done);
  }
  pthread_mutex_unlock(&p->mutex);
  return NULL;
}
static void g_pool_init(GPool *p) {
  pthread_mutex_init(&p->mutex, NULL);
  pthread_cond_init(&p->ready, NULL);
  pthread_cond_init(&p->done, NULL);
  long cpus = sysconf(_SC_NPROCESSORS_ONLN);
  size_t width = cpus > 0 ? (size_t)cpus : 1;
  const char *setting = getenv("GAUSS_NUM_THREADS");
  if (setting) {
    char *end;
    unsigned long n = strtoul(setting, &end, 10);
    if (!*end && n)
      width = n;
  }
  if (width > 16)
    width = 16;
  p->width = 1;
  for (size_t i = 1; i < width; ++i) {
    p->workers[i - 1] = (GWorker){p, i};
    if (pthread_create(&p->threads[i - 1], NULL, g_worker, &p->workers[i - 1]))
      break;
    ++p->width;
  }
}
/* Dispatch coarse jobs whose cost is known by the caller. */
static void g_parallel_jobs(GPool *p, size_t count, GWork work, void *arg) {
  if (count < 2 || p->width == 1) {
    work(arg, 0, count);
    return;
  }
  pthread_mutex_lock(&p->mutex);
  p->count = count;
  p->work = work;
  p->arg = arg;
  p->pending = p->width - 1;
  ++p->epoch;
  pthread_cond_broadcast(&p->ready);
  pthread_mutex_unlock(&p->mutex);
  work(arg, 0, count / p->width);
  pthread_mutex_lock(&p->mutex);
  while (p->pending)
    pthread_cond_wait(&p->done, &p->mutex);
  pthread_mutex_unlock(&p->mutex);
}
static void g_parallel(GPool *p, size_t count, GWork work, void *arg) {
  if (count < 32768) {
    work(arg, 0, count);
    return;
  }
  g_parallel_jobs(p, count, work, arg);
}
static void g_pool_close(GPool *p) {
  pthread_mutex_lock(&p->mutex);
  p->stop = 1;
  pthread_cond_broadcast(&p->ready);
  pthread_mutex_unlock(&p->mutex);
  for (size_t i = 1; i < p->width; ++i)
    pthread_join(p->threads[i - 1], NULL);
  pthread_cond_destroy(&p->ready);
  pthread_cond_destroy(&p->done);
  pthread_mutex_destroy(&p->mutex);
}
#endif
