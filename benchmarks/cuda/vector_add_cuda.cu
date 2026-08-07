#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void check(cudaError_t e, const char *expr, const char *file, int line) {
  if (e == cudaSuccess)
    return;
  std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
               file, line, expr, cudaGetErrorString(e));
  std::abort();
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  long long n) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

static long long parse_i64(int argc, char **argv, int index,
                           long long fallback) {
  if (argc <= index)
    return fallback;
  return std::atoll(argv[index]);
}

static int parse_i32(int argc, char **argv, int index, int fallback) {
  if (argc <= index)
    return fallback;
  return std::atoi(argv[index]);
}

int main(int argc, char **argv) {
  long long n = parse_i64(argc, argv, 1, 1048576);
  int reps = parse_i32(argc, argv, 2, 100);

  std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  float *a = nullptr;
  float *b = nullptr;
  float *c = nullptr;

  CHECK(cudaMallocHost(&a, bytes));
  CHECK(cudaMallocHost(&b, bytes));
  CHECK(cudaMallocHost(&c, bytes));

  for (long long i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i + 1);
    b[i] = 2.0f * static_cast<float>(i + 1);
    c[i] = 0.0f;
  }

  float *da = nullptr;
  float *db = nullptr;
  float *dc = nullptr;

  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));

  CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);

  vector_add_kernel<<<grid, block>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start));
  for (int r = 0; r < reps; ++r)
    vector_add_kernel<<<grid, block>>>(da, db, dc, n);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));

  int errors = 0;
  for (long long i = 0; i < n; ++i) {
    float expected = a[i] + b[i];
    if (std::fabs(c[i] - expected) > 1.0e-5f)
      ++errors;
  }

  double avgSeconds = (static_cast<double>(ms) / 1.0e3) / reps;
  double bytesPerRep = 3.0 * static_cast<double>(n) * sizeof(float);
  double gbps = bytesPerRep / avgSeconds / 1.0e9;

  std::printf("cuda_vector_add,%lld,1,%d,%.8e,%.8e,%d\n",
              n, reps, avgSeconds, gbps, errors);

  CHECK(cudaFree(da));
  CHECK(cudaFree(db));
  CHECK(cudaFree(dc));
  CHECK(cudaFreeHost(a));
  CHECK(cudaFreeHost(b));
  CHECK(cudaFreeHost(c));

  return errors == 0 ? 0 : 1;
}

