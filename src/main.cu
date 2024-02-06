#include <stdio.h>

#include <cuda_runtime.h>

#include <cublas_v2.h>

#define M 8
#define N 4
#define K 2

void gemm(int m, int n, int k, double alpha, double *a, double *b, double beta,
          double *c) {
  for (int x = 0; x < m; ++x)
    for (int y = 0; y < n; ++y) {
      double value = beta * c[x + y * m];

      for (int z = 0; z < k; ++z)
        value += alpha * a[x + z * m] * b[z + y * k];

      c[x + y * m] = value;
    }
}

void print(double *a, int m, int n) {
  for (int x = 0; x < m; ++x) {
    for (int y = 0; y < n; ++y)
      printf("%f ", a[x + y * m]);

    printf("\n");
  }
}

int main(void) {
  double alpha = 5.0;
  double beta = 3.0;

  double a[M * K];
  double b[K * N];
  double c[M * N];
  double d[M * N];

  for (int i = 0; i < M * K; ++i)
    a[i] = i;

  for (int i = 0; i < K * N; ++i)
    b[i] = i;

  for (int i = 0; i < M * N; ++i) {
    c[i] = i;
    d[i] = i;
  }

  {
    gemm(M, N, K, alpha, a, b, beta, c);

    print(c, M, N);
  }

  printf("\n");

  {
    double *device_a;
    double *device_b;
    double *device_d;

    cudaMalloc((void **)&device_a, M * K * sizeof(double));
    cudaMalloc((void **)&device_b, K * N * sizeof(double));
    cudaMalloc((void **)&device_d, M * N * sizeof(double));

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSetMatrix(M, K, sizeof(double), a, M, device_a, M);
    cublasSetMatrix(K, N, sizeof(double), b, K, device_b, K);
    cublasSetMatrix(M, N, sizeof(double), d, M, device_d, M);

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, device_a, M,
                device_b, K, &beta, device_d, M);

    cublasGetMatrix(M, N, sizeof(double), device_d, M, d, M);

    cublasDestroy(handle);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_d);

    print(d, M, N);
  }

  return 0;
}