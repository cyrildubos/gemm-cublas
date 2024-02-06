#include <stdio.h>

#include <cblas.h>

// #include <cuda_runtime.h>

// #include <cublas_v2.h>

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
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, M,
                b, K, beta, d, M);

    print(d, M, N);
  }

  return 0;
}