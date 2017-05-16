module cuda_d_examples.modify;

import std.stdio;

import cuda_d.cuda;
import cuda_d.cublas;
import cuda_d.cublas_api;
import cuda_d.cublas_v2;
import cuda_d.cuda_runtime_api;



int IDX2F(int i, int j, int ld) {
  return (j-1)*ld+(i-1);
}


void modify_routine (cublasHandle_t handle, float *m, int ldm, int
 n, int p, int q, float alpha, float beta){
 cublasSscal (handle, n-p+1, &alpha, &m[IDX2F(p,q,ldm)], ldm);
 cublasSscal (handle, ldm-p+1, &beta, &m[IDX2F(p,q,ldm)], 1);
}

void call_modify()
{
  writeln("Edit source/app.d to start your project.");
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  int M = 6;
  int N = 5;
  float* devPtrA;
  auto a = new double[M * N ];
  if (!a) {
  writeln ("host memory allocation failed");
  writeln("EXIT_FAILURE");
  }
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= M; i++) {
      a[IDX2F(i,j,M)] = (i-1) * M + j;
    }
  }
  cudaStat = cudaMalloc (cast(void**)&devPtrA, M*N*a.sizeof);
  writeln(cudaStat);
  //if (cudaStat != cudaSuccess) {
  //writeln("device memory allocation failed");
  //writeln("EXIT_FAILURE");
  //}
  stat = cublasCreate(&handle);
  //if (stat != CUBLAS_STATUS_SUCCESS) {
  //writeln("CUBLAS initialization failed\n");
  //writeln("EXIT_FAILURE");
  //}
  stat = cublasSetMatrix (M, N, cast(int)a.sizeof, cast(void*)a, M, devPtrA, M);
  //if (stat != CUBLAS_STATUS_SUCCESS) {
  writeln(stat);
  cudaFree(devPtrA);
  cublasDestroy(handle);
  //writeln("EXIT_FAILURE");
  //}
  modify_routine(handle, devPtrA, M, N, 2, 3, 16.0f, 12.0f);
  stat = cublasGetMatrix (M, N, cast(int)a.sizeof, cast(void*)devPtrA, M, cast(void*)a, M);
  //if (stat != CUBLAS_STATUS_SUCCESS) {
  writeln(stat);
  cudaFree (devPtrA);
  //cublasDestroy(handle);
  //writeln("EXIT_FAILURE");
  //}
  //cudaFree (devPtrA);
  //cublasDestroy(handle);
  for (j = 1; j <= N; j++) {
    for (i = 1; i <= M; i++) {
      writeln(a[IDX2F(i,j,M)]);
    }
    writeln("\n");
  }
  //free(a);
  //return EXIT_SUCCESS;
}
