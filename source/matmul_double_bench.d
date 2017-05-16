module cuda_d_examples.matmul_double_bench;

import std.stdio;
import std.datetime;

import cuda_d.cuda;
import cuda_d.cublas;
import cuda_d.cublas_api;
import cuda_d.cublas_v2;
import cuda_d.cuda_runtime_api;
import cuda_d.curand;

void gpu_blas_mmul(const double *A, const double *B, double *C, const int m, const int k, const int n) {
  int lda=m,ldb=k,ldc=m;
  const double alf = 1;
  const double bet = 0;
  const double *alpha = &alf;
  const double *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Do the actual multiplication
  auto start = Clock.currTime();
  cublasDgemm(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  auto end = Clock.currTime();
  writeln("Time taken for gemm is : ", end - start , " .");

  // Destroy the handle
  cublasDestroy(handle);
}

void print_matrix(const double *A, int nr_rows_A, int nr_cols_A) {

  for(int i = 0; i < nr_rows_A; ++i){
    for(int j = 0; j < nr_cols_A; ++j){
      writeln(A[j * nr_rows_A + i]);
    }
    writeln();
  }
  writeln();
}

void call_gemm_routine(int side){
  // Allocate 3 arrays on CPU
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
  // for simplicity we are going to use square arrays
  nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = side;

  auto h_A = new double[nr_rows_A * nr_cols_A];
  auto h_B = new double[nr_rows_B * nr_cols_B];
  auto h_C = new double[nr_rows_C * nr_cols_C];

  // Allocate 3 arrays on GPU
  double* d_A, d_B, d_C;
  auto start = Clock.currTime();

  cudaMalloc(cast(void **)&d_A,nr_rows_A * nr_cols_A * cast(int)double.sizeof);
  cudaMalloc(cast(void **)&d_B,nr_rows_B * nr_cols_B * cast(int)double.sizeof);
  cudaMalloc(cast(void **)&d_C,nr_rows_C * nr_cols_C * cast(int)double.sizeof);

  auto end = Clock.currTime();
  writeln("Time taken for malloc is : ", end - start , " .");

  // Fill the arrays A and B on GPU with random numbers
  double[] random_A = new double[side * side];
  double[] random_B = new double[side * side];

  for(int i = 0; i < side*side; i++){
    random_A[i] = 5;
    random_B[i] = 2;
  }

  // Optionally we can copy the data back on CPU and print the arrays

  start = Clock.currTime();

  cudaMemcpy(cast(void*)d_A, cast(void*)random_A, nr_rows_A * nr_cols_A * cast(int)double.sizeof, cudaMemcpyKind.cudaMemcpyHostToDevice);
  //cudaMemcpy(h_A.ptr,d_A,nr_rows_A * nr_cols_A * cast(int)double.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);
  cudaMemcpy(cast(void*)d_B, cast(void*)random_B, nr_rows_B * nr_cols_B * cast(int)double.sizeof, cudaMemcpyKind.cudaMemcpyHostToDevice);
  //cudaMemcpy(h_B.ptr,d_A,nr_rows_A * nr_cols_A * cast(int)double.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);

  end = Clock.currTime();
  writeln("Time taken for copying data from device to host is : ", end - start , " .");
  //writeln( "h_A =");
  //print_matrix(h_A.ptr, nr_rows_A, nr_cols_A);

  //writeln( "B =");
  //print_matrix(h_B.ptr, nr_rows_B, nr_cols_B);

  // Multiply A and B on GPU

  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

  // Copy (and print) the result on host memory
  start = Clock.currTime();
  cudaMemcpy(h_C.ptr,d_C,nr_rows_C * nr_cols_C * cast(int)double.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);
  end = Clock.currTime();
  writeln("Time taken to copy data from device to host : ", end - start , " .");

  writeln( "C[0] =", h_C[0] );
  //print_matrix(h_C.ptr, nr_rows_C, nr_cols_C);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  //free(h_A);
  //free(h_B);
  //free(h_C);
}

void call_gemm_routine_benchmark(){
  int[] sides = [2, 5, 10 , 100 , 500, 1000, 2000, 5000, 10000];
  foreach(side; sides){
    writeln("side = ", side , "; elements = ", side * side );
    call_gemm_routine(side);
    writeln("------------------------------------");
  }
}