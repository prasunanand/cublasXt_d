module cuda_d_examples.matmul;

import std.stdio;

import cuda_d.cuda;
import cuda_d.cublas;
import cuda_d.cublas_api;
import cuda_d.cublas_v2;
import cuda_d.cuda_runtime_api;
import cuda_d.curand;

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
  // Create a pseudo-random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed for the random number generator using the system clock
  curandSetPseudoRandomGeneratorSeed(prng, 4);

  // Fill the array with random numbers on the device
  curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
  int lda=m,ldb=k,ldc=m;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Do the actual multiplication
  cublasSgemm(handle, cublasOperation_t.CUBLAS_OP_N, cublasOperation_t.CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  // Destroy the handle
  cublasDestroy(handle);
}

void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

  for(int i = 0; i < nr_rows_A; ++i){
    for(int j = 0; j < nr_cols_A; ++j){
      writeln(A[j * nr_rows_A + i]);
    }
    writeln();
  }
  writeln();
}

void call_gemm_routine(){
  // Allocate 3 arrays on CPU
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

  // for simplicity we are going to use square arrays
  nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 3;

  auto h_A = new float[nr_rows_A * nr_cols_A];
  auto h_B = new float[nr_rows_B * nr_cols_B];
  auto h_C = new float[nr_rows_C * nr_cols_C];

  // Allocate 3 arrays on GPU
  float* d_A, d_B, d_C;
  cudaMalloc(cast(void **)&d_A,nr_rows_A * nr_cols_A * cast(int)float.sizeof);
  cudaMalloc(cast(void **)&d_B,nr_rows_B * nr_cols_B * cast(int)float.sizeof);
  cudaMalloc(cast(void **)&d_C,nr_rows_C * nr_cols_C * cast(int)float.sizeof);

  // Fill the arrays A and B on GPU with random numbers
  GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
  GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

  // Optionally we can copy the data back on CPU and print the arrays
  cudaMemcpy(h_A.ptr,d_A,nr_rows_A * nr_cols_A * cast(int)float.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);

  cudaMemcpy(h_B.ptr,d_B,nr_rows_B * nr_cols_B * cast(int)float.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);
  writeln( "A =");
  print_matrix(h_A.ptr, nr_rows_A, nr_cols_A);
  writeln( "B =");
  print_matrix(h_B.ptr, nr_rows_B, nr_cols_B);

  // Multiply A and B on GPU
  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

  // Copy (and print) the result on host memory
  cudaMemcpy(h_C.ptr,d_C,nr_rows_C * nr_cols_C * cast(int)float.sizeof, cudaMemcpyKind.cudaMemcpyDeviceToHost);
  writeln( "C =" );
  print_matrix(h_C.ptr, nr_rows_C, nr_cols_C);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // Free CPU memory
  //free(h_A);
  //free(h_B);
  //free(h_C);
}