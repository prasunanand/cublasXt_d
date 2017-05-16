import std.stdio;

import cuda_d_examples.matmul_double_bench;
import cuda_d_examples.modify;

void main(){
  //call_modify();
  call_gemm_routine_benchmark();
}
