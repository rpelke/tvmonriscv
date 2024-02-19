// tvm target: c -keys=cpu

#define TVM_EXPORTS

#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <iostream>
#include <dlpack/dlpack.h>

#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t test_fadd_pipeline(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle);
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t test_fadd_pipeline(void* args, int32_t* arg_type_ids, int32_t num_args, void* out_ret_value, int32_t* out_ret_tcode, void* resource_handle) {
  int32_t A_code = arg_type_ids[0];
  int32_t B_code = arg_type_ids[1];
  int32_t C_code = arg_type_ids[2];

  void* A = (((TVMValue*)args)[0].v_handle);
  void* B = (((TVMValue*)args)[1].v_handle);
  void* C = (((TVMValue*)args)[2].v_handle);

  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "A: " << A << std::endl;
  std::cout << "((DLTensor*)(A))[0].shape: " << ((DLTensor*)(A))[0].shape << std::endl;
  std::cout << "((DLTensor*)(A))[0].data: " << ((DLTensor*)(A))[0].data << std::endl;

  /*
  DLTensor in DLPack.h
  */
  void* test_fadd_pipeline_A_shape = (((DLTensor*)A)[0].shape);
  void* test_fadd_pipeline_A_strides = (((DLTensor*)A)[0].strides);
  int32_t dev_id = (((DLTensor*)A)[0].device.device_id);
  void* A_1 = (((DLTensor*)A)[0].data);

  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "A_1: " << A_1 << std::endl;
  std::cout << "(float*)A_1: " << (float*)A_1 << std::endl;
  std::cout << "((float*)A_1)[0]: " << ((float*)A_1)[0] << std::endl;

  void* test_fadd_pipeline_B_shape = (((DLTensor*)B)[0].shape);
  void* test_fadd_pipeline_B_strides = (((DLTensor*)B)[0].strides);
  void* B_1 = (((DLTensor*)B)[0].data);
  void* test_fadd_pipeline_C_shape = (((DLTensor*)C)[0].shape);
  void* test_fadd_pipeline_C_strides = (((DLTensor*)C)[0].strides);
  void* C_1 = (((DLTensor*)C)[0].data);
  if (!(test_fadd_pipeline_A_strides == NULL)) {
  }
  if (!(test_fadd_pipeline_B_strides == NULL)) {
  }
  if (!(test_fadd_pipeline_C_strides == NULL)) {
  }
  void* A_2 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4096, 2, 32);
  if (A_2 == NULL) {
    return -1;
  }
  void* B_2 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)4096, 2, 32);
  if (B_2 == NULL) {
    return -1;
  }
  for (int32_t i0 = 0; i0 < 1024; ++i0) {
    ((float*)A_2)[i0] = ((float*)A_1)[i0];
  }
  for (int32_t i0_1 = 0; i0_1 < 1024; ++i0_1) {
    ((float*)B_2)[i0_1] = ((float*)B_1)[i0_1];
  }
  for (int32_t i0_2 = 0; i0_2 < 1024; ++i0_2) {
    ((float*)A_2)[i0_2] = (((float*)A_2)[i0_2] + ((float*)B_2)[i0_2]);
  }
  for (int32_t i0_outer_outer = 0; i0_outer_outer < 20; ++i0_outer_outer) {
    for (int32_t i0_outer_inner = 0; i0_outer_inner < 13; ++i0_outer_inner) {
      for (int32_t i0_inner = 0; i0_inner < 4; ++i0_inner) {
        if (((i0_outer_outer * 13) + i0_outer_inner) < 256) {
          int32_t cse_var_1 = (((i0_outer_outer * 52) + (i0_outer_inner * 4)) + i0_inner);
          ((float*)C_1)[cse_var_1] = ((float*)A_2)[cse_var_1];
        }
      }
    }
  }
  if (TVMBackendFreeWorkspace(1, dev_id, B_2) != 0) {
    return -1;
  }
  if (TVMBackendFreeWorkspace(1, dev_id, A_2) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t __tvm_main__(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
  return test_fadd_pipeline(args, arg_type_ids, num_args, out_ret_value, out_ret_tcode, resource_handle);
}


int main(void) {
  int64_t n = 1024;

  volatile float A[n];
  for (int i = 0; i < n; ++i) {
      A[i] = 2.5f;
  }

  volatile float B[n];
  for (int i = 0; i < n; ++i) {
      B[i] = 1.0f;
  }

  volatile float C[n];
  for (int i = 0; i < n; ++i) {
      C[i] = 0.0f;
  }

  DLDevice dev{kDLCPU, 0};
  DLDataType dtype{2, 32, 1};

  int32_t arg_type_ids[] = {0, 1, 2};

  int32_t ndim = 1;
  int64_t shape[1] = {n};

  DLTensor tensor_A;
  tensor_A.data = (void*)A;
  tensor_A.device = dev;
  tensor_A.ndim = ndim;
  tensor_A.dtype = dtype;
  tensor_A.shape = shape;
  tensor_A.strides = NULL;
  tensor_A.byte_offset = 0;

  DLTensor tensor_B;
  tensor_B.data = (void*)B;
  tensor_B.device = dev;
  tensor_B.ndim = ndim;
  tensor_B.dtype = dtype;
  tensor_B.shape = shape;
  tensor_B.strides = NULL;
  tensor_B.byte_offset = 0;

  DLTensor tensor_C;
  tensor_C.data = (void*)C;
  tensor_C.device = dev;
  tensor_C.ndim = ndim;
  tensor_C.dtype = dtype;
  tensor_C.shape = shape;
  tensor_C.strides = NULL;
  tensor_C.byte_offset = 0;

  void *args[] = {&tensor_A, &tensor_B, &tensor_C};

  __tvm_main__(args, arg_type_ids, 0, nullptr, nullptr, nullptr);

  for (int i=0; i<shape[0]; i++)
  {
    std::cout << "C[" << i << "]: " << C[i] << std::endl;
  }

  return 0;
}
