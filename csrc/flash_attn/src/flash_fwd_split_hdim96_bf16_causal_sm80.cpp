// !!! This is a file automatically generated by hipify!!!
// #include <ATen/dtk_macros.h>
// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template_hip.h"

template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 96, true>(Flash_fwd_params &params, hipStream_t stream);
