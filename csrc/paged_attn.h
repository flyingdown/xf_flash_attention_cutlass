#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif
void fmha_fwd(
  void* q_ptr,         // batch_size x seqlen_q x num_heads x head_size
  void* k_ptr,         // batch_size x seqlen_k x num_heads_k x head_size
  void* v_ptr,         // batch_size x seqlen_k x num_heads_k x head_size
  void* o_ptr,             // batch_size x seqlen_q x num_heads x head_size
  void* alibi_slopes_ptr, // num_heads or batch_size x num_heads
  const int32_t seqlen_q,
  const int32_t seqlen_k,
  const int32_t batch_size,
  const int32_t num_heads,
  const int32_t num_heads_k,
  const int32_t head_size,
  const float p_dropout,
  hipStream_t stream,
  hipDeviceProp_t* dprops,
  const float softmax_scale,
  void * p_ptr,
  void * softmax_lse_ptr,
  int window_size_left,
  int window_size_right,
  const float softcap,
  const bool return_softmax, 
  bool is_fp16,
  int num_splits = 0);

void fmha_varlen_fwd(
  void* q_ptrs, // total_q x num_heads_q x head_size, total_q := \sum_{i=0}^{b} s_i
  void* k_ptrs, // total_kv x num_heads_kv x head_size, total_kv := \sum_{i=0}^{b} s_i
  void* v_ptrs, // total_kv x num_heads_kv x head_size, total_kv := \sum_{i=0}^{b} s_i
  void* o_ptrs, // total_q x num_heads_q x head_size, total_kv := \sum_{i=0}^{b} s_i
  void* cu_seqlens_q_ptrs,  // b+1
  void* cu_seqlens_k_ptrs, // b+1
  // const int total_q,
  const int32_t max_seqlen_q, 
  const int32_t max_seqlen_k, 
  const int32_t batch_size, 
  const int32_t num_heads, 
  const int32_t num_heads_k, 
  const int32_t head_size, 
  hipStream_t stream,
  const float softmax_scale, 
  // void * softmax_lse_ptr,
  const bool is_causal,
  const bool is_fp16,
  int window_size_left = -1,
  int window_size_right = -1);

void fmha_page_kvcache_fwd(
  //Flash_fwd_params &params,
  void* q_ptr,
  void* kcache_ptr,
  void* vcache_ptr,
  void* k_ptr,
  void* v_ptr,
  void* o_ptr,
  void* block_table_ptr,
  void* cache_seqlens_k_ptr,
  const int32_t max_cache_seq_k,
  const int32_t seqlen_q,
  const int32_t seqlen_k,
  const int32_t batch_size,
  const int32_t num_heads,
  const int32_t num_heads_k,
  const int32_t head_size,
  const int32_t page_block_size,
  hipStream_t stream,
  const float softmax_scale,
  // void* softmax_lse,
  int window_size_left,
  int window_size_right,
  const int32_t num_splits,
  void* cache_batch_idx_ptr,
  void* rotary_cos_ptr,
  void* rotary_sin_ptr,
  bool is_causal,
  bool is_rotary_interleaved, bool is_fp16
);
#ifdef __cplusplus
} // 结束 extern "C"
#endif
