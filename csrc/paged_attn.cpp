#include "flash_hip.h"
#include "static_switch.h"
#include <cutlass/numeric_types.h>
#include <iostream>

void set_params_fprop_strided(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      void * const q,
                      void * const k,
                      void * const v,
                      void * const out,
                      void * const cu_seqlens_q_d,
                      void * const cu_seqlens_k_d,
                      void *seqused_k,
                      void * const p_d,
                      void * const softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool is_bf16,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false

) {
    // Reset the parameters
    params = {};

    params.is_bf16 = is_bf16;
    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;
    // All stride are in elements, not bytes.
    params.q_row_stride = h * d;
    params.k_row_stride = h_k * d;
    params.v_row_stride = h_k * d;
    params.q_head_stride = d;
    params.k_head_stride = d;
    params.v_head_stride = d;
    params.o_ptr = out;
    params.o_row_stride = h * d;
    params.o_head_stride = d;

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = seqlen_q * h * d;
        params.k_batch_stride = seqlen_k * h_k * d;
        params.v_batch_stride = seqlen_k * h_k * d;
        params.o_batch_stride = seqlen_q * h * d;
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;


    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    // ASSERT_CHECK(p_dropout < 1.f);

    params.is_causal = window_size_left < 0 && window_size_right == 0;
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;


}

inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // printf("num_SMs = %d batch_nheads_mblocks = %d\n", num_SMs, batch_nheads_mblocks);
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

void set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, hipDeviceProp_t *dprops, void * softmax_lse_accum_ptr, void * out_accum_ptr) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 64 : (head_size % 64 == 0 ? 32 : 64);

    // const int block_n = head_size <= 64 ? 128 : (head_size <= 128 ? 64 : 32);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // printf("max_seqlen_k = %d num_n_blocks = %d\n", max_seqlen_k, num_n_blocks);
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            HIP_CHECK(hipMalloc(&softmax_lse_accum_ptr, params.num_splits * batch_size * max_seqlen_q * num_heads * sizeof(float)));
            HIP_CHECK(hipMalloc(&out_accum_ptr, params.num_splits * batch_size * max_seqlen_q * num_heads * head_size_rounded * sizeof(float)));
            params.softmax_lseaccum_ptr = softmax_lse_accum_ptr;
            params.oaccum_ptr = out_accum_ptr;
        }
        // TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
        ASSERT_CHECK(params.num_splits <= 128);
    }
    // params.num_splits = 1;
    // printf("!!!params.num_splits = %d\n", params.num_splits);
}

// void run_mha_fwd(Flash_fwd_params &params, hipStream_t stream) {
//     FP16_SWITCH(!params.is_bf16, [&] {
//         HEADDIM_SWITCH(params.d, [&] {
//             BOOL_SWITCH(params.is_causal, Is_causal, [&] {
//                 // printf("params.num_splits = %d !force_split_kernel = %d\n", params.num_splits, !force_split_kernel);
//                 run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
//             });
//         });
//     });
// }

void run_mha_fwd__(Flash_fwd_params &params, hipStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                // printf("params.num_splits = %d !force_split_kernel = %d\n", params.num_splits, !force_split_kernel);
                // if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                //     run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                // } else { 
                    printf("!!!!!split!!!!  params.num_splits = %d\n",  params.num_splits);
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                // }
            });
        });
    });
}


void print_params(Flash_fwd_params &params)
{
    std::cout << " params.q_head_stride " <<  params.q_head_stride << "\n";
    std::cout << " params.q_row_stride " << params.q_row_stride << "\n";
    std::cout << " params.q_batch_stride " << params.q_batch_stride << "\n";

    std::cout << "params.k_head_stride " << params.k_head_stride << "\n";
    std::cout << "params.k_row_stride " << params.k_row_stride << "\n";
    std::cout << "params.k_batch_stride " << params.k_batch_stride << "\n";

    std::cout << " params.v_head_stride " << params.v_head_stride << "\n";
    std::cout << " params.v_row_stride " << params.v_row_stride << "\n";
    std::cout << " params.v_batch_stride " << params.v_batch_stride << "\n";

    std::cout << " params.o_head_stride " << params.o_head_stride << "\n";
    std::cout << " params.o_row_stride " << params.o_row_stride << "\n";
    std::cout << " params.o_batch_stride " << params.o_batch_stride << "\n";


    std::cout << " params.block_table_batch_stride " << params.block_table_batch_stride << "\n";
    std::cout << " params.page_block_size " << params.page_block_size << "\n";



}


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
  int num_splits) {
    
    // FLASHATTNLIB_BEGIN_FUNC

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    const int head_size_rounded = round_multiple(head_size, 32);

    Flash_fwd_params params;

    set_params_fprop_strided(params,
      batch_size,
      seqlen_q, seqlen_k,
      seqlen_q_rounded, seqlen_k_rounded,
      num_heads, num_heads_k,
      head_size, head_size_rounded,
      const_cast<void *>(q_ptr),
      const_cast<void *>(k_ptr), 
      const_cast<void *>(v_ptr), 
      o_ptr,
      /*cu_seqlens_q_d=*/nullptr,
      /*cu_seqlens_k_d=*/nullptr,
      /*seqused_k=*/nullptr,
      /*p_ptr=*/return_softmax ? p_ptr : nullptr,
      /*softmax_lse=*/softmax_lse_ptr,
      /*p_dropout=*/p_dropout,
      softmax_scale,
      window_size_left,
      window_size_right,
      softcap,
      !is_fp16);

    params.num_splits = 1;
    // void * softmax_lse_accum_ptr = nullptr;
    // void * out_accum_ptr = nullptr;
    // set_params_splitkv(params, batch_size, num_heads,
    //   head_size, seqlen_k, seqlen_q,
    //   head_size_rounded, p_dropout, /*num_splits*/0, dprops, 
    //   softmax_lse_accum_ptr, out_accum_ptr);
    
    params.alibi_slopes_ptr = alibi_slopes_ptr;
    params.alibi_slopes_batch_stride = batch_size > 1 ? num_heads : 0;

    run_mha_fwd__(params, stream);

    // if (params.num_splits > 1) {
    //   HIP_CHECK(hipFree(softmax_lse_accum_ptr));
    //   HIP_CHECK(hipFree(out_accum_ptr));
    // }
}

void fmha_page_kvcache_fwd(
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
            bool is_rotary_interleaved, 
            bool is_fp16
) 
{
    // FLASHATTNLIB_BEGIN_FUNC

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    const int head_size_rounded = round_multiple(head_size, 32);

    Flash_fwd_params params;

    const float softcap = 0.0f;

    set_params_fprop_strided(params,
                    batch_size,
                    seqlen_q, seqlen_k,
                    seqlen_q_rounded, seqlen_k_rounded,
                    num_heads, num_heads_k,
                    head_size, head_size_rounded,
                    const_cast<void *>(q_ptr),
                    const_cast<void *>(kcache_ptr), 
                    const_cast<void *>(vcache_ptr), 
                    o_ptr,
                    /*cu_seqlens_q_d=*/nullptr,
                    /*cu_seqlens_k_d=*/nullptr,
                    /*seqused_k=*/nullptr,
                    /*p_ptr=*/nullptr,
                    /*softmax_lse=*/nullptr,
                    /*p_dropout=*/0.f,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                    softcap,
                    !is_fp16
                    );
                    
    params.k_batch_stride = page_block_size * num_heads_k * head_size;
    params.v_batch_stride = page_block_size * num_heads_k * head_size;

    params.block_table = (int *)block_table_ptr;
    params.block_table_batch_stride = max_cache_seq_k / page_block_size;
    params.page_block_size = page_block_size;

    params.rotary_cos_ptr = rotary_cos_ptr;
    params.rotary_sin_ptr = rotary_sin_ptr;
    params.is_rotary_interleaved = is_rotary_interleaved;

    params.cache_batch_idx = (int *)cache_batch_idx_ptr;
    params.cu_seqlens_k = (int *)cache_seqlens_k_ptr;
    params.is_seqlens_k_cumulative = !cache_seqlens_k_ptr;

    params.num_splits = 1;
    
    // hipDeviceProp_t dprops;
    // hipGetDeviceProperties(&dprops, 0);
    // void * softmax_lse_accum_ptr = nullptr;
    // void * out_accum_ptr = nullptr;
    // set_params_splitkv(params, batch_size, num_heads,
    //   head_size, seqlen_k, seqlen_q,
    //   head_size_rounded, 0.f, /*num_splits*/0, &dprops, 
    //   softmax_lse_accum_ptr, out_accum_ptr);


    // printf("aaa\n");
    // print_params(params);
    // params.softmax_lse_ptr = softmax_lse;



    run_mha_fwd__(params, stream);

    // if (params.num_splits > 1) {
    //   HIP_CHECK(hipFree(softmax_lse_accum_ptr));
    //   HIP_CHECK(hipFree(out_accum_ptr));
    // }



    // params.k_head_stride = k_head_stride;
    // params.k_row_stride = k_row_stride;
    // params.k_batch_stride = k_batch_stride;
    // FLASHATTNLIB_END_FUNC
}
#ifdef __cplusplus
}
#endif