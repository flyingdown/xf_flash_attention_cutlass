#include "src/flash_hip.h"
#include "src/static_switch.h"
#include <cutlass/numeric_types.h>

#define ASSERT_CHECK(__cond)                             \
      do {                                               \
        const bool __cond_var = (__cond);                \
        if (!__cond_var) {                               \
          ::std::string __err_msg = ::std::string("`") + \
                #__cond + "` check failed at " +         \
                __FILE__ + ":" +                         \
                ::std::to_string(__LINE__);              \
          throw std::runtime_error(__err_msg);           \
        }                                                \
      } while (0)


#ifdef __cplusplus
extern "C" {
#endif

// static thread_local std::unique_ptr<char[]> flash_attn_err_msg;


// void flash_attn_set_error(const char *msg) {
//   if (msg == nullptr || *msg == '\0') {
//     msg = "unknown error";
//   }

//   auto n = strlen(msg);
//   std::unique_ptr<char[]> new_err_msg(new char[n+1]);
//   std::strcpy(new_err_msg.get(), msg);
//   flash_attn_err_msg = std::move(new_err_msg);
// }

// const char *flash_attn_error() {
//   return flash_attn_err_msg.get();
// }

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// #define FLASHATTNLIB_BEGIN_FUNC try {
// #define FLASHATTNLIB_END_FUNC } catch (::std::exception &__e) { flash_attn_set_error(__e.what()); return false; } catch (...) { flash_attn_set_error(nullptr); return false; }

// #define CHECK_FWD_EXECTUABLE(__seqlen_q, __seqlen_k)                     \
//       ASSERT_CHECK(batch_size > 0);                                      \
//       ASSERT_CHECK(head_size % 8 == 0);                                  \
//       ASSERT_CHECK(head_size <= 256);                                    \
//       ASSERT_CHECK(num_heads % num_heads_k == 0);                        


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
                      bool is_causal,
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
    // params.q_row_stride = q_row_stride;
    // params.k_row_stride = k_row_stride;
    // params.v_row_stride = v_row_stride;
    // params.q_head_stride = q_head_stride;
    // params.k_head_stride = k_head_stride;
    // params.v_head_stride = v_head_stride;
    params.o_ptr = out;
    // params.o_row_stride = o_row_stride;
    // params.o_head_stride = o_head_stride;
    // params.varlen_padded_input = varlen_padded_input;

    // if (cu_seqlens_q_d == nullptr) {
    //     params.q_batch_stride = q_batch_stride;
    //     params.k_batch_stride = k_batch_stride;
    //     params.v_batch_stride = v_batch_stride;
    //     params.o_batch_stride = o_batch_stride;
    // }

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
    ASSERT_CHECK(p_dropout < 1.f);

    params.is_causal = window_size_left < 0 && window_size_right == 0;
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;


}

void run_mha_fwd(Flash_fwd_params &params, hipStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                // printf("params.num_splits = %d !force_split_kernel = %d\n", params.num_splits, !force_split_kernel);
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
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
    
    const int q_head_stride = head_size;
    const int q_row_stride = num_heads * head_size;
    const int q_batch_stride = seqlen_q * q_row_stride;
    
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
                    is_causal,
                    !is_fp16
                    );

    params.block_table = (int *)block_table_ptr;
    params.block_table_batch_stride = page_block_size;
    params.page_block_size = page_block_size;

    params.rotary_cos_ptr = rotary_cos_ptr;
    params.rotary_sin_ptr = rotary_sin_ptr;
    params.is_rotary_interleaved = is_rotary_interleaved;

    params.cache_batch_idx = (int *)cache_batch_idx_ptr;
    params.cu_seqlens_k = (int *)cache_seqlens_k_ptr;
    params.is_seqlens_k_cumulative = !cache_seqlens_k_ptr;

    run_mha_fwd(params, stream);


    // FLASHATTNLIB_END_FUNC
}

#ifdef __cplusplus
}
#endif