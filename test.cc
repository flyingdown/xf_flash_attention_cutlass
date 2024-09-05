#include <iostream>
#include "flash_hip.h"
#include "static_switch.h"
#include "paged_attn.h"



using half_t = _Float16;

void mha_fwd() {

    hipDeviceProp_t dprops;
    hipGetDeviceProperties(&dprops, 0);

    const int batch_size = 2;
    int seqlen_q = 128;
    int num_heads = 6;
    const int head_size_og = 128;
    const int seqlen_k = 128;
    const int num_heads_k = 6;
    const float softcap = 0.f;
    const float p_dropout = 0.f;
    void* alibi_slopes_ptr = nullptr;
    const float softmax_scale = 0.08838834;
    bool is_causal = false;
    int window_size_left = -1;
    int window_size_right = -1;

    if (softcap > 0.f) { ASSERT_CHECK(p_dropout == 0.f); }
    // ASSERT_CHECK(head_size_og % 8 == 0);
    const bool return_softmax = false;

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_ptr) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = false;
    // const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size_og % 8 == 0 && !alibi_slopes_ptr;
    // const int ngroups = num_heads / num_heads_k;
    // if (seqlenq_ngroups_swapped) {
    //     q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
    //     seqlen_q = ngroups;
    //     num_heads = num_heads_k;
    // }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);
    const int head_size_rounded = round_multiple(head_size, 32);

    void* q_ptr, * k_ptr, * v_ptr, * o_ptr;
    HIP_CHECK(hipMalloc(&q_ptr, batch_size * seqlen_q * num_heads * head_size_og * sizeof(half_t)));
    HIP_CHECK(hipMalloc(&k_ptr, batch_size * seqlen_k * num_heads_k * head_size_og * sizeof(half_t)));
    HIP_CHECK(hipMalloc(&v_ptr, batch_size * seqlen_k * num_heads_k * head_size_og * sizeof(half_t)));
    HIP_CHECK(hipMalloc(&o_ptr, batch_size * seqlen_q * num_heads * head_size_og * sizeof(half_t)));

    void * softmax_lse_ptr;
    HIP_CHECK(hipMalloc(&softmax_lse_ptr, batch_size * seqlen_q * num_heads * sizeof(float)));
    void * p_ptr = nullptr;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        ASSERT_CHECK(p_dropout > 0.0f);
        HIP_CHECK(hipMalloc(&p_ptr, batch_size * seqlen_q_rounded * num_heads * seqlen_k_rounded * sizeof(float)));
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    fmha_fwd(q_ptr, k_ptr, v_ptr, o_ptr, 
        alibi_slopes_ptr, seqlen_q, seqlen_k, batch_size, num_heads, num_heads_k, head_size,
        p_dropout, stream, &dprops, softmax_scale, p_ptr, softmax_lse_ptr,
        window_size_left, window_size_right, softcap, return_softmax, true);
    
}

int main() {
    mha_fwd();
}